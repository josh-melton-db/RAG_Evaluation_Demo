# Databricks notebook source
# MAGIC %md
# MAGIC In this notebook, we'll take the data we curated with RAG Studio and fine tune an AI system using [DSPy](https://dspy-docs.vercel.app/), an open source framework for "programming - not prompting - language models". The aim is to eliminate brittle attachments to models or prompts by defining the process of our AI system, and allowing AI to optimize it for us. We can swap new models in and out for various pieces of the system and be confident our results are efficient and accurate without hand-tuning prompting techniques or relying on subjective, imprecise evaluations of prompting techniques.

# COMMAND ----------

# MAGIC %run ./utils/wheel_installer 

# COMMAND ----------

# DBTITLE 1,Install Libraries
# MAGIC %pip install dspy-ai --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup
from databricks import rag
rag_config = rag.RagConfig("configs/rag_config.yaml")
synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
index_name = rag_config.get("vector_search_index")
doc_id = rag_config.get("document_source_id")
chunk_column = rag_config.get("chunk_column_name")

# COMMAND ----------

# DBTITLE 1,Set DSPy Models
import dspy
from dspy.retrieve.databricks_rm import DatabricksRM

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# Set up the models
lm = dspy.Databricks(model="databricks-mpt-7b-instruct", model_type="completions", api_key=token, 
                     api_base=url + '/serving-endpoints', max_tokens=1000)
judge = dspy.Databricks(model="databricks-dbrx-instruct", model_type="chat", api_key=token, 
                        api_base=url + "/serving-endpoints", max_tokens=200)
dspy.settings.configure(lm=lm) # TODO: set a separate cache folder os.environ["DSP_NOTEBOOK_CACHEDIR"]

# COMMAND ----------

# DBTITLE 1,Define our Respond Signature
class Respond(dspy.Signature):
    """Generates a response to the request given some context"""
    request = dspy.InputField(desc="Request from an end user")
    context = dspy.InputField(desc="Context retrieved from vector search")
    response = dspy.OutputField(desc="Response to the user's question given the retrieved context")

# COMMAND ----------

# DBTITLE 1,Create DSPy Module
class RAG(dspy.Module):
    """Generates a response to the request using retrieved input for grounding"""
    def __init__(self):
        super().__init__()
        self.retrieve = DatabricksRM( # Set up retrieval from our vector search
            databricks_index_name=index_name,
            databricks_endpoint=url, 
            databricks_token=token,
            columns=["category", doc_id, chunk_column],
            text_column_name=chunk_column,
            docs_id_column_name=doc_id,
        )
        self.respond = dspy.ChainOfThought(Respond) # Responses will use chain of thought, i.e. "think this through step by step..."

    def forward(self, request):
        context = self.retrieve(request, query_type="text").docs
        return self.respond(request=request, context=str(context))

# COMMAND ----------

# DBTITLE 1,Create Datasets
from pyspark.sql.functions import expr

golden_dataset = (
    spark.read.table(synthetic_eval_set_table_uc_fqn+"_eval_metrics")
    .where(expr("response_metrics.llm_judged_relevant_to_question_and_context = 1"))
    .select("request", "response", 
            expr("concat_ws('; ', transform(synthetic_eval_set_eval_metrics.retrieval_context, x -> x.content))").alias("context"))
).toPandas()
trainset = [dspy.Example(request=row['request'], response=row['response']).with_inputs('request')
           for i, row in golden_dataset.iterrows() if i % 5 < 4]
testset = [dspy.Example(request=row['request'], response=row['response']).with_inputs('request')
           for i, row in golden_dataset.iterrows() if i % 5 == 4]

# COMMAND ----------

# DBTITLE 1,Define Assessment
class AssessResponse(dspy.Signature):
    """Assess the quality of an outline along the specified dimension."""
    request = dspy.InputField()
    response_to_assess = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

# COMMAND ----------

# DBTITLE 1,Define Metric
def metric(gold, pred, trace=None):
    request, response = gold.request, pred.response
    with dspy.context(lm=judge): # TODO: measure retrieval?
        harmless_q = "Is the response harmless and unoffensive to the user?"
        specific_q = "Does the response provide very detailed, specific actions to take to identify or resolve the request?"
        value_add_q = "Does the response avoid simply repeating back the provided request and add value to the conversation?"
        harmless_eval =  dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=harmless_q)
        specific_eval =  dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=specific_q)
        value_add_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=value_add_q)

    evals = ['yes' in m.assessment_answer.lower() for m in [harmless_eval, specific_eval, value_add_eval]]
    score = sum(evals)

    # if trace is not None: return score
    return score

# COMMAND ----------

# DBTITLE 1,Caclulate Baseline Metric
rag = RAG()
scores = []
for x in testset:
    pred = rag(x.request)
    score = metric(x, pred)
    scores.append(score)
raw_score = sum(scores) / len(scores)
print("Baseline average score (out of 3):    ", raw_score)

# COMMAND ----------

# DBTITLE 1,Calculate Llama Metric
llama = dspy.Databricks(model="databricks-meta-llama-3-70b-instruct", model_type="chat", api_key=token, 
                        api_base=url + "/serving-endpoints", max_tokens=200)
with dspy.context(lm=llama): 
    rag = RAG()
    llama_scores = []
    for x in testset:
        pred = rag(x.request)
        score = metric(x, pred)
        llama_scores.append(score)
llama_score = sum(llama_scores) / len(llama_scores)
print("Smart model average score (out of 3):    ", llama_score)

# COMMAND ----------

# DBTITLE 1,Inspect Judge History
judge.inspect_history(n=3)

# COMMAND ----------

# DBTITLE 1,Optimize DSPy Module
from dspy.teleprompt import BootstrapFewShotWithRandomSearch # Which optimizer you use depends on the number of examples

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) few-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=2, max_labeled_demos=2, max_errors=1, max_rounds=1) # TODO: optimal hyperparameters

# Use the proprietary training dataset you've collected and labelled with RAG Studio to
# optimize your model. The metric is going to tell the optimizer how well it's doing
optimizer = BootstrapFewShotWithRandomSearch(metric=metric, teacher_settings=dict({'lm': llama}), **config)
optimized_rag = optimizer.compile(student=RAG(), trainset=trainset)

# COMMAND ----------

# DBTITLE 1,Calculate Optimized Metric
scores = []
for x in testset:
    pred = optimized_rag(x.request)
    score = metric(x, pred)
    scores.append(score)
optimized_score = sum(scores) / len(scores)
print("Optimized score (out of 3):  ", optimized_score) 

# COMMAND ----------

# DBTITLE 1,Compare Metrics
print("% Improvement over raw:      ", 100*(optimized_score - raw_score) / raw_score)
print("% Improvement over llama:    ", 100*(optimized_score - llama_score) / llama_score)

# COMMAND ----------

# MAGIC %md
# MAGIC According to the metric above, the DSPy optimized MPT-7b system scores noticably higher than the baseline, or even the un-optimized Llama-3-70b model (which is 6x more expensive per output token). Alternatively, you could optimize a Llama-3-70b system to deliver significantly improved performance. Whether you aim for greater accuracy or reduced cost, you've used the data curated by RAG Studio to develop a proprietary improvement to the ROI of your AI systems!
