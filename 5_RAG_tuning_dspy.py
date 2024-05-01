# Databricks notebook source
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
lm = dspy.Databricks(model='databricks-mpt-7b-instruct', model_type='completions', api_key=token, 
                     api_base=url + '/serving-endpoints', max_tokens=1000)
judge = dspy.Databricks(model='databricks-dbrx-instruct', model_type='chat', api_key=token, 
                        api_base=url + '/serving-endpoints', max_tokens=200)
dspy.settings.configure(lm=lm) # TODO: set the cache folder

# COMMAND ----------

class CoT(dspy.Signature):
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
        self.retrieve = DatabricksRM(
            databricks_index_name=index_name,
            databricks_endpoint=url, 
            databricks_token=token,
            columns=["category", doc_id, chunk_column],
            text_column_name=chunk_column,
            docs_id_column_name=doc_id,
        )
        self.CoT = dspy.ChainOfThought("request, context -> response") # TODO: pull this into its own module

    def forward(self, request):
        context = self.retrieve(request, query_type="text").docs
        return self.CoT(request=request, context=str(context))

# COMMAND ----------

# DBTITLE 1,Sample Unoptimized Response
test_question = "What's wrong with my turbocharger?" # TODO: remove this cell
regular_RAG = RAG()
regular_RAG(request=test_question).response

# COMMAND ----------

from pyspark.sql.functions import expr

golden_dataset = (
    spark.read.table(synthetic_eval_set_table_uc_fqn+"_eval_metrics")
    .where(expr("response_metrics.llm_judged_relevant_to_question_and_context = 1"))
    .select("request", "response", 
            expr("concat_ws('; ', transform(synthetic_eval_set_eval_metrics.retrieval_context, x -> x.content))").alias("context"))
).toPandas()
trainset = [dspy.Example(request=row['request'], response=row['response']).with_inputs('request')
           for i, row in golden_dataset.iterrows()]

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
for x in trainset:
    pred = rag(x.request)
    score = metric(x, pred)
    scores.append(score)
raw_score = sum(scores) / len(scores)
print("Average score (out of 3):    ", raw_score)

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
optimizer = BootstrapFewShotWithRandomSearch(metric=metric, **config)
optimized_rag = optimizer.compile(RAG(), trainset=trainset)

# COMMAND ----------

# DBTITLE 1,Calculate Optimized Metric
rag = RAG()
scores = []
for x in trainset:
    pred = optimized_rag(x.request)
    score = metric(x, pred)
    scores.append(score)
optimized_score = sum(scores) / len(scores)
print("Optimized score (out of 3):  ", optimized_score) 
print("% Improvement over raw:      ", 100*(optimized_score - raw_score) / raw_score)

# COMMAND ----------

# TODO: save chain to cache folder, set chain, deploy to studio
# from langchain.chains import LLMChain
# from dspy.predict.langchain import LangChainModule
# langchain_module = LangChainModule(optimized_rag)
# chain = LLMChain(langchain_module)

# COMMAND ----------

# chain.run("What's wrong with the turbocharger?")
# rag.set_chain(chain)

# COMMAND ----------


