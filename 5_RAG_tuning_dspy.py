# Databricks notebook source
# DBTITLE 1,Install Libraries
# MAGIC %pip install dspy-ai --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup
import dspy
from dspy.retrieve.databricks_rm import DatabricksRM

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

# Set up the models
lm = dspy.Databricks(model='databricks-mpt-7b-instruct', model_type='completions', api_key=token, 
                      api_base=url + '/serving-endpoints', max_tokens=1000)
judge = dspy.Databricks(model='databricks-dbrx-instruct', model_type='chat', api_key=token, 
                       api_base=url + '/serving-endpoints', max_tokens=200)
dspy.settings.configure(lm=lm)

# COMMAND ----------

# DBTITLE 1,Create DSPy Module
class RAG(dspy.Module):
    """Generates a response to the request using retrieved input for grounding"""
    def __init__(self):
        super().__init__()
        self.retrieve = DatabricksRM(
            databricks_index_name="default.generated_rag_demo.field_service_tickets_index", # TODO: dynamic
            databricks_endpoint=url, 
            databricks_token=token,
            columns=["issue_description", "category"],
            text_column_name="issue_description",
            docs_id_column_name="ticket_number",
        )
        self.CoT = dspy.ChainOfThought("request, context -> response") # TODO: pull this into its own module
    
    def forward(self, request):
        context = self.retrieve(request, query_type="text").docs
        return self.CoT(request=request, context=str(context))

# COMMAND ----------

# DBTITLE 1,Sample Unoptimized Response
test_question = "What's wrong with my turbocharger?" # TODO: remove
regular_RAG = RAG()
regular_RAG(request=test_question).response

# COMMAND ----------

# DBTITLE 1,Create Golden Dataset
golden_dataset = spark.sql("""
    SELECT request, response,
        CONCAT_WS('\nContext: ', transform(eval.retrieval_context, x -> x.content)) as context
    FROM `default`.`generated_rag_demo`.`synthetic_eval_set_eval_metrics` eval
    WHERE eval.response_metrics.llm_judged_relevant_to_question_and_context = 1
""").toPandas()

trainset = [dspy.Example(request=row['request'], response=row['response']).with_inputs('request', 'context')
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

rag = RAG()
scores = []
for x in trainset:
    pred = rag(x.request)
    score = metric(x, pred)
    scores.append(score)
print("Average score (out of 3):    ", sum(scores) / len(scores))

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

rag = RAG()
scores = []
for x in trainset:
    pred = optimized_rag(x.request)
    score = metric(x, pred)
    scores.append(score)
print("Average score (out of 3):    ", sum(scores) / len(scores))

# COMMAND ----------

# TODO: put dspy into langchain
# TODO: set chain for studio
