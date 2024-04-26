# Databricks notebook source
# DBTITLE 1,Install Libraries
# MAGIC %pip install dspy-ai --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Setup
import dspy

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() + '/serving-endpoints'

# Set up the LM
dbrx = dspy.Databricks(model='databricks-dbrx-instruct', model_type='chat', api_key=token, api_base=url, max_tokens=200)
mpt = dspy.Databricks(model='databricks-mpt-7b-instruct', model_type='completions', api_key=token, api_base=url, max_tokens=1000)
dspy.settings.configure(lm=mpt)

# COMMAND ----------

# DBTITLE 1,Create DSPy Module
class CoT(dspy.Module):
    """Generates a response to the request using retrieved input for grounding"""
    def __init__(self):
        super().__init__()
        self.program = dspy.ChainOfThought("request, context -> response")
    
    def forward(self, request, context):
        return self.program(request=request, context=context)

# COMMAND ----------

# DBTITLE 1,Sample Unoptimized Response
test_question = "What's wrong with my turbocharger?"
regular_CoT = CoT()
regular_CoT(request=test_question, context="").response

# COMMAND ----------

# DBTITLE 1,Create Golden Dataset
golden_dataset = spark.sql("""
    SELECT request, response,
        CONCAT_WS('\nContext: ', transform(eval.retrieval_context, x -> x.content)) as context
    FROM `default`.`generated_rag_demo`.`synthetic_eval_set_eval_metrics` eval
    WHERE eval.response_metrics.llm_judged_relevant_to_question_and_context = 1
""").toPandas()

trainset = [dspy.Example(request=row['request'], context=row['context'], response=row['response']).with_inputs('request', 'context')
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
    with dspy.context(lm=dbrx):
        harmless_q = "Is the response harmless and unoffensive to the user?"
        specific_q = "Does the response provide very detailed, specific actions to take to identify or resolve the request?"
        value_add_q = "Does the response avoid simply repeating back the provided request and add value to the conversation?"
        harmless_eval =  dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=harmless_q)
        specific_eval =  dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=specific_q)
        value_add_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, assessment_question=value_add_q)

    evals = ['yes' in m.assessment_answer.lower() for m in [harmless_eval, specific_eval, value_add_eval]]
    score = sum(evals)

    if trace is not None: return score >= 2
    return score >= 1

# COMMAND ----------

# DBTITLE 1,Optimize DSPy Module
from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 1-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=2, max_labeled_demos=2, max_errors=1)

# Optimize! Use the custom here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=trainset)

# COMMAND ----------

# DBTITLE 1,Inspect Judge History
dbrx.inspect_history(n=3)

# COMMAND ----------

# DBTITLE 1,Generate Optimized Response
optimized_output = optimized_cot(request=test_question, context="")
optimized_output.response

# COMMAND ----------

# optimized_cot(request=test_question, context="")

# COMMAND ----------

# model_path = "/tmp/model.json"
# optimized_cot.save(model_path)

# COMMAND ----------

# import mlflow
# import mlflow.pyfunc

# class DSPyModel(mlflow.pyfunc.PythonModel):
#     def load_context(self, context):
#         self.model = CoT().load(path=context.artifacts["model"])
    
#     def predict(self, request):
#         return self.model(request)

# # Log the model with mlflow
# logged_model = mlflow.pyfunc.log_model(artifact_path="model_path", python_model=DSPyModel())

# COMMAND ----------

# DBTITLE 1,Load and predict with MLflow model
# loaded_model = mlflow.pyfunc.load_model(logged_model.model_uri)
# loaded_model.predict("test")

# COMMAND ----------


