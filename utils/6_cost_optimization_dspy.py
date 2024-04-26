# Databricks notebook source
# MAGIC %pip install dspy-ai --upgrade -q
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import dspy

token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get() + '/serving-endpoints'

# Set up the various LMs
mpt = dspy.Databricks(model='databricks-mpt-7b-instruct', model_type='completions', api_key=token, api_base=url, max_tokens=1000)
mixtral = dspy.Databricks(model='databricks-mixtral-8x7b-instruct', model_type='chat', api_key=token, api_base=url, max_tokens=750)
dbrx = dspy.Databricks(model='databricks-dbrx-instruct', model_type='chat', api_key=token, api_base=url, max_tokens=500)
llama3 = dspy.Databricks(model='databricks-meta-llama-3-70b-instruct', model_type='chat', api_key=token, api_base=url, max_tokens=250)

# Use the cheapest LM by default
dspy.settings.configure(lm=mpt)

# COMMAND ----------

class CoT(dspy.Module):
    """Generates a response to the request using retrieved input for grounding"""
    def __init__(self, model):
        super().__init__()
        with dspy.context(lm=model):
            self.program = dspy.ChainOfThought("request, context -> response")
    
    def forward(self, request, context):
        return self.program(request=request, context=context)
    
lm_dictionary = {
    1: CoT(mpt),
    2: CoT(mixtral),
    3: CoT(dbrx)
}

# COMMAND ----------

class AssessResponse(dspy.Signature):
    """Assess the quality of an outline along the specified dimension."""
    request = dspy.InputField()
    response_to_assess = dspy.InputField()
    assessment_question = dspy.InputField()
    assessment_answer = dspy.OutputField(desc="Yes or No")

# COMMAND ----------

def eval_metric(response, request, trace=None):
    harmless_q = "Is the response harmless and unoffensive to the user?"
    specific_q = "Does the response provide very detailed, specific actions to take to identify or resolve the request?"
    value_add_q = "Does the response avoid simply repeating back the provided request and add value to the conversation?"
    with dspy.context(lm=llama3):
        harmless_eval =  dspy.Predict(AssessResponse)(request=request, response_to_assess=response,
                                                       assessment_question=harmless_q)
        specific_eval =  dspy.Predict(AssessResponse)(request=request, response_to_assess=response, 
                                                      assessment_question=specific_q)
        value_add_eval = dspy.Predict(AssessResponse)(request=request, response_to_assess=response, 
                                                      assessment_question=value_add_q)
    evals = [m.assessment_answer.lower() == 'yes' for m in [harmless_eval, specific_eval, value_add_eval]]
    score = sum(evals)

    if trace is not None: return score >= 2
    return score

# COMMAND ----------

def evaluate_response(response, request, level):
    if eval_metric(response, request) < 3:
        if level < 3:
            return dspy.Suggest(action="rerun", reason="Low score")
        else:
            return dspy.Assert(False, "Response is unsatisfactory")
    else:
        return dspy.Assert(True, "Response is satisfactory")

# COMMAND ----------

def process_request(request, level):
    response = lm_dictionary[level](request.request, request.context)
    evaluation = evaluate_response(response, request, level)
    if isinstance(evaluation, dspy.Suggest) and evaluation.action == "rerun":
        return process_request(request, level+1)
    return response

# COMMAND ----------

request = trainset[0]
response = lm_dictionary[1](request.request, request.context)
print(response)

# COMMAND ----------

process_request(trainset[0], 1)

# COMMAND ----------

from dspy.teleprompt import BootstrapFewShot

# Set up the optimizer: we want to "bootstrap" (i.e., self-generate) 1-shot examples of our CoT program.
config = dict(max_bootstrapped_demos=1, max_labeled_demos=2, max_errors=1)

# Optimize! Use the custom here. In general, the metric is going to tell the optimizer how well it's doing.
teleprompter = BootstrapFewShot(metric=eval_metric, **config)
optimized_cot = teleprompter.compile(CoT(), trainset=trainset)

# COMMAND ----------

dbrx.inspect_history(n=3)

# COMMAND ----------


