# Databricks notebook source
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/databricks-rag-studio/679d2f69-6d26-4340-b301-319a955c3ebd/databricks_rag_studio-0.0.0a2-py3-none-any.whl"
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/rag-eval/releases/databricks_rag_eval-0.0.0a2-py3-none-any.whl"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

eval_set = [
    {
        "request_id": "1328726",  # required, a unique identifer for this example
        "request": """Elevated exhaust temperature reading on LB #1 cylinder head. Valve adjustment was checked and found that the exhaust valve non transducer side was found to be at .011" (Spec = 0.032"). Checking the valve stem height it read 98.89mm which was at the limit for caution.""",  # optional, but could be a user's question if you turn this into a RAG system
        "expected_retrieval_context": [
            {
                "doc_uri": "1-114673703323",  # should have been inthere, was the best, but wasn't there
                "content": "Optional content of ticket",  # optional, this could be the text of the ticket to make debugging easier
            },
            {
                "doc_uri": "1-109904693511",  # required, this is the ID of the ground truth ticket
                "content": "Optional content of ticket",  # optional, this could be the text of the ticket to make debugging easier
            }
        ],
        # "expected_response": "Written answer", # optional, but could be a summary if you turn this into a RAG system
    }
]
eval_set

# COMMAND ----------

answer_sheet = [
    {
        "request_id": "1328726",  # matches the evaluation
        "app_version": "archana_initial_version_1",
        "response": "n/a",  # technically required for now, but can be set to nothing
        "retrieval_context": [
            {"doc_uri": "1-109904693511", "content": """This TSR is being raised to advise infant care team about the failure of a cylinder head seal on HSK78 engine located at Walkers Dinnington set A1
Coolant has been observed from under the cylinder head but work has yet to commence on repair.
Cylinder preliminary identification L5.
Solution Wed Apr 06 2022 12:39:39 EDT Summary: - Description: Closing per discussion"""},
            {"doc_uri": "1162554", "content": """Testing only
Case Complaint: Engine shutting down FC1791
SPT Group:013 Electrical Equipment"""},
            {"doc_uri": "780524", "content": "Optional content of ticket"},
            {"doc_uri": "1-107967853301", "content": "Optional content of ticket"},
            {"doc_uri": "1-109118716951", "content": "Optional content of ticket"},
            {"doc_uri": "1-106958669721", "content": "Optional content of ticket"},
            {"doc_uri": "1-109909729731", "content": "Optional content of ticket"},
            {"doc_uri": "1-110268819311", "content": "Optional content of ticket"},
            {"doc_uri": "1-113771179980", "content": "Optional content of ticket"},
            {"doc_uri": "1-109913710151", "content": "Optional content of ticket"},
            {"doc_uri": "1-109267229731", "content": "Optional content of ticket"},
            {"doc_uri": "1-109980827479", "content": "Optional content of ticket"},
            {"doc_uri": "1-110269059780", "content": "Optional content of ticket"},
            {"doc_uri": "1-108670920161", "content": "Optional content of ticket"},
            {"doc_uri": "1-114673703323", "content": "Optional content of ticket"},
            {"doc_uri": "1-113991246110", "content": "Optional content of ticket"},
            {"doc_uri": "1-113771179883", "content": "Optional content of ticket"},
            {"doc_uri": "1-109267229701", "content": "Optional content of ticket"},
            {"doc_uri": "1-114263116428", "content": "Optional content of ticket"},
            {"doc_uri": "1-108779882779", "content": "Optional content of ticket"},
            {"doc_uri": "1592976", "content": "Optional content of ticket"},
            {"doc_uri": "1-106270380941", "content": "Optional content of ticket"},
            {"doc_uri": "1-109488330164", "content": "Optional content of ticket"},
            {"doc_uri": "1-109980827685", "content": "Optional content of ticket"},
            {"doc_uri": "1-112825534911", "content": "Optional content of ticket"},
        ],
    },
    {
        "request_id": "1328726",  # matches the evaluation
        "app_version": "eric_cheated_v2",
        "response": "n/a",  # technically required for now, but can be set to nothing
        "retrieval_context": [
            {"doc_uri": "1-114673703323", "content": "Optional content of ticket"},
            {"doc_uri": "1-109904693511", "content": "Optional content of ticket"},
                        {"doc_uri": "780524", "content": "Optional content of ticket"},
            {"doc_uri": "1-107967853301", "content": "Optional content of ticket"},
            {"doc_uri": "1-109118716951", "content": "Optional content of ticket"},
            {"doc_uri": "1-106958669721", "content": "Optional content of ticket"},
            {"doc_uri": "1-109909729731", "content": "Optional content of ticket"},
            {"doc_uri": "1-110268819311", "content": "Optional content of ticket"},
            {"doc_uri": "1-113771179980", "content": "Optional content of ticket"},
            {"doc_uri": "1-109913710151", "content": "Optional content of ticket"},
            {"doc_uri": "1-109267229731", "content": "Optional content of ticket"},
            {"doc_uri": "1-109980827479", "content": "Optional content of ticket"},
            {"doc_uri": "1-110269059780", "content": "Optional content of ticket"},
            {"doc_uri": "1-108670920161", "content": "Optional content of ticket"},
            {"doc_uri": "1-114673703323", "content": "Optional content of ticket"},
            {"doc_uri": "1-113991246110", "content": "Optional content of ticket"},
            {"doc_uri": "1-113771179883", "content": "Optional content of ticket"},
            {"doc_uri": "1-109267229701", "content": "Optional content of ticket"},
            {"doc_uri": "1-114263116428", "content": "Optional content of ticket"},
            {"doc_uri": "1-108779882779", "content": "Optional content of ticket"},
            {"doc_uri": "1592976", "content": "Optional content of ticket"},
            {"doc_uri": "1-106270380941", "content": "Optional content of ticket"},
            {"doc_uri": "1-109488330164", "content": "Optional content of ticket"},
            {"doc_uri": "1-109980827685", "content": "Optional content of ticket"},
            {"doc_uri": "1-112825534911", "content": "Optional content of ticket"},
        ],
    }
]
answer_sheet

# COMMAND ----------

list = ['1-109904693511',
'1162554',
'780524',
'1-107967853301',
'1-109118716951',
'1-106958669721',
'1-109909729731',
'1-110268819311',
'1-113771179980',
'1-109913710151',
'1-109267229731',
'1-109980827479',
'1-110269059780',
'1-108670920161',
'1-114673703323',
'1-113991246110',
'1-113771179883',
'1-109267229701',
'1-114263116428',
'1-108779882779',
'1592976',
'1-106270380941',
'1-109488330164',
'1-109980827685',
'1-112825534911']

context = [{
            "doc_uri": f"{item}",  # required, this is the ID of the ground truth ticket
            "content": "Optional content of ticket",  # optional, this could be the text of the ticket to make debugging easier
        } for item in list]
context

# COMMAND ----------

answer_sheet_table = "rag.ericp_cummins.answer_sheet"
eval_set_table = "rag.ericp_cummins.eval_set"

answer_sheet_df = spark.read.json(spark.sparkContext.parallelize(answer_sheet))
answer_sheet_df.write.format("delta").option("mergeSchema", "true").mode(
    "overwrite"
).saveAsTable(answer_sheet_table)


eval_set_df = spark.read.json(spark.sparkContext.parallelize(eval_set))
eval_set_df.write.format("delta").option("mergeSchema", "true").mode(
    "overwrite"
).saveAsTable(eval_set_table)

# COMMAND ----------

import yaml
config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval_dbrx",
            "endpoint_name": "endpoints:/databricks-dbrx-instruct",
            "assessments": [
                # "harmful",
                # "faithful_to_context",
                # "relevant_to_question_and_context",
                # "relevant_to_question",
                # "answer_good",
                "context_relevant_to_question",
            ],
        }
    ]
}

config_yml = yaml.dump(config_json)
config_yml

# COMMAND ----------

from databricks import rag_eval

eval_results = rag_eval.evaluate(
    eval_set_table_name=eval_set_table,
    answer_sheet_table_name=answer_sheet_table,
    config=config_yml
)

# COMMAND ----------

# MAGIC %sql 
# MAGIC SELECT * FROM rag.ericp_cummins.eval_set_assessments
