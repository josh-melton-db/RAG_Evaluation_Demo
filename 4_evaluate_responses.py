# Databricks notebook source
# MAGIC %run ./utils/wheel_installer 

# COMMAND ----------

# MAGIC %pip install --upgrade openai==1.14.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from databricks import rag_studio, rag_eval, rag
import os
import json
from openai import OpenAI
import concurrent.futures
from pyspark.sql.functions import concat, col, named_struct, lit, array
import requests
import mlflow
import yaml
from utils.demo import generate_questions

# COMMAND ----------

rag_config = rag.RagConfig("configs/rag_config.yaml")
chunk_table = rag_config.get("demo_config").get("chunk_table")
chunks_df = spark.table(chunk_table)
# display(chunks_df)

chunk_text_key = rag_config.get("chunk_column_name")
chunk_id_key = rag_config.get("chunk_id_column_name")
doc_uri_key =  rag_config.get("document_source_id")

# COMMAND ----------

# DBTITLE 1,Generate Questions
synthetic_data_raw = generate_questions(chunks_df, chunk_id_key, dbutils)
synthetic_data_raw[0]

# COMMAND ----------

# DBTITLE 1,Turn the synthetic questions into a RAG Studio Evaluation Set
# Join generated questions back to the chunks to get the chunk text
synthetic_questions_df = spark.read.json(spark.sparkContext.parallelize(synthetic_data_raw)).withColumnRenamed(chunk_id_key, chunk_id_key+"_")
synthetic_questions_df = synthetic_questions_df.join(chunks_df, chunks_df[chunk_id_key] == synthetic_questions_df[chunk_id_key+"_"], 'inner').drop(chunk_id_key+"_")

# Format into RAG Studio Evaluation Set schema
synthetic_eval_set = synthetic_questions_df.select(
    concat(lit("synthetic_"), col(chunk_id_key)).alias("request_id"),
    col("question").alias("request"),
    array(
        named_struct(
            lit("chunk_id"), col(chunk_id_key),
            lit("doc_uri"), col(doc_uri_key),
            lit("content"), col(chunk_text_key),
        )
    ).alias("expected_retrieval_context"),
)

display(synthetic_eval_set)
synthetic_eval_set_table_uc_fqn = rag_config.get("demo_config").get("synthetic_eval_set_table_uc_fqn")
synthetic_eval_set.write.format("delta").mode("overwrite").saveAsTable(synthetic_eval_set_table_uc_fqn)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Initial evaluation of the chain using the data

# COMMAND ----------

model_fqdn = rag_config.get("demo_config").get("model_fqdn")
version = 1 # TODO: how to set this dymanically
model_uri = f"models:/{model_fqdn}/{version}"

# COMMAND ----------

import yaml

############
# Currently, evaluation is slow with the Databricks provided LLM judge due to a limitation we are working to remove.  You can temporarily use any Model Serving endpoint to overcome this limitation, including DBRX.
############
config_json = {
    "assessment_judges": [
        {
            "judge_name": "databricks_eval_dbrx",
            "endpoint_name": "endpoints:/databricks-dbrx-instruct",
            "assessments": [
                "harmful",
                "faithful_to_context",
                "relevant_to_question_and_context",
                "relevant_to_question",
                "answer_good",
                "context_relevant_to_question",
            ],
        }
    ]
}

config_yml = yaml.dump(config_json)

# Run evaluation, logging the results to a sub-run of the chain's MLflow run
evaluation_results = rag_eval.evaluate(eval_set_table_name=synthetic_eval_set_table_uc_fqn, model_uri=model_uri, config=config_yml)

# COMMAND ----------

from utils.demo import load_review_qa
endpoint_name = rag_config.get("demo_config").get("endpoint_name")
load_review_qa(synthetic_data_raw, endpoint_name, dbutils)

# COMMAND ----------

# MAGIC %md Run the `5 - Get Inference Table Logs` Notebook first

# COMMAND ----------

inference_table_uc_fqn = rag_config.get("demo_config").get("inference_table_uc_fqn")
request_log_output_uc_fqn = rag_config.get("demo_config").get("request_log_output_uc_fqn")
assessment_log_output_uc_fqn = rag_config.get("demo_config").get("assessment_log_output_uc_fqn")

# COMMAND ----------

# Unpack the payloads
payload_df = spark.table(inference_table_uc_fqn)
request_logs, raw_assessment_logs = unpack_and_split_payloads(payload_df)

# The Review App logs every user interaction with the feedback widgets to the inference table - this code de-duplicates them
deduped_assessments_df = dedup_assessment_logs(raw_assessment_logs, granularity="hour")
deduped_assessments_df.write.format("delta").mode("overwrite").saveAsTable(assessment_log_output_uc_fqn)
display(request_logs)

# COMMAND ----------

(
    request_logs.write.format("delta")
    .option("mergeSchema", "true").mode("overwrite")
    .saveAsTable(request_log_output_uc_fqn)
)
(
    deduped_assessments_df.write.format("delta")
    .option("mergeSchema", "true").mode("overwrite")
    .saveAsTable(assessment_log_output_uc_fqn)
)

# COMMAND ----------

print(f"Wrote `request_log` to: {get_table_url(request_log_output_uc_fqn).replace('`', '')}")
print(f"Wrote `assessment_log` to: {get_table_url(assessment_log_output_uc_fqn).replace('`', '')}")

# COMMAND ----------

requests_df = spark.table(request_log_output_uc_fqn)
display(requests_df)

# COMMAND ----------

model_fqdn = rag_config.get("demo_config").get("model_fqdn")
rag_studio.enable_trace_reviews(model_name=model_fqdn) # request_ids=["d36b3691-6376-46f7-90dc-9b8bdfdf637e"])
