# Databricks notebook source
# MAGIC %run ./utils/wheel_installer 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,PySpark DataFrame Utilities
from pyspark.sql import DataFrame, SparkSession, window, functions as F, types as T
from typing import Optional, Tuple
from databricks import rag_studio, rag_eval, rag

# COMMAND ----------

# DBTITLE 1,Configure table names
demo_config = rag.RagConfig("configs/rag_config.yaml").get("demo_config")
inference_table_uc_fqn = demo_config.get("inference_table_uc_fqn")
request_log_output_uc_fqn = demo_config.get("request_log_output_uc_fqn")
assessment_log_output_uc_fqn = demo_config.get("assessment_log_output_uc_fqn")

# COMMAND ----------

# DBTITLE 1,Unpack the payloads
# Unpack the payloads
payload_df = spark.table(inference_table_uc_fqn)
request_logs, raw_assessment_logs = unpack_and_split_payloads(payload_df)

# The Review App logs every user interaction with the feedback widgets to the inference table - this code de-duplicates them
deduped_assessments_df = dedup_assessment_logs(raw_assessment_logs, granularity="hour")
deduped_assessments_df.write.format("delta").mode("overwrite").saveAsTable(assessment_log_output_uc_fqn)
display(request_logs)

# COMMAND ----------

display(deduped_assessments_df)

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


