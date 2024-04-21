# Databricks notebook source
# Set your catalog, source schema, table, and vector endpoint names. Optionally, pass target_schema to get_config() below
catalog = "default"
schema = "generated_rag_demo"
table = "field_service_tickets"
text_col_name = "issue_description"
text_id_name = "ticket_number"
vector_search_endpoint_name = "one-env-shared-endpoint-5"

# Set to True and provide the domain and categories if you'd like to generate new data 
generate_data_for_demo = False
text_domain = "Field service maintenance tickets for a diesel engine manufacturer"
category_ls = ["Overheated Turbocharger", "Fuel System Fault", "Worn Cylinder Head"]

from utils.demo import get_config, save_config, reset_tables, generate_source_data
config = get_config(catalog, schema, table, text_id_name, text_col_name, vector_search_endpoint_name)
save_config(dbutils, config)

# COMMAND ----------

# TODO: pull into demo.py
if generate_data_for_demo:
    from langchain.chat_models import ChatDatabricks
    chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)
    reset_tables(spark, catalog, schema, config["demo_config"]["target_schema"])
    generate_source_data(chat_model, text_domain, category_ls, text_col_name, text_id_name, catalog, schema, table, spark)
else:
    dbx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = dbx.notebookPath().get()
    folder_path = '/'.join(str(x) for x in notebook_path.split('/')[:-1])
    import pandas as pd
    df = spark.createDataFrame(pd.read_csv("/Workspace/"+folder_path+"/utils/example_data.csv"))
    df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{table}")
spark.read.table(f"{catalog}.{schema}.{table}").display()

# COMMAND ----------


