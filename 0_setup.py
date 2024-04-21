# Databricks notebook source
# Set your catalog, source schema, table, and vector endpoint names. Optionally, pass target_schema to get_config() below
catalog = "default"
schema = "generated_rag_demo"
table = "field_service_tickets"
text_col_name = "issue_description"
text_id_name = "ticket_number"
vector_search_endpoint_name = "one-env-shared-endpoint-5"

from utils.demo import get_config, save_config, reset_tables, generate_source_data

config = get_config(catalog, schema, table, text_id_name, text_col_name, vector_search_endpoint_name)
save_config(dbutils, config)

# COMMAND ----------

# TODO: set to default, if default use a csv in the repo, if not default generate the data
# # Uncomment these lines to generate new data with your custom domain
# # If you'd like, you can add categories for the domain.
# # We will generate up to 10 categories for you
# text_domain = "Field service maintenance tickets for a diesel engine manufacturer"
# category_ls = ["Overheated Turbocharger", "Fuel System Fault", "Worn Cylinder Head"]
# from langchain.chat_models import ChatDatabricks
# chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)
# reset_tables(spark, catalog, schema, config["demo_config"]["target_schema"])
# generate_source_data(chat_model, text_domain, category_ls, text_col_name, text_id_name, catalog, schema, table, spark)
spark.read.table(f"{catalog}.{schema}.{table}").display()

# COMMAND ----------

