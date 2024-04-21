# Databricks notebook source
# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %run ./utils/wheel_installer 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./utils/RAG_Experimental_Code

# COMMAND ----------

# DBTITLE 1,Import Libraries
import os
import mlflow
from databricks import rag_studio, rag_eval, rag
import json
import html
from utils.demo import parse_deployment_info

# COMMAND ----------

# DBTITLE 1,Setup
# Specify the full path to the chain notebook & config YAML
# Assuming your chain notebook is in the current directory, this helper line grabs the current path, prepending /Workspace/
# Limitation: RAG Studio does not support logging chains stored in Repos
current_path = '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

chain_notebook_file = "2_set_rag_chain"
chain_config_file = "configs/rag_config.yaml"
chain_notebook_path = f"{current_path}/{chain_notebook_file}"
chain_config_path = f"{current_path}/{chain_config_file}"
rag_config = rag.RagConfig(chain_config_file)
print(f"Saving chain from: {chain_notebook_path}, config from: {chain_config_path}")

# COMMAND ----------

# DBTITLE 1,Log the chain
# Log the chain to the Notebook's MLflow Experiment inside a Run
logged_chain_info = rag_studio.log_model(code_path=chain_notebook_path, config_path=chain_config_path)
print(f"MLflow Run: {logged_chain_info.run_id}")
print(f"Model URI: {logged_chain_info.model_uri}")

# If you see this error, go to your chain code and comment out all usage of `dbutils`
# ValueError: The file specified by 'code_path' uses 'dbutils' command which are not supported in a chain model. To ensure your code functions correctly, remove or comment out usage of 'dbutils' command.

# COMMAND ----------

# DBTITLE 1,Local Model Testing
# Test the model locally, this is the same input that the REST API will accept once deployed.
model_input = {
    "messages": [{
        "role": "user",
        "content": "Hello world!!",
    },]
}

# Run the model to see the output
loaded_model = mlflow.langchain.load_model(logged_chain_info.model_uri)
loaded_model.invoke(model_input)

# COMMAND ----------

# DBTITLE 1,Register the model
# To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.
model_fqdn = rag_config.get("demo_config").get("model_fqdn")
mlflow.set_registry_uri('databricks-uc')
uc_registered_chain_info = mlflow.register_model(logged_chain_info.model_uri, model_fqdn)

# COMMAND ----------

# DBTITLE 1,Deploy the Chain
############
# Deploy the chain to:
# 1) Review App so you & your stakeholders can chat with the chain & given feedback via a web UI.
# 2) Chain REST API endpoint to call the chain from your front end
# 3) Feedback REST API endpoint to pass feedback back from your front end.
############

deployment_info = rag_studio.deploy_model(model_fqdn, uc_registered_chain_info.version)
print(parse_deployment_info(deployment_info))

# Note: It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

# DBTITLE 1,View deployments
# # If you lost the deployment information captured above, you can find it using list_deployments()
# deployments = rag_studio.list_deployments()
# for deployment in deployments:
#   if deployment.model_name == model_fqdn and deployment.model_version==uc_registered_chain_info.version:
#     print(parse_deployment_info(deployment))

# COMMAND ----------

rag_studio.enable_trace_reviews(model_name=model_fqdn, request_ids=["528b0a3b-2b25-4e5b-b954-55ace9826839"]) 

# COMMAND ----------

# DBTITLE 1,Manage Permissions
from databricks.rag_studio import set_permissions
from databricks.rag_studio.entities import PermissionLevel
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
user_name = w.current_user.me().user_name
set_permissions(model_fqdn, [user_name], PermissionLevel.CAN_MANAGE)
set_permissions(model_fqdn, [user_name], PermissionLevel.CAN_REVIEW)

# COMMAND ----------


