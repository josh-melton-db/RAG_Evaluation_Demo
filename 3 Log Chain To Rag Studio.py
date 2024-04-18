# Databricks notebook source
# DBTITLE 1,Databricks RAG Studio Installer
# MAGIC %run ./wheel_installer

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------

import os
import mlflow
from databricks import rag_studio, rag_eval, rag
import json
import html

### START: Ignore this code, temporary workarounds given the Private Preview state of the product
from mlflow.utils import databricks_utils as du
os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = "false"

def parse_deployment_info(deployment_info):
  browser_url = du.get_browser_hostname()
  message = f"""Deployment of {deployment_info.model_name} version {deployment_info.model_version} initiated.  This can take up to 15 minutes and the Review App & REST API will not work until this deployment finishes. 

  View status: https://{browser_url}/ml/endpoints/{deployment_info.endpoint_name}
  Review App: {deployment_info.rag_app_url}"""
  return message
### END: Ignore this code, temporary workarounds given the Private Preview state of the product

# COMMAND ----------

# MAGIC %run ./RAG_Experimental_Code

# COMMAND ----------

# MAGIC %md
# MAGIC # Configure the driver notebook 

# COMMAND ----------

# DBTITLE 1,Setup
############
# Specify the full path to the chain notebook & config YAML
############

# Assuming your chain notebook is in the current directory, this helper line grabs the current path, prepending /Workspace/
# Limitation: RAG Studio does not support logging chains stored in Repos
current_path = '/Workspace' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())

chain_notebook_file = "2_rag_chain"
chain_config_file = "2_rag_chain.yaml"
chain_notebook_path = f"{current_path}/{chain_notebook_file}"
chain_config_path = f"{current_path}/{chain_config_file}"

print(f"Saving chain from: {chain_notebook_path}, config from: {chain_config_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Log the chain

# COMMAND ----------

# DBTITLE 1,Log the model
############
# Log the chain to the Notebook's MLflow Experiment inside a Run
# The model is logged to the Notebook's MLflow Experiment as a run
############

logged_chain_info = rag_studio.log_model(code_path=chain_notebook_path, config_path=chain_config_path)

# Optionally, tag the run to save any additional metadata
with mlflow.start_run(run_id=logged_chain_info.run_id):
  mlflow.set_tag(key="rag_eval", value="roughdraft")

# Save YAML config params to the Run for easy filtering / comparison later(requires experimental import)
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
RagConfig(chain_config_path).experimental_log_to_mlflow_run(run_id=logged_chain_info.run_id)

print(f"MLflow Run: {logged_chain_info.run_id}")
print(f"Model URI: {logged_chain_info.model_uri}")

############
# If you see this error, go to your chain code and comment out all usage of `dbutils`
############
# ValueError: The file specified by 'code_path' uses 'dbutils' command which are not supported in a chain model. To ensure your code functions correctly, remove or comment out usage of 'dbutils' command.

# COMMAND ----------

# MAGIC %md
# MAGIC # Test the model locally & view the trace

# COMMAND ----------

# DBTITLE 1,Local Model Testing and Tracing
############
# Test the model locally
# This is the same input that the REST API will accept once deployed.
############

model_input = {
    "messages": [
        {
            "role": "user",
            "content": "Hello world!!",
        },
        
    ]
}

loaded_model = mlflow.langchain.load_model(logged_chain_info.model_uri)

# Run the model to see the output
# loaded_model.invoke(question)


############
# Experimental: View the trace
# ⚠️⚠️ 🐛🐛 Experimental features likely have bugs! 🐛🐛 ⚠️⚠️
############
json_trace = experimental_get_json_trace(loaded_model, model_input)

json_string = json.dumps(json_trace, indent=4)

# Escape HTML characters to avoid XSS or rendering issues
escaped_json_string = html.escape(json_string)

# Generate HTML with the escaped JSON inside <pre> and <code> tags
pretty_json_html = f"<html><body><pre><code>{escaped_json_string}</code></pre></body></html>"

# To use the HTML string in a context that renders HTML, 
# such as a web application or a notebook cell that supports HTML output
displayHTML(pretty_json_html)

# COMMAND ----------

# MAGIC %md
# MAGIC # Deploy the model to the Review App

# COMMAND ----------

# DBTITLE 1,Deploy the model
############
# To deploy the model, first register the chain from the MLflow Run as a Unity Catalog model.
############
# TODO: config file that does this for you
uc_catalog = "josh_melton"
uc_schema = "rag_eval"
model_name = "rag_eval_service_tickets"
uc_model_fqdn = f"{uc_catalog}.{uc_schema}.{model_name}" 

mlflow.set_registry_uri('databricks-uc')
uc_registered_chain_info = mlflow.register_model(logged_chain_info.model_uri, uc_model_fqdn)

# COMMAND ----------

############
# Deploy the chain to:
# 1) Review App so you & your stakeholders can chat with the chain & given feedback via a web UI.
# 2) Chain REST API endpoint to call the chain from your front end
# 3) Feedback REST API endpoint to pass feedback back from your front end.
############

deployment_info = rag_studio.deploy_model(uc_model_fqdn, uc_registered_chain_info.version)
print(parse_deployment_info(deployment_info))

# Note: It can take up to 15 minutes to deploy - we are working to reduce this time to seconds.

# COMMAND ----------

# DBTITLE 1,View deployments
############
# If you lost the deployment information captured above, you can find it using list_deployments()
############
# deployments = rag_studio.list_deployments()
for deployment in deployments:
  if deployment.model_name == uc_model_fqdn and deployment.model_version==uc_registered_chain_info.version:
    print(parse_deployment_info(deployment))

# COMMAND ----------

uc_catalog = "josh_melton"
uc_schema = "rag_eval"
model_name = "rag_eval_service_tickets"
uc_model_fqdn = f"{uc_catalog}.{uc_schema}.{model_name}"

from databricks.rag_studio import set_permissions
from databricks.rag_studio.entities import PermissionLevel

set_permissions(uc_model_fqdn, ["josh.melton@databricks.com"], PermissionLevel.CAN_MANAGE)

# COMMAND ----------

set_permissions(uc_model_fqdn, ["josh.melton@databricks.com"], PermissionLevel.CAN_REVIEW)

# COMMAND ----------


