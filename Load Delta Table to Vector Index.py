# Databricks notebook source
# MAGIC %md
# MAGIC # Read Delta Table
# MAGIC
# MAGIC This is an example notebook that provides a **starting point** to build a data pipeline that loads, parses, chunks, and embeds Delta tables from a UC Catalog into a Databricks Vector Search Index

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install libraries & import packages

# COMMAND ----------

# MAGIC %pip install -U --quiet pypdf==4.1.0 databricks-sdk langchain==0.1.13
# MAGIC %pip install databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import io
from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceDoesNotExist
from databricks.sdk.service.vectorsearch import (
    EndpointType,
    DeltaSyncVectorIndexSpecRequest,
    VectorIndexType,
    EmbeddingSourceColumn,
    PipelineType,
    EndpointStatusState
)

from databricks.vector_search.client import VectorSearchClient
client = VectorSearchClient()

# Init workspace client
w = WorkspaceClient()

# Use optimizations if available
dbr_majorversion = int(spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion").split(".")[0])
if dbr_majorversion >= 14:
  spark.conf.set("spark.sql.execution.pythonUDF.arrow.enabled", True)

# COMMAND ----------

client.create_endpoint(name="mfg_rag_demo_test_2", endpoint_type="STANDARD")

# COMMAND ----------

# Check the status of the endpoint
endpoint_name = "mfg_rag_demo_test_2"
endpoint_status = client.get_endpoint(endpoint_name)

# Print the endpoint status
print(endpoint_status)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
source_table_fullname = "mfg_rag_demo.rag_tables.field_service_tickets"
# Where we want to store our index
vs_index_fullname = "mfg_rag_demo.rag_tables.field_service_vs_index"
VECTOR_SEARCH_ENDPOINT_NAME = "mfg_rag_demo_test_2"


client.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="TicketNumber",
    embedding_source_column='embedded_text', #The column containing our text
    embedding_model_endpoint_name='databricks-bge-large-en' #The embedding endpoint used to create the embeddings
  )

# COMMAND ----------

# DBTITLE 1,Databricks Vector Search Configuration
QUESTION = "what are the issues with Hydraulic pump?"

from databricks.vector_search.client import VectorSearchClient
from langchain_community.vectorstores import DatabricksVectorSearch
from langchain_community.embeddings import DatabricksEmbeddings
import os

index_name = "mfg_rag_demo.rag_tables.field_service_vs_index"

# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(f"Test embeddings: {embedding_model.embed_query(QUESTION)[:20]}...")

def get_retriever(persist_dir: str = None):
    vs_index = client.get_index(
        endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
        index_name=index_name
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="embedded_text", embedding=embedding_model
    )
    return vectorstore.as_retriever(search_kwargs={"k": 3, "score_threshold": 0.5})


# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.get_relevant_documents(QUESTION)
print(f"{len(similar_documents)} relevant documents found")
print(f"Sample: {similar_documents[0]}")

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)
print(f"Test chat model: {chat_model.predict(QUESTION)}")

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks

TEMPLATE = """You are an expert in solving engineering problems, and will be asked a question from a non-expert. Sound professional.
Use only the following pieces of context to answer the question at the end, and suggest 3 other similar issues that the user may be interested in:
{context}
Question: {question}
Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=chat_model, # which model to "face" the user
    retriever=get_retriever(), # how to retrieve information using vector searchb
    chain_type_kwargs={"prompt": prompt}, # prompt including the system template we defined above
    chain_type="stuff"
)

# COMMAND ----------

import langchain
langchain.debug = False #switch to True to see the chain details and the full prompt being sent
question = {"query": "What are the major issues with Hydraulic Press about?"}
answer = chain.run(question)
print(answer)

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = "mfg_rag_demo.rag_tables.mfg_rag_demo_chatbot_model"


# COMMAND ----------

with mlflow.start_run(run_name="mfg_rag_demo_run") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever,
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature,
    )

# COMMAND ----------

# Create or update serving endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput, ServedModelInputWorkloadSize

serving_endpoint_name = "mfg_rag_demo_chatbot_model_3"

w = WorkspaceClient()
endpoint_config = EndpointCoreConfigInput(
    name=serving_endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=model_name,
            model_version=2,
            workload_size=ServedModelInputWorkloadSize.SMALL,
            scale_to_zero_enabled=True,
            environment_vars={
                "DATABRICKS_TOKEN": "dapi31dc1a1011e7f54d6dc75a77965aae9a"
            }
        )
    ]
)

aexisting_endpoint = next(
    (e for e in w.serving_endpoints.list() if e.name == serving_endpoint_name), None
)



# COMMAND ----------

host = "https://" + spark.conf.get("spark.databricks.workspaceUrl")

serving_endpoint_url = f"{host}/ml/endpoints/{serving_endpoint_name}"
w.serving_endpoints.create_and_wait(name=serving_endpoint_name, config=endpoint_config)
    
displayHTML(f'Your Model Endpoint Serving is now available. Open the <a href="/ml/endpoints/{serving_endpoint_name}">Model Serving Endpoint page</a> for more details.')
