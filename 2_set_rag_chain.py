# Databricks notebook source
# DBTITLE 1,Databricks Rag Studio Installer
# MAGIC %run ./utils/wheel_installer

# COMMAND ----------

# MAGIC %pip install --quiet --upgrade databricks-vectorsearch
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import packages
from langchain_community.chat_models import ChatDatabricks
from langchain_community.vectorstores import DatabricksVectorSearch
from databricks.vector_search.client import VectorSearchClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.runnable import RunnableLambda
from operator import itemgetter
from databricks import rag 

# COMMAND ----------

# DBTITLE 1,Multistage Chat Generator
############
# RAG Studio requires your chain to accept an array of OpenAI-formatted messages as a `messages` parameter. Schema: https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html#chatmessage
# These helper functions help parse the `messages` array
############
# Return the string contents of the most recent message from the user
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]


# Return the chat history, which is is everything before the last question
def extract_chat_history(chat_messages_array):
    return chat_messages_array[:-1]


############
# Get the configuration YAML
############
rag_config = rag.RagConfig("configs/rag_config.yaml")

############
# Connect to the Vector Search Index
############
vs_client = VectorSearchClient(disable_notice=True)
vs_index = vs_client.get_index(
    endpoint_name=rag_config.get("vector_search_endpoint_name"),
    index_name=rag_config.get("vector_search_index"),
)

############
# Turn the Vector Search index into a LangChain retriever
############
vector_search_as_retriever = DatabricksVectorSearch(
    vs_index,
    text_column=rag_config.get("demo_config").get("source_column_name"),
    columns=[
        "category",
        rag_config.get("chunk_column_name"),
        rag_config.get("document_source_id"),
    ],
).as_retriever(search_kwargs=rag_config.get("vector_search_parameters"))

############
# Required to:
# 1. Enable the RAG Studio Review App to properly display retrieved chunks
# 2. Enable evaluation suite to measure the retriever
############
rag.set_vector_search_schema(
    primary_key="chunk_id", 
    text_column=rag_config.get("chunk_column_name"), 
    doc_uri=rag_config.get("document_source_id"),  # Review App uses `doc_uri` to display chunks from the same document in a single view
)

############
# Method to format the docs returned by the retriever into the prompt
############
def format_context(docs):
    chunk_template = rag_config.get("chunk_template")
    chunk_contents = [chunk_template.format(chunk_text=d.page_content) for d in docs]
    return "".join(chunk_contents)

############
# Prompt Template for generation
############
prompt = PromptTemplate(
    template=rag_config.get("chat_prompt_template"),
    input_variables=rag_config.get("chat_prompt_template_variables"),
)

############
# FM for generation
############
model = ChatDatabricks(
    endpoint=rag_config.get("chat_endpoint"),
    extra_params=rag_config.get("chat_model_parameters"),
)

############
# RAG Chain
############
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

############
# Test the Chain
############
model_input_sample = {
    "messages": [
        {
            "role": "user",
            "content": "What are some issues that might come up with my turbocharger?",
        }
    ]
}
# print(chain.invoke(model_input_sample)) # comment this out before running the next notebook

############
# Tell RAG Studio about the chain - required for logging, but not local testing of this chain
############
rag.set_chain(chain)

# COMMAND ----------


