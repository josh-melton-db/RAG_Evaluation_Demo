# Databricks notebook source
# MAGIC %pip install -U --quiet pypdf==4.1.0 databricks-sdk langchain==0.1.13 tokenizers torch transformers
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
import pyspark.sql.functions as func
from pyspark.sql.types import MapType, StringType
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from pyspark.sql import Column
from pyspark.sql.types import *
from datetime import timedelta
from typing import List
import warnings

# Init workspace client
w = WorkspaceClient()

# COMMAND ----------

# Source and Target Configuration
source_catalog = "josh_melton"
source_schema = "rag_eval"
source_table_name = "field_service_tickets"
source_column_name = "issue_description" # TODO: should naming be consistent between these sections?
source_id_name = "ticket_number" # TODO: add in other metadata columns for the vector index to be filtered on (category, priority, etc)
source_table = f"{source_catalog}.{source_schema}.{source_table_name}"
target_table = f"{source_catalog}.{source_schema}.{source_table_name}_chunked"
index_name = f"{source_catalog}.{source_schema}.{source_table_name}_index"
vector_search_endpoint_name = "one-env-shared-endpoint-5"

# Vector Search Configuration
NUM_DOCS = 3
BGE_CONTEXT_WINDOW_LENGTH_TOKENS = 512
CHUNK_SIZE_TOKENS = 425
CHUNK_OVERLAP_TOKENS = 75
DATABRICKS_FMAPI_BGE_ENDPOINT = "databricks-bge-large-en"
FMAPI_EMBEDDINGS_TASK = "llm/v1/embeddings"
CHUNK_COLUMN_NAME = "chunked_text"
CHUNK_ID_COLUMN_NAME = "chunk_id"

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
# TODO: should we do the same runtime test as Eric's example to automate this bit?
@func.udf(returnType=ArrayType(StringType()), useArrow=True) # Comment out useArrow if using a runtime < 14.3
def split_char_recursive(content: str) -> List[str]:
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS
    )
    chunks = text_splitter.split_text(content)
    return [doc for doc in chunks]

# COMMAND ----------

@func.udf(returnType=ArrayType(StringType()), useArrow=True) # Comment out useArrow if using a runtime < 14.3
def split_char_recursive(content: str) -> List[str]:
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=CHUNK_SIZE_TOKENS, chunk_overlap=CHUNK_OVERLAP_TOKENS
    )
    chunks = text_splitter.split_text(content)
    return [doc for doc in chunks]

# COMMAND ----------

# DBTITLE 1,Chunk Docs
chunked_docs = (
    spark.read.table(source_table).limit(50) # TODO: remove limit
    .select("*", func.explode(split_char_recursive(func.col(source_column_name))).alias(CHUNK_COLUMN_NAME))
    .select("*", func.md5(func.col(CHUNK_COLUMN_NAME)).alias(CHUNK_ID_COLUMN_NAME))
)
chunked_docs.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(target_table)
spark.sql(f"ALTER TABLE {target_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)") # Required for Vector Search
chunked_docs.display()

# COMMAND ----------

# DBTITLE 1,Create Vector Search Index
# If index already exists, re-sync
try:
    w.vector_search_indexes.sync_index(index_name=index_name)
# Otherwise, create new index
except ResourceDoesNotExist as ne_error:
    w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=vector_search_endpoint_name,
        primary_key=CHUNK_ID_COLUMN_NAME,
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            embedding_source_columns=[
                EmbeddingSourceColumn(
                    embedding_model_endpoint_name=DATABRICKS_FMAPI_BGE_ENDPOINT,
                    name=CHUNK_COLUMN_NAME,
                )
            ],
            pipeline_type=PipelineType.TRIGGERED,
            source_table=target_table,
        ),
    )

# COMMAND ----------

def get_table_url(table_fqdn):
    split = table_fqdn.split(".")
    url = f"https://{dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()}/explore/data/{split[0]}/{split[1]}/{split[2]}"
    return url

print("Vector index:\n")
print(w.vector_search_indexes.get_index(index_name).status.message)

# COMMAND ----------

# TODO: write this out to the yaml automatically (include the commented out text in the next cell)
# TODO: write multiple versions to yaml automatically and iterate to show the value of the evaluation process?
rag_config_yaml = f"""
vector_search_endpoint_name: "{vector_search_endpoint_name}"
vector_search_index: "{index_name}"
index_delta_table: "{target_table}"
# These must be set to use the Review App to match the columns in your index
vector_search_schema:
  primary_key: {CHUNK_ID_COLUMN_NAME}
  chunk_text: {CHUNK_COLUMN_NAME}
  document_source: {source_id_name}
vector_search_parameters:
  k: {NUM_DOCS}
"""

print(rag_config_yaml)

# COMMAND ----------

# chunk_template: "`{chunk_text}`\n"
# chat_endpoint: "databricks-dbrx-instruct"
# chat_prompt_template: "You are a trusted assistant that helps answer questions about field service maintenance tickets based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this history and context, answer this question: {question}."
# chat_prompt_template_variables:
#   - "context"
#   - "question"
# chat_endpoint_parameters:
#   temperature: 0.01
#   max_tokens: 500
