# Databricks notebook source
# MAGIC %pip install --upgrade openai==1.14.1
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/databricks-rag-studio/679d2f69-6d26-4340-b301-319a955c3ebd/databricks_rag_studio-0.0.0a2-py3-none-any.whl"
# MAGIC %pip install "https://ml-team-public-read.s3.us-west-2.amazonaws.com/wheels/rag-eval/releases/databricks_rag_eval-0.0.0a2-py3-none-any.whl"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from databricks import rag_studio, rag_eval, rag
import os
import json
from openai import OpenAI
import json
import concurrent.futures
from pyspark.sql.functions import concat, col, named_struct, lit, array
import requests
import mlflow
import json
import yaml

# Model to use for synthetic data generation
MODEL_SERVING_ENDPOINT_NAME = "databricks-dbrx-instruct"

# Connect the OpenAI client to Databricks Model Serving
API_ROOT = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
)
API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ["OPENAI_API_KEY"] = API_TOKEN
os.environ["OPENAI_BASE_URL"] = f"{API_ROOT}/serving-endpoints/"

client = OpenAI()

# COMMAND ----------

# MAGIC %md
# MAGIC # Get the chunks from the Vector Index
# MAGIC
# MAGIC Access the chain's configuration to get this data

# COMMAND ----------

rag_config = rag.RagConfig("2_rag_chain.yaml")

index_delta_table = rag_config.get("index_delta_table")
# parsed_docs_table = rag_config.get("parsed_docs_table")

chunks_df = spark.table(index_delta_table)
display(chunks_df)

# docs_df = spark.table(parsed_docs_table)
# display(docs_df)

chunk_text_key = rag_config.get("vector_search_schema").get("chunk_text")
chunk_id_key = rag_config.get("vector_search_schema").get("primary_key")
doc_uri_key =  rag_config.get("vector_search_schema").get("document_source")

# Load to JSON
json_df = chunks_df.toJSON().collect()

parsed_json_chunks = []
for row in json_df:
    parsed_row = json.loads(row)
    parsed_json_chunks.append(parsed_row)

# COMMAND ----------

# MAGIC %md
# MAGIC # Generate synthetic questions for the chunks

# COMMAND ----------

# source: https://thetechbuffet.substack.com/p/evaluate-rag-with-synthetic-data

PROMPT_TEMPLATE = """\
Your task is to formulate exactly 1 question from given context.

The question must satisfy the rules given below:
1.The question should make sense to humans even when read without the given context.
2.The question should be fully answered from the given context.
3.The question should be framed from a part of context that contains important information. It can also be from tables,code,etc.
4.The answer to the question should not contain any links.
5.The question should be of moderate difficulty.
6.The question must be reasonable and must be understood and responded by humans.
7.Do no use phrases like 'provided context', 'context', etc in the question
8.Avoid framing question using word "and" that can be decomposed into more than one question.
9.The question should not contain more than 10 words, make of use of abbreviation wherever possible.
    
context: {context}

Please return your output as a JSON as follows:

{{"question": "question for the chunk"}}"""

system_prompt = "You are an expert at understanding academic research papers.  You are also an expert at generating questions that a human would likely ask about specific content from academic research papers. You pride yourself on your ability to be realistic, yet a bit creative, and you know that a human will evaluate your output, so you put extra effort into following instructions exactly, including only outputing in JSON format as instructed.  You will lose your job if you don't output valid JSON."

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Test one chunk

# COMMAND ----------

# DBTITLE 1,Databricks OpenAI Chat Integration
def generate_question(chunk_row):
    prompt = PROMPT_TEMPLATE.format(context=chunk_row[chunk_text_key])
    response = client.chat.completions.create(
                model=MODEL_SERVING_ENDPOINT_NAME,
                messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                        "content": prompt
                }
                ],
                temperature=1.0
            )
    return json.loads(response.choices[0].message.content)

print(parsed_json_chunks[0])
print(generate_question(parsed_json_chunks[0]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate the questions

# COMMAND ----------

# Try 4 times if the question contains chunk
MAX_TRIES = 4
# Specify the number of threads to use
NUM_THREADS = 7

def process_one_chunk(row):
    # print(type(row))
    
    print(f"Trying for doc `{row[doc_uri_key]}` chunk_id {row[chunk_id_key]}")
    tries = 0
    try: 
        gen_questions = generate_question(row)
        # print(gen_questions)
        while "chunk" in gen_questions['question'] and tries < MAX_TRIES:
            tries = tries + 1
            print("wrote question with chunk in it, trying again")
            gen_questions = generate_question(row)    
                
        out_data = {f'{chunk_id_key}': row[chunk_id_key]}
        out_data['question'] = gen_questions['question']
        # print(gen_questions)
        return out_data
    except Exception as e:
        print(f"failed to parse output for doc `{row[doc_uri_key]}` chunk_id {row[chunk_id_key]}")

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    # Submit the processing function for each item in parallel
    futures = [executor.submit(process_one_chunk, row) for row in parsed_json_chunks]

    # Wait for all tasks to complete and get the results
    synthetic_data_raw = [future.result() for future in concurrent.futures.as_completed(futures)]

# Remove failed records
synthetic_data_raw = [x for x in synthetic_data_raw if x is not None]

# COMMAND ----------

# MAGIC %md
# MAGIC # Turn the synthetic questions into a RAG Studio Evaluation Set

# COMMAND ----------


# Join generated questions back to the chunks to get the chunk text
synthetic_questions_df = spark.read.json(spark.sparkContext.parallelize(synthetic_data_raw)).withColumnRenamed(chunk_id_key, chunk_id_key+"_")
synthetic_questions_df = synthetic_questions_df.join(chunks_df, chunks_df[chunk_id_key] == synthetic_questions_df[chunk_id_key+"_"], 'inner').drop(chunk_id_key+"_")

# Format into RAG Studio Evaluation Set schema
synthetic_eval_set = synthetic_questions_df.select(
    concat(lit("synthetic_"), col(chunk_id_key)).alias("request_id"),
    col("question").alias("request"),
    array(
        named_struct(
            lit("chunk_id"),
            col(chunk_id_key),
            lit("doc_uri"),
            col(doc_uri_key),
            lit("content"),
            col(chunk_text_key),
        )
    ).alias("expected_retrieval_context"),
)

display(synthetic_eval_set)

# Write to UC
uc_catalog = "rag"
uc_schema = "ericp_cummins"

synthetic_eval_set_table_uc_fqn = f"{uc_catalog}.{uc_schema}.`synthetic_eval_set`"
synthetic_eval_set.write.format("delta").mode("overwrite").saveAsTable(
    synthetic_eval_set_table_uc_fqn
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Initial evaluation of the chain using the data

# COMMAND ----------

uc_catalog = "rag"
uc_schema = "ericp_cummins"
model_name = "pdf_bot_cummins"
version = 1

model_fqn = f"{uc_catalog}.{uc_schema}.{model_name}"
model_uri = f"models:/{model_fqn}/{version}"

model_uri

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

############
# Run evaluation, logging the results to a sub-run of the chain's MLflow run
############
evaluation_results = rag_eval.evaluate(eval_set_table_name=synthetic_eval_set_table_uc_fqn, model_uri=model_uri, config=config_yml)


# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC # Load the questions/responses to the Review App

# COMMAND ----------

def query_chain(question):
  endpoint_name = "rag_studio_rag-ericp_cummins-pdf_bot_cummins"
  API_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

  model_input_sample = {
      "messages": [
          {
              "role": "user",
              "content": question,
          }
      ]
  }

  headers = {"Context-Type": "text/json", "Authorization": f"Bearer {API_TOKEN}"}

  response = requests.post(
      url=f"{API_ROOT}/serving-endpoints/{endpoint_name}/invocations", json=model_input_sample, headers=headers
  )

  return json.dumps(response.json())

# COMMAND ----------

# Specify the number of threads to use
num_threads = 7

# Create a ThreadPoolExecutor
with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit the processing function for each item in parallel
    futures = [executor.submit(query_chain, row['question']) for row in synthetic_data_raw]

    # Wait for all tasks to complete and get the results
    outputs = [future.result() for future in concurrent.futures.as_completed(futures)]


# COMMAND ----------

# MAGIC %md Run the `5 - Get Inference Table Logs` Notebook first

# COMMAND ----------

############
# UC FQN to the Inference Table
# You can find this from the chain's Model Serving Endpoint
############

uc_catalog = 'rag'
uc_schema = 'ericp_cummins'

# dbutils.widgets.text(
#     "inference_table_uc_fqn",
#     label="1. Inference Table UC table",
#     defaultValue="catalog.schema.inference_table",
# )
inference_table_uc_fqn = f"{uc_catalog}.{uc_schema}.`rag_studio-pdf_bot_cummins_payload`"
############
# Specify UC FQN to output the `request_log` table to
############
# dbutils.widgets.text(
#     "request_log_output_uc_fqn",
#     label="2a. Request Log output UC table",
#     defaultValue="catalog.schema.request_log",
# )
# request_log_output_uc_fqn = dbutils.widgets.get("request_log_output_uc_fqn")
request_log_output_uc_fqn = f"{uc_catalog}.{uc_schema}.`rag_studio-pdf_bot_cummins_request_log`"


############
# Specify UC FQN to output the `assessment_log` table to
############
# dbutils.widgets.text(
#     "assessment_log_output_uc_fqn",
#     label="2b. Assessment Log output UC table",
#     defaultValue="catalog.schema.assessment_log",
# )
# assessment_log_output_uc_fqn = dbutils.widgets.get("assessment_log_output_uc_fqn")
assessment_log_output_uc_fqn = f"{uc_catalog}.{uc_schema}.`rag_studio-pdf_bot_cummins_assessment_log`"



# COMMAND ----------

requests_df = spark.table(request_log_output_uc_fqn)
display(requests_df)

# COMMAND ----------


uc_catalog = 'rag'
uc_schema = 'ericp_cummins'
model_name = f"{uc_catalog}.{uc_schema}.pdf_bot_cummins"
rag_studio.enable_trace_reviews(model_name=model_name) # request_ids=["d36b3691-6376-46f7-90dc-9b8bdfdf637e"])
