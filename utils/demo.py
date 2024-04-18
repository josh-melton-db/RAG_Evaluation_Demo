import yaml
from databricks.sdk import WorkspaceClient
import re
import random
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import expr


# TODO: reduce number of args - use kwargs?
def get_config(catalog, source_schema, source_table_name, source_id_name, source_column_name, vs_endpoint,
               text_domain, category_ls, target_schema=None, num_docs=3, chunk_size_tokens=425,
               chunk_overlap_tokens=75, fmapi_endpoint="databricks-bge-large-en"):
    w = WorkspaceClient()
    username = w.current_user.me().user_name.replace('@', '_').replace('.', '_')
    source_table = f"{catalog}.{source_schema}.{source_table_name}"
    chunk_table = f"{catalog}.{source_schema}.{source_table_name}_chunked"
    chunk_id_column_name = "chunk_id"
    index_name = f"{catalog}.{source_schema}.{source_table_name}_index" # TODO: unique per user

    if not target_schema:
        target_schema = f"{username}_rag_eval"
    model_name = "rag_studio-{source_table_name}"
    uc_model_fqdn = f"{catalog}.{target_schema}.{model_name}"
    endpoint_name = f"rag_studio-{username}-{source_table_name}"
    synthetic_eval_set_table_uc_fqn = f"{catalog}.{target_schema}.`synthetic_eval_set`"
    inference_table_uc_fqn = f"{catalog}.{target_schema}.`rag_studio-{source_table_name}_payload`"
    request_log_output_uc_fqn = f"{catalog}.{target_schema}.`rag_studio-{source_table_name}_request_log`"
    assessment_log_output_uc_fqn = f"{catalog}.{target_schema}.`rag_studio-{source_table_name}_assessment_log`"

    return dict(
        k = num_docs, 
        chunk_size = chunk_size_tokens, 
        chunk_overlap = chunk_overlap_tokens, 
        fmapi_endpoint = fmapi_endpoint, 
        primary_key = chunk_id_column_name, 
        document_source = source_id_name,
        demo_config = dict(
            source_table = source_table,
            source_column_name = source_column_name,
            vector_search_endpoint_name = vs_endpoint,
            text_domain = text_domain,
            category_ls = category_ls,
            target_schema = target_schema,
            model_fqdn = uc_model_fqdn,
            endpoint_name = endpoint_name
        ),
    )

def save_config(dbutils, rag_config, fname="rag_config.yaml"):
    dbx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    notebook_path = dbx.notebookPath().get()
    folder_path = '/'.join(str(x) for x in notebook_path.split('/')[:-1])
    with open(f"/Workspace/{folder_path}/utils/{fname}", 'w') as outfile:
        yaml.dump(rag_config, outfile, default_flow_style=False)

def generate_category(chat_model, text_domain, category_ls):
    category_prompt = "Given the domain '{text_domain}', generate a classification or category for a piece of text in the domain. For example, if the domain was 'airplane pilot notes' a category might be 'control panel malfunction'. Come up with a category different from the following, if available: {category_ls}. Give only the category, in three words or less, no description, no filler, nothing about a response, ONLY THE CATEGORY:"
    category = chat_model.predict(category_prompt.format(text_domain=text_domain, category_ls=category_ls))
    category_ls.append(category)
    return category_ls

def generate_categories(chat_model, text_domain, category_ls):
    while len(category_ls) < 10:
        category_ls = generate_category(chat_model, text_domain, category_ls)
    cleaned_categories = [re.sub(r'[^a-zA-Z\s]', '', category) for category in category_ls]
    return list(set(cleaned_categories))

def generate_symptom(chat_model, domain, category, symptom_ls):
    symptom_prompt = "Generate a a symptom description for the category '{category}' within the domain '{domain}'. For example, if the domain was 'airplane pilot notes' and the category was 'control panel malfunction' a symptom set might be 'altitude gauge showing irregular readings'. Come up with a symptom different from the following, if available: {symptom_ls}. Give only the symptom, in ten words or less, no description, no filler, nothing like a numbered list, ONLY THE SYMPTOM:"
    symptom = chat_model.predict(symptom_prompt.format(category=category, domain=domain, symptom_ls=symptom_ls))
    symptom_ls.append(symptom)
    return symptom

def generate_symptoms(chat_model, categories, text_domain):
    symptoms_sets = {}
    for category in categories:
        symptom_ls = []
        num_documents = random.randint(5, 10)
        for _ in range(num_documents):
            symptom = generate_symptom(chat_model, text_domain, category, symptom_ls)
            symptom_ls.append(symptom)
        symptoms_sets[category] = list(set(symptom_ls))
    cleaned_symptoms = {}
    for category in symptoms_sets.keys():
        cleaned_symptoms[category] = [re.sub(r'[^a-zA-Z\s]', '', symptom_set) for symptom_set in symptoms_sets[category]]
    return cleaned_symptoms

def generate_document(chat_model, symptoms, category, document_ls):
    document_prompt = "Given the symptoms {symptoms}, generate a piece of text reporting the symptoms in detail. Indicate some relationship to {category}, although not directly. Indicate whether you think there is a potential resolution to the problem. Use an objective, fact-based, expert perspective. Give only the text, in one hundred words or less, no filler, nothing to indicate you're not the expert writing notes, don't explicitly say you were given a category, no lists, only detailed notes, reporting of the symptoms, and potentially next steps"
    document = chat_model.predict(document_prompt.format(symptoms=symptoms, category=category))
    document_ls.append(document)
    return document_ls

def generate_documents(chat_model, symptom_sets):
    data_dict = {}
    for category in symptom_sets.keys():
        document_ls = []
        for symptoms in symptom_sets[category]:
            doc = generate_document(chat_model, symptoms, category, document_ls)
        data_dict[category] = document_ls
    return data_dict

def create_df(data_dict, text_col_name):
    data = [(category, item) for category, items in data_dict.items() for item in items]
    schema = StructType([
        StructField("category", StringType(), True),
        StructField(text_col_name, StringType(), True)
    ])
    return spark.createDataFrame(data, schema=schema)

def reset_tables(spark, catalog, schema, tried=False):
    try:
        spark.sql(f"drop schema if exists {catalog}.{schema} CASCADE")
        spark.sql(f'''create schema {catalog}.{source_schema}''')
        except Exception as e:
    if 'NO_SUCH_CATALOG_EXCEPTION' in str(e) and not tried:
            spark.sql(f'create catalog {config["catalog"]}')
            reset_tables(spark, catalog, schema, True)
        else:
            raise

def generate_data(chat_model, text_domain, category_ls, text_col_name, text_id_name, catalog, schema, table):
    print('Generating data, this may take a couple minutes')
    categories = generate_categories(chat_model, text_domain, category_ls)
    symptoms = generate_symptoms(chat_model, categories, text_domain)
    documents = generate_documents(chat_model, symptoms) 
    df = create_df(documents, source_column_name)
    df = df.withColumn(text_id_name, expr("substring(md5(cast(rand() as string)), 1, 7)"))
    source_table = f"{catalog}.{schema}.{table}"
    df.write.saveAsTable(source_table)


