# Databricks notebook source
# Fill in your custom text domain. If you'd like, you can add categories for the domain,
# otherwise we will generate 10 categories for you
text_domain = "Field service maintenance tickets for a diesel engine manufacturer"
category_ls = ["Turbocharger Failure", "Fuel System Fault"]

# COMMAND ----------

category_prompt = "Given the domain '{text_domain}', generate a classification or category for a piece of text in the domain. For example, if the domain was 'airplane pilot notes' a category might be 'control panel malfunction'. Come up with a category different from the following, if available: {category_ls}. Give only the category, in three words or less, no description, no filler, nothing about a response, ONLY THE CATEGORY:"

# COMMAND ----------

from langchain.chat_models import ChatDatabricks
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 200)

def generate_category(text_domain, category_ls):
    category = chat_model.predict(category_prompt.format(text_domain=text_domain, category_ls=category_ls))
    category_ls.append(category)
    return category

while len(category_ls) < 10:
    generate_category(text_domain, category_ls)

# COMMAND ----------

import re
cleaned_categories = [re.sub(r'[^a-zA-Z\s]', '', category) for category in category_ls]
print(cleaned_categories)

# COMMAND ----------

symptom_prompt = "Generate a a symptom description for the category '{category}' within the domain '{domain}'. For example, if the domain was 'airplane pilot notes' and the category was 'control panel malfunction' a symptom set might be 'altitude gauge showing irregular readings'. Come up with a symptom different from the following, if available: {symptom_ls}. Give only the symptom, in ten words or less, no description, no filler, nothing like a numbered list, ONLY THE SYMPTOM:"

# COMMAND ----------

import random

# TODO: add multithreading
def generate_symptom(domain, category, symptom_ls):
    symptom = chat_model.predict(symptom_prompt.format(category=category, domain=domain, symptom_ls=symptom_ls))
    symptom_ls.append(symptom)
    return symptom

symptoms_sets = {}
for category in cleaned_categories:
    symptom_ls = []
    num_documents = random.randint(5, 10)
    for _ in range(num_documents):
        generate_symptom(text_domain, category, symptom_ls)
    symptoms_sets[category] = symptom_ls
symptoms_sets

# COMMAND ----------

cleaned_symptoms = {}
for category in symptoms_sets.keys():
    cleaned_symptoms[category] = [re.sub(r'[^a-zA-Z\s]', '', symptom_set) for symptom_set in symptoms_sets[category]]
print(cleaned_symptoms)

# COMMAND ----------

document_prompt = "Given the symptoms {symptoms}, generate a piece of text reporting the symptoms in detail. Indicate some relationship to {category}, although not directly. Indicate whether you think there is a potential resolution to the problem. Use an objective, fact-based, expert perspective. Give only the text, in one hundred words or less, no filler, nothing to indicate you're not the expert writing notes, don't explicitly say you were given a category, no lists, only detailed notes, reporting of the symptoms, and potentially next steps"

# COMMAND ----------

def generate_document(symptoms, category):
    document = chat_model.predict(document_prompt.format(symptoms=symptoms, category=category))
    document_ls.append(document)
    return document

# TODO: add multithreading 
data_dict = {}
for category in cleaned_symptoms.keys():
    document_ls = []
    for symptoms in cleaned_symptoms[category]:
        generate_document(symptoms, category)
    data_dict[category] = document_ls
data_dict

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType
data = [(category, item) for category, items in data_dict.items() for item in items]
schema = StructType([
    StructField("Category", StringType(), True),
    StructField("Ticket", StringType(), True)
])
df = spark.createDataFrame(data, schema=schema)
df.display()

# COMMAND ----------

source_catalog = "josh_melton"
source_schema = "rag_eval_generated"
source_table_name = "field_service_tickets"
source_table = f"{source_catalog}.{source_schema}.{source_table_name}"
df.write.saveAsTable(source_table)

# COMMAND ----------


