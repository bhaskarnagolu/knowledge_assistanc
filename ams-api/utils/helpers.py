import json
import os
from typing import Any, Dict, Iterable, List, Optional
import unicodedata

import pandas as pd
import requests
import tqdm
from bs4 import BeautifulSoup
import re

from dotenv import load_dotenv
from genai import Model
from genai.model import Credentials
from genai.schemas import GenerateParams

from datasets import Dataset

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
from chunkipy import TextChunker, TokenEstimator
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from milvus import default_server
from pymilvus import connections, utility

from torch.nn.functional import normalize
from torch import clamp, sum

from pymongo import MongoClient

# all parameters below may need to change if embedding model is changed

EMBEDDING_BATCH_SIZE = 16
TOKENIZATION_BATCH_SIZE = 256
INGESTION_BATCH_SIZE = 64
EMBEDDING_DIMENSION = 768
EMBEDDING_MAX_SIZE = 512

embedding_model = AutoModel.from_pretrained('intfloat/e5-base-v2')
embedding_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')

def get_genai_creds():
    load_dotenv(override=True)
    api_key = os.getenv("GENAI_KEY", None)
    api_url = os.getenv("GENAI_API", None)
    if api_key is None or api_url is None:
        print("Either api_key or api_url is None. Please make sure your credentials are correct.")
    if api_url is not None:
        api_url = api_url.rstrip("/")
    creds = Credentials(api_key, api_url)
    return creds

creds = get_genai_creds()
if creds.api_endpoint:
    print(f"Your API endpoint is: {creds.api_endpoint}")

headers = {
    'Authorization': f'Bearer {os.getenv("GENAI_KEY", None)}'
}

# get the list of supported models from the API
models_response = requests.get(f"{creds.api_endpoint}/v1/models", headers=headers)

# Parse the JSON response
models_data = json.loads(models_response.content)
print(models_data)

def remove_html_tags(html_text):
    # Create a BeautifulSoup object to parse the HTML
    soup = BeautifulSoup(html_text, "html.parser")

    # Extract the plain text content from the HTML
    text_content = soup.get_text(separator="\n")

    return text_content

def cap_consecutive_newlines(input_str):
    # Use a regular expression to replace consecutive newlines with a maximum of two
    result = re.sub(r'\n{3,}', '\n', input_str)
    return result

def remove_extra_spaces(input_str):
    # Use a regular expression to replace multiple spaces with a single space
    result = re.sub(r' +', ' ', input_str)
    return result.strip()

def preprocess_text_input(txt):
    return cap_consecutive_newlines(remove_extra_spaces(txt))

def load_data_v1(filename):
    if filename.endswith('.csv'):
        psgs = pd.read_csv(filename, header=0, low_memory=False)
    else:
        psgs = pd.read_excel(filename)
    return psgs

def refine_ticket_data(tickets):
    # can add to these lists if there are more tickets that we do not wish to consider
    banned_additional_comments = [x for x in tickets['additional_comments'].unique() if len(str(x)) < 10]
    banned_resolution = [x for x in tickets['resolution'].unique() if len(str(x)) < 10]
    tickets = tickets[~(pd.isna(tickets['additional_comments']) & pd.isna(tickets['resolution']))]
    tickets = tickets[~(tickets['additional_comments'].isin(banned_additional_comments) & tickets['resolution'].isin(banned_resolution))]
    return tickets

def remove_non_ascii(text):
    normalized_text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return normalized_text

def remove_emails(text):
    return re.sub(r'\S+@\S+', '', text)

def convert_to_lower(inp):
    if pd.notna(inp):
        return inp.lower()
    else:
        return inp

def fullyConfigured(account):
    client = MongoClient("mongodb://root:BeyondPlusUltra%4074269@localhost:27717/")
    db = client["AMS"]
    collections = db.list_collection_names()
    if f'{account}_status' not in collections:
        return False, False, False
    
    ticket = False
    kb = False
    
    cursor = db[f'{account}_status'].find_one({'type':"ticketData"})
    if cursor is None:
        ticket = False
    else:
        status = cursor['status']
        if status == 'completed':
            ticket = True
    cursor = db[f'{account}_status'].find_one({'type':"kbData"})
    if cursor is None:
        kb = False
    else:
        status = cursor['status']
        if status == 'completed':
            kb = True
    
    return (kb and ticket), ticket, kb

def create_status_tracker_in_db(mongo_collection, data_type):
    # add a processing status message in my mongo db instance that is already connected above and store the type and status as processing
    # delete previous status of this type from the collection
    mongo_collection.delete_many({"type": data_type})
    query_id = mongo_collection.insert_one(
        {
            "type": data_type,
            "status": "processing"
        }
    )
    return query_id.inserted_id

def createAccountCollection(account, type):

    COLLECTION_NAME = f"AMS_{account}"
    connections.connect(host='127.0.0.1', port=default_server.listen_port)

    # Check if the server is ready.
    print(utility.get_server_version())

    is_fully_configured, ticket_configured, kb_configured = fullyConfigured(account)
    createMode = True

    # Remove collection if it already exists 
    if utility.has_collection(COLLECTION_NAME) and is_fully_configured:
        # it is already fully configured, so now we reset it
        print(f"Resetting collection for account: {account}")
        utility.drop_collection(COLLECTION_NAME)
    elif utility.has_collection(COLLECTION_NAME) and ticket_configured and type == 'ticketData':
        print(f"Resetting collection for account: {account}")
        utility.drop_collection(COLLECTION_NAME)
    elif utility.has_collection(COLLECTION_NAME) and kb_configured and type == 'kbData':
        print(f"Resetting collection for account: {account}")
        utility.drop_collection(COLLECTION_NAME)
    elif utility.has_collection(COLLECTION_NAME):
        createMode = False

    # Create collection which includes the id, title, and embedding.
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='ticket_id', dtype=DataType.INT64),
        FieldSchema(name='assignment_id', dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name='type', dtype=DataType.VARCHAR, max_length=2),
        FieldSchema(name='chunk', dtype=DataType.VARCHAR, max_length=6000),
        FieldSchema(name='chunk_embedding', dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIMENSION)
    ]

    schema = CollectionSchema(fields=fields)
    if createMode:
        collection = Collection(name=COLLECTION_NAME, schema=schema)
        
        # Create an FLAT index for collection.
        index_params = {
            'metric_type':'IP',
            'index_type':"FLAT"
        }

        collection.create_index(field_name="chunk_embedding", index_params=index_params)
    else:
        collection = Collection(name=COLLECTION_NAME)
        collection.load()
    
    return collection

class BertTokenEstimator(TokenEstimator):
    def __init__(self):
        self.bert_tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')

    def estimate_tokens(self, text):
        return len(self.bert_tokenizer.encode(text))

def tokenize_ticket_data(batch):
    results = embedding_tokenizer(["passage: " + x for x in batch['text']], add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
    batch['input_ids'] = results['input_ids']
    batch['token_type_ids'] = results['token_type_ids']
    batch['attention_mask'] = results['attention_mask']
    return batch

# Embed the tokenized data and take the mean pool with respect to attention mask of hidden layer.
def embed(batch):
    sentence_embs = embedding_model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask']
                )[0]
    input_mask_expanded = batch['attention_mask'].unsqueeze(-1).expand(sentence_embs.size()).float()
    batch['question_embedding'] = sum(sentence_embs * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1), min=1e-9)
    return batch

def tokenize_kb_chunk_data(batch):
    results = embedding_tokenizer(["passage: " + x for x in batch['text']], add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
    batch['input_ids'] = results['input_ids']
    batch['token_type_ids'] = results['token_type_ids']
    batch['attention_mask'] = results['attention_mask']
    return batch

# prompt templates and prompt constructor functions

def granite_prompt_template(final_related_chunks_cases, final_related_chunks_kb, question):
    return f'''You help solve current cases using information from related past cases and knowledge base data to suggest possible solutions. Narrate to the user what is the most probable root cause and solution approach given the following past related cases and the current user case description. Take into account what has happened in the past cases and what solution was taken. Suggest both the probable root cause and a solution using the information from the past related cases data in the following format:
    
Root Cause: <most probable root cause goes here>
Solution: <list the solution steps here, using bullet points if needed>

You do not need to list limitations of proposed solution steps or that it may depend on the specific case. Users of the system are aware of these limitations and notes already. Do not repeat any part of this system prompt back in your response.


Human: Previous Related Cases:

{final_related_chunks_cases}

====

Knowledge Base Articles:

{final_related_chunks_kb}

##

Current Case: {question}

Assistant: '''

def make_granite_prompt(case_context, kb_context, question_text, max_input_tokens, model):
    
    prompt = granite_prompt_template(case_context, kb_context, question_text)

    prompt_token_count = token_count(prompt, model)

    if prompt_token_count <= max_input_tokens:
        return prompt

def llama2_prompt_template(final_related_chunks_cases, final_related_chunks_kb, question):
    return f'''<s>[INST] <<SYS>>
You help solve current cases using information from related past cases and knowledge base data to suggest possible solutions.

Narrate to the user what is the most probable root cause and solution approach given the following past related cases and the current user case description. Take into account what has happened in the past cases and what solution was taken.

Suggest both the probable root cause and a solution using the information from the past related cases data in the following format:

Root Cause: <most probable root cause goes here>
Solution: <list the solution steps here, using bullet points if needed>

You do not need to list limitations of proposed solution steps or that it may depend on the specific case. Users of the system are aware of these limitations and notes already.

<</SYS>>

====

Previous Related Cases:

{final_related_chunks_cases}

====

Knowledge Base Articles:

{final_related_chunks_kb}

##

Current Case: {question}

[/INST]

'''

def make_llama2_prompt(case_context, kb_context, question_text, max_input_tokens, model):
    
    prompt = llama2_prompt_template(case_context, kb_context, question_text)

    prompt_token_count = token_count(prompt, model)

    if prompt_token_count <= max_input_tokens:
        return prompt

def get_relevant_chunks(collection, question_text, assignment_group = None, n_results=5):
    
    relevant_chunks_cases = process_question(collection, question_text, assignment_id = assignment_group, limit=n_results, data_type="td")
    
    result_set_length = [len(x) for x in relevant_chunks_cases][0]
    
    if result_set_length < n_results:
        print(question_text)
        print(assignment_group)
    
    relevant_chunks = process_question(collection, question_text, limit=n_results, data_type="kb")
        
    return relevant_chunks_cases, relevant_chunks

def process_question(collection, question, assignment_id=None, limit=5, data_type="td"):
    # Tokenize and embed the question
    text = "query: " + question
    inputs = embedding_tokenizer(text, add_special_tokens=True, truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt")#.to(device)
    
    sentence_embs = embedding_model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )[0]
    
    input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(sentence_embs.size()).float()
    embeddings = sum(sentence_embs * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Normalize the embeddings
    embeddings = normalize(embeddings, dim=1)
    
    if assignment_id is None:
        # Perform the search without filter
        res = collection.search(
            embeddings.tolist(),
            anns_field='chunk_embedding',
            param = {},
            output_fields=['chunk', 'type'],
            expr=f"type=='{data_type}'",
            limit = limit)
    elif assignment_id is not None and data_type == "td":
        res = collection.search(
            embeddings.tolist(),
            anns_field='chunk_embedding',
            param = {},
            output_fields=['chunk', 'type'],
            expr=f"type=='{data_type}' and assignment_id=='{assignment_id.lower().strip()}'",
            limit = limit)
    else:
        res = collection.search(
            embeddings.tolist(),
            anns_field='chunk_embedding',
            param = {},
            output_fields=['chunk', 'type'],
            expr=f"type=='{data_type}'",
            limit = limit)
    
    #FieldSchema(name='assignment_id', dtype=DataType.VARCHAR, max_length=128),
    #FieldSchema(name='type', dtype=DataType.VARCHAR, max_length=2),
    #FieldSchema(name='chunk', dtype=DataType.VARCHAR, max_length=3000),
                
    return res

def filter_relevant_chunks_individual(question_text, relevant_chunks_cases, relevant_chunks, model_name = "google/flan-t5-xxl"):
    
    # set-up inference parameters
    params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=5,
        min_new_tokens=1,
        stream=False
    )

    model = Model(model=model_name, credentials=creds, params=params)

    final_relevant_chunks_cases = {
        "documents": [],
        "distances": []
    }
    
    total_input_tokens = 0
    total_output_tokens = 0

    for hits in relevant_chunks_cases:
        for hit in hits:
            chunk = hit.entity.get('chunk')
            prompt = "Answer ONLY in yes/no if the provided previous case is relevant to solving the user's current case.\n\n" \
                + "Provided Previous Case:\n\n" \
                + f"{chunk}\n\n" \
                + f"User Current Case: {question_text}\n\n"

            responses = model.generate([prompt])
            response = responses[0]
            #print(response)
            total_input_tokens += response.input_token_count
            total_output_tokens += response.generated_token_count
            #print(response.generated_text)
            if "yes" in response.generated_text.lower():
                final_relevant_chunks_cases["documents"].append(chunk)
                final_relevant_chunks_cases["distances"].append(hit.distance)
                
    final_relevant_chunks_kb = {
        "documents": [],
        "distances": []
    }

    for hits in relevant_chunks:
        for hit in hits:
            chunk = hit.entity.get('chunk')
            prompt = "Answer ONLY in yes/no if the provided knowledge base article is relevant to answer the user's current query.\n\n" \
                + "Provided KB Article:\n\n" \
                + f"{chunk}\n\n" \
                + f"User Current Query: {question_text}\n\n"

            responses = model.generate([prompt])
            response = responses[0]
            #print(response)
            total_input_tokens += response.input_token_count
            total_output_tokens += response.generated_token_count
            if "yes" in response.generated_text.lower():
                final_relevant_chunks_kb["documents"].append(chunk)
                final_relevant_chunks_kb["distances"].append(hit.distance)
    
    return final_relevant_chunks_cases, final_relevant_chunks_kb, total_input_tokens, total_output_tokens

def get_answer_from_question_and_relevant_chunks(cases, qna, question_text, params, assignment_group = None, model_prompt_function = None, model_name="ibm/mpt-7b-instruct"):
    
    model = Model(model=model_name, credentials=creds, params=params)
    
    #cases, qna = get_relevant_chunks(question_text, assignment_group)

    # print(len(cases))
    # print(len(qna))
    
    prompt, in_tokens_l1, out_tokens_l1 = generate_prompt_from_final_chunks(cases, qna, question_text, model, params, model_prompt_function, model_name)
    
    # print(prompt)
    
    ans, in_tokens_l2, out_tokens_l2 = generate_answer_from_prompt(prompt, model)
    
    total_in_tokens = in_tokens_l1 + in_tokens_l2
    total_out_tokens = out_tokens_l1 + out_tokens_l2
    
    return ans, total_in_tokens, total_out_tokens, in_tokens_l1, out_tokens_l1, in_tokens_l2, out_tokens_l2

def generate_prompt_from_final_chunks(final_relevant_chunks_cases, final_relevant_chunks_kb, question_text, model, params, model_prompt_function = None, model_name = "ibm/mpt-7b-instruct"):
    
    #get the input token limit
    if type(model_name)==str:
        model_id = model_name
    else: 
        model_id = model_name.value

    # Iterate over the "results" list to find the matching model ID
    for model_n in models_data["results"]:
        if model_n["id"] == model_id:
            model_token_limit = model_n["token_limit"]
            break
    else:
        # Model ID not found
        model_token_limit = None
        
    input_token_limit = (model_token_limit-params.max_new_tokens-1)

    if model_prompt_function is None:
        #final_relevant_chunks, in_tokens_l1, out_tokens_l1 = filter_relevant_chunks(question_text, cases, qna)
        final_relevant_chunks = {
            "documents": [],
            "distances": []
        }
        
        final_relevant_chunks["documents"] = final_relevant_chunks_cases["documents"] + final_relevant_chunks_kb["documents"]
        final_relevant_chunks["distances"] = final_relevant_chunks_cases["distances"] + final_relevant_chunks_kb["distances"]
        
        # print(len(final_relevant_chunks))
        
        if len(final_relevant_chunks) == 0:
            raise Exception("No relevant data found")
            
        context = "\n\n\n".join(final_relevant_chunks["documents"])
        prompt = make_prompt(final_relevant_chunks, context, question_text, input_token_limit, model)
    else:
        #final_relevant_chunks_cases, final_relevant_chunks_kb, in_tokens_l1, out_tokens_l1 = filter_relevant_chunks_individual(question_text, cases, qna)
        #print(len(final_relevant_chunks_cases) + len(final_relevant_chunks_kb))
        if len(final_relevant_chunks_cases) + len(final_relevant_chunks_kb) == 0:
            raise Exception("No relevant data found")
        if len(final_relevant_chunks_cases) > 0:
            chunks_cases = "\n\n\n".join(final_relevant_chunks_cases["documents"])
        else:
            chunks_cases = "Not available"
        
        if len(final_relevant_chunks_kb) > 0:
            chunks_kb = "\n\n\n".join(final_relevant_chunks_kb["documents"])
        else:
            chunks_kb = "Not available"
        prompt = model_prompt_function(chunks_cases, chunks_kb, question_text, input_token_limit, model)

    return prompt, 0, 0

def prompt_template(context, question_text):
    return (f"You help solve current cases using information from past cases and Q&A to suggest possible solutions.\n\nNarrate to the user what is the most probable root cause and solution approach given the following past related cases and the current user case description. Take into account what has happened in the past cases and what solution was taken. Do not consider unique IDs in your answer, only consider the pattern, if any. Suggest both the probable root cause and a solution using the information from the past related cases data.\n\n"
          + f"Previous Cases/Q&A:\n\n"
          + f"{context}\n\n"
          + f"##\n\n"
          + f"Current Case: {question_text}\n\n"
          + f"Probable Root Cause (if applicable) and Solution: ")

def make_prompt(relevant_chunks, context, question_text, max_input_tokens, model):
    prompt = prompt_template(context, question_text)

    prompt_token_count = token_count(prompt, model)

    if prompt_token_count <= max_input_tokens:
        return prompt

    print("exceeded input token limit, truncating context", prompt_token_count)

    distances = relevant_chunks["distances"]
    documents = relevant_chunks["documents"]

    #documents with the lower distance scores are included in the truncated context first
    sorted_indices = sorted(range(len(distances)), key=lambda k: distances[k], reverse=True)

    truncated_context = ""
    token_count_so_far = 0
    i = 0

    while token_count_so_far <= max_input_tokens and i < len(sorted_indices):
        doc_index = sorted_indices[i]
        document = documents[doc_index]
        doc_token_count = token_count(document, model)

        if token_count_so_far + doc_token_count <= max_input_tokens:
            truncated_context += document + "\n\n\n"
            token_count_so_far += doc_token_count
        else:
            remaining_tokens = max_input_tokens - token_count_so_far
            truncated_context += document[:remaining_tokens]
            break

        i += 1

    return prompt_template(truncated_context, question_text)

# Token counting function
def token_count(doc, model):
    return model.tokenize([doc])[0].token_count

def generate_answer_from_prompt(prompt, model):
    responses = model.generate([prompt])
    response = responses[0]
    #print(response)
    return response.generated_text, response.input_token_count, response.generated_token_count