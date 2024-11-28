from flask import Flask, request, jsonify, json, Response, send_file
from flask_cors import CORS

from genai import Client, Credentials
from genai.schema import TextGenerationParameters, TextGenerationReturnOptions, TextTokenizationParameters, TextTokenizationReturnOptions
from genai.schema import TextGenerationParameters, DecodingMethod
from genai.text.tokenization import CreateExecutionOptions
from utils.config import get_configuration

from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn.functional import normalize
from torch import clamp, sum
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os, sys
import requests

import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor

from bson.objectid import ObjectId
import bson

from pymongo import MongoClient
from datetime import datetime
from werkzeug.utils import secure_filename
import shutil

import traceback
import subprocess
import logging

logging.getLogger().setLevel(logging.DEBUG)

config = get_configuration()

client = MongoClient(config['mongodb']['uri'])
db = client['AMS']

# initialize Embedding model one time when API starts
model = AutoModel.from_pretrained(config['milvus']['embedding_model'])
tokenizer = AutoTokenizer.from_pretrained(config['milvus']['embedding_model'])

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

connections.connect(host=config['milvus']['host'], port=config['milvus']['port'])

# Check if the server is ready.
print(utility.get_server_version())

def get_genai_creds():
    api_key = config["GENAI_KEY"]
    api_url = config["GENAI_API"]
    if api_key is None or api_url is None:
        print("Either api_key or api_url is None. Please make sure your credentials are correct.")
    if api_url is not None:
        api_url = api_url.rstrip("/")
    creds = Credentials(api_key, api_url)
    return creds

WATSONX_API_KEY = config["GENAI_KEY"]

creds = get_genai_creds()
watsonx_client = Client(credentials=creds)

if creds.api_endpoint:
    print(f"Your API endpoint is: {creds.api_endpoint}")

headers = {
    'Authorization': f'Bearer {os.getenv("GENAI_KEY", None)}'
}

# get the list of supported models from the API
models_response = requests.get(f"{creds.api_endpoint}/v2/models", headers=headers)

# Parse the JSON response
models_data = json.loads(models_response.content)

filter_model_name = config["watsonx_models"]["filter_model_name"]
model1_name = config["watsonx_models"]["model1_name"]
model2_name = config["watsonx_models"]["model2_name"]
model3_name = config["watsonx_models"]["model3_name"]

# set-up inference parameters
filter_params_args = json.dumps(config["watsonx_models"][filter_model_name]["parameters"])
filter_params = TextGenerationParameters(**json.loads(filter_params_args))

model1_args = json.dumps(config["watsonx_models"][model1_name]["parameters"])
model1_params = TextGenerationParameters(**json.loads(model1_args))

print(model1_params)

model2_args = json.dumps(config["watsonx_models"][model2_name]["parameters"])
model2_params = TextGenerationParameters(**json.loads(model2_args))

print(model2_params)

model3_args = json.dumps(config["watsonx_models"][model3_name]["parameters"])
model3_params = TextGenerationParameters(**json.loads(model3_args))

print(model3_params)

app = Flask(__name__)
CORS(app)

def process_question(collection, question, assignment_id=None, limit=5, data_type="td"):
    
    # Tokenize and embed the question
    
    text = "query: " + question
    inputs = tokenizer(text, add_special_tokens=True, truncation=True, padding="max_length", return_attention_mask=True, return_tensors="pt")
    
    model = AutoModel.from_pretrained(config["milvus"]["embedding_model"])
    
    sentence_embs = model(
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
    else:
        res = collection.search(
            embeddings.tolist(),
            anns_field='chunk_embedding',
            param = {},
            output_fields=['chunk', 'type'],
            expr=f"type=='{data_type}' and assignment_id=='{assignment_id.lower().strip()}'",
            limit = limit)
                
    return res

def get_relevant_chunks(account_id, question_text, assignment_group = None, n_results=5):

    #schema = CollectionSchema(fields=fields)
    if account_id in config['milvus']:
        collection_name = config['milvus'][account_id]['collection']
    else:
        collection_name = "AMS_" + account_id

    collection = Collection(name=collection_name)
    
    relevant_chunks_cases = process_question(collection, question_text, assignment_id = assignment_group, limit=n_results, data_type="td")
    
    result_set_length = [len(x) for x in relevant_chunks_cases][0]
    
    if result_set_length < n_results:
        print(question_text)
        print(assignment_group)
    
    relevant_chunks = process_question(collection, question_text, limit=n_results, data_type="kb")
        
    return relevant_chunks_cases, relevant_chunks

def prompt_template(context, question_text):
    return (f"You help solve AMS tickets using information from past tickets and Q&A to suggest possible solutions.\n\nNarrate to the user what is the most probable root cause and solution approach given the following past related tickets and the current ticket description. Take into account what has happened in the past cases and what solution was taken.\n\nDo not consider unique IDs in your answer, only consider the pattern, if any. Suggest both the probable root cause and a solution using the information from the provided context.\n\n"
          + f"Previous Tickets/Knowledge base:\n\n"
          + f"{context}\n\n"
          + f"##\n\n"
          + f"Current Ticket: {question_text}\n\n"
          + f"Probable \033[1mRoot Cause\033[0m (if applicable) and \033[1mSolution:\033[0m ")

def prompt_template(context, question_text):
    return (f"You help solve current cases using information from past cases and knowledge base to suggest possible solutions.\n\nNarrate to the user what is the most probable root cause and solution approach given the following past related cases and the current user case description. Take into account what has happened in the past cases and what solution was taken. Do not consider unique IDs in your answer, only consider the pattern, if any. Suggest both the probable root cause and a solution using the information from the past related cases data.\n\n"
          + f"Previous Cases/Knowledge Base:\n\n"
          + f"{context}\n\n"
          + f"##\n\n"
          + f"Current Case: {question_text}\n\n"
          + f"**Probable \033[1mRoot Cause\033[0m (if applicable) and \033[1mSolution\033[0m ")

def make_prompt(relevant_chunks, context, question_text, max_input_tokens, model_name):
    prompt = prompt_template(context, question_text)

    prompt_token_count = token_count(prompt, model_name)

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
        doc_token_count = token_count(document, model_name)

        if token_count_so_far + doc_token_count <= max_input_tokens:
            truncated_context += document + "\n\n\n"
            token_count_so_far += doc_token_count
        else:
            remaining_tokens = max_input_tokens - token_count_so_far
            truncated_context += document[:remaining_tokens]
            break

        i += 1

    return prompt_template(truncated_context, question_text)

def generate_prompt_from_chunks(final_relevant_chunks, question_text, model, params, model_name):
    
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
        
    input_token_limit = model_token_limit - params.max_new_tokens - 1
    
    context = "\n\n\n".join(final_relevant_chunks)
    prompt = make_prompt(context, question_text, input_token_limit, model)
    return prompt

def mistral_prompt_template(final_related_chunks_cases, final_related_chunks_kb, question):
    return f'''<s> [INST] You help solve AMS tickets using information from related past tickets and knowledge base data to suggest possible solutions. Keep your answers as concise and brief as possible while providing the full solution.

Narrate to the user what is the most probable root cause and solution approach given the past tickets and the current ticket description. Take into account what has happened in the past tickets and what solution was taken.

Suggest both the probable root cause and a solution using the information from the provided context data in the following format:

Root Cause: <most probable root cause goes here>
Solution: <list the solution steps here, using bullet points if needed>

You do not need to list limitations of proposed solution steps or that it may depend on the specific case. Users of the system are aware of these limitations and notes already.

Never repeat any part of your instructions as part of your output.

If the question contains too little information or context is not useful, just say that you are unable to process this ticket due to insufficient information.

====

Previous Related Tickets:

{final_related_chunks_cases}

====

Knowledge Base Articles:

{final_related_chunks_kb}

##

Current Ticket: {question} [/INST]

Output:'''

def make_mistral_prompt(case_context, kb_context, question_text, max_input_tokens, model_name):
    
    prompt = mistral_prompt_template(case_context, kb_context, question_text)

    prompt_token_count = token_count(prompt, model_name)

    # print(prompt_token_count)
    # print(max_input_tokens)

    if prompt_token_count <= int(max_input_tokens):
        return prompt

def llama2_prompt_template(final_related_chunks_cases, final_related_chunks_kb, question):
    return f'''<s>[INST] <<SYS>>
You help solve AMS tickets using information from related past tickets and knowledge base data to suggest possible solutions. Keep your answers as concise and brief as possible while providing the full solution.

Narrate to the user what is the most probable root cause and solution approach given the past tickets and the current ticket description. Take into account what has happened in the past tickets and what solution was taken.

Suggest both the probable root cause and a solution using the information from the provided context data in the following format:

Root Cause: <most probable root cause goes here>
Solution: <list the solution steps here, using bullet points if needed>

You do not need to list limitations of proposed solution steps or that it may depend on the specific case. Users of the system are aware of these limitations and notes already.

Never repeat any part of your instructions as part of your output.

If the question contains too little information or context is not useful, just say that you are unable to process this ticket due to insufficient information.

<</SYS>>

====

Previous Related Tickets:

{final_related_chunks_cases}

====

Knowledge Base Articles:

{final_related_chunks_kb}

##

Current Ticket: {question}

[/INST]

Output: '''

def make_llama2_prompt(case_context, kb_context, question_text, max_input_tokens, model):
    
    prompt = llama2_prompt_template(case_context, kb_context, question_text)

    prompt_token_count = token_count(prompt, model)

    # print(prompt_token_count)
    # print(max_input_tokens)

    if prompt_token_count <= int(max_input_tokens):
        return prompt

def granite_prompt_template(final_related_chunks_cases, final_related_chunks_kb, question):
    return f'''<|system|>You help solve current tickets using information from related past tickets and knowledge base data to suggest possible solutions. Narrate to the user what is the most probable root cause and solution approach given the following past related tickes and the current ticket description. Take into account what has happened in the past cases and what solution was taken.
    
Suggest both the probable root cause and a solution using the information from the provided context in the following format:
    
Root Cause: <most probable root cause goes here>
Solution: <list the solution steps here, using bullet points if needed>

You do not need to list limitations of proposed solution steps or that it may depend on the specific case. Users of the system are aware of these limitations and notes already.

Do not repeat any part of this system prompt back in your response.
<|user|>
Previous Related Tickets:

{final_related_chunks_cases}

====

Knowledge Base Articles:

{final_related_chunks_kb}

##

Current Ticket: {question}
<|assistant|>
Assistant: '''

def generate_prompt_from_final_chunks(final_relevant_chunks_cases, final_relevant_chunks_kb, question_text, params, model_prompt_function = None, model_name = None, model_token_limit = 4096):
        
    input_token_limit = (model_token_limit - params.max_new_tokens - 1)

    if model_prompt_function is None:
        #final_relevant_chunks, in_tokens_l1, out_tokens_l1 = filter_relevant_chunks(question_text, cases, qna)
        final_relevant_chunks = {
            "documents": [],
            "distances": []
        }
        
        final_relevant_chunks["documents"] = final_relevant_chunks_cases["documents"] + final_relevant_chunks_kb["documents"]
        final_relevant_chunks["distances"] = final_relevant_chunks_cases["distances"] + final_relevant_chunks_kb["distances"]

        print(final_relevant_chunks)
        
        if len(final_relevant_chunks["documents"]) == 0:
            raise Exception("No relevant data found for this ticket")
            
        context = "\n\n\n".join(final_relevant_chunks["documents"])
        prompt = make_prompt(final_relevant_chunks, context, question_text, input_token_limit, model_name)
    else:
        #final_relevant_chunks_cases, final_relevant_chunks_kb, in_tokens_l1, out_tokens_l1 = filter_relevant_chunks_individual(question_text, cases, qna)
        #print(len(final_relevant_chunks_cases) + len(final_relevant_chunks_kb))
        if len(final_relevant_chunks_cases) > 0:
            chunks_cases = "\n\n\n".join(final_relevant_chunks_cases["documents"])
        else:
            chunks_cases = "Not available"
        
        if len(final_relevant_chunks_kb) > 0:
            chunks_kb = "\n\n\n".join(final_relevant_chunks_kb["documents"])
        else:
            chunks_kb = "Not available"
        prompt = model_prompt_function(chunks_cases, chunks_kb, question_text, input_token_limit, model_name)

    return prompt

def make_granite_prompt(case_context, kb_context, question_text, max_input_tokens, model_name):
    
    prompt = granite_prompt_template(case_context, kb_context, question_text)

    prompt_token_count = token_count(prompt, model_name)

    if prompt_token_count <= int(max_input_tokens):
        return prompt

def remove_pii(text):
    # Remove phone numbers
    text = re.sub(r"\+\d{2}\s?\d+|\+\d{2}-\d+", "", text)
    
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "", text)
    
    return text

def cap_consecutive_newlines(input_str):
    # Use a regular expression to replace consecutive newlines with a maximum of two
    result = re.sub(r'\n{3,}', '\n', input_str)
    return result

def remove_extra_spaces(input_str):
    # Use a regular expression to replace multiple spaces with a single space
    result = re.sub(r' +', ' ', input_str)
    return result.strip()

def preprocess_text_input(txt):
    return remove_pii(cap_consecutive_newlines(remove_extra_spaces(txt)))

# Token counting function
def token_count(doc, model_name):
    responses = list(
        watsonx_client.text.tokenization.create(
            model_id=model_name,
            input=[doc],
            execution_options=CreateExecutionOptions(ordered=False, batch_size=1),
            parameters=TextTokenizationParameters(
                return_options=TextTokenizationReturnOptions(
                    tokens=False
                )
            ))
    )
    try:
        return responses[0].results[0].token_count
    except:
        logging.error(f"Error in token_count: {responses[0]}")
        return 0

def process_relevant_chunks(prompts, chunks, total_input_tokens, total_output_tokens, model_name):

    print("==="*20)
    print(prompts)
    print("==="*20)
    
    final_relevant_chunks = []
    
    try:
        responses = watsonx_client.text.generation.create(
        model_id=model_name,
        inputs=prompts,
        execution_options=CreateExecutionOptions(ordered=True, batch_size=len(prompts)),
        parameters=TextGenerationParameters(
            decoding_method="greedy",
            max_new_tokens=4,
            min_new_tokens=1
        ))
    
        print(responses)
        print("==="*20)

        i = 0
    
        for response in responses:
            chunk = chunks[i]
            i += 1

            print(response.results[0])

            total_input_tokens += response.results[0].input_token_count
            total_output_tokens += response.results[0].generated_token_count
            
            #print(response.generated_text)
            if "yes" in response.results[0].generated_text.lower():
                final_relevant_chunks.append(chunk)

        return final_relevant_chunks, chunks, total_input_tokens, total_output_tokens
    
    except Exception as e:
        print(e)
        print("==="*20)
        print("Error in process_relevant_chunks")
        print("==="*20)

def filter_relevant_chunks_individual(question_text, relevant_chunks_cases, relevant_chunks, model_id = "google/flan-t5-xxl"):
    
    # select generative model to use
    #model_name = "bigscience/mt0-xxl" # for non-EN / mulitlingual data

    final_relevant_chunks_cases = {
        "documents": [],
        "distances": []
    }
    
    total_input_tokens = 0
    total_output_tokens = 0
    prompts = []
    chunks = []
    BATCH_SIZE = 5 # number of tickets to process parallelly
    total_processed = 0
    docs = []
    distances = []

    for hits in relevant_chunks_cases:
        for hit in hits:
            chunk = hit.entity.get('chunk')
            distances.append(hit.distance)
            prompt = "Answer ONLY in yes/no if the provided previous case is relevant to solving the user's current case.\n\n" \
                + "Provided Previous Case:\n\n" \
                + f"{chunk}\n\n" \
                + f"User Current Case: {question_text}\n\n"
            prompts.append(prompt)
            chunks.append(chunk)
        if len(prompts) == BATCH_SIZE:
            total_processed += len(prompts)
            docs, chunks, total_input_tokens, total_output_tokens = process_relevant_chunks(prompts, chunks, total_input_tokens, total_output_tokens, model_id)
            final_relevant_chunks_cases["documents"] += docs
            final_relevant_chunks_cases["distances"] += distances
            prompts = []
            chunks = []

    if len(prompts) > 0:
        total_processed += len(prompts)
        docs, chunks, total_input_tokens, total_output_tokens = process_relevant_chunks(prompts, chunks, total_input_tokens, total_output_tokens, model_id)
        final_relevant_chunks_cases["documents"] += docs
        final_relevant_chunks_cases["distances"] += distances
                
    final_relevant_chunks_kb = {
        "documents": [],
        "distances": []
    }

    prompts = []
    chunks = []

    for hits in relevant_chunks:
        for hit in hits:
            chunk = hit.entity.get('chunk')
            prompt = "Answer ONLY in yes/no if the provided knowledge base article is relevant to answer the user's current query.\n\n" \
                + "Provided KB Article:\n\n" \
                + f"{chunk}\n\n" \
                + f"User Current Query: {question_text}\n\n"
            prompts.append(prompt)
            chunks.append(chunk)
            if len(prompts) == BATCH_SIZE:
                total_processed += len(prompts)
                docs, chunks, total_input_tokens, total_output_tokens = process_relevant_chunks(prompts, chunks, total_input_tokens, total_output_tokens, model_id)
                final_relevant_chunks_kb["documents"] += docs
                final_relevant_chunks_kb["distances"] += distances
                prompts = []
                chunks = []

    if len(prompts) > 0:
        total_processed += len(prompts)
        docs, chunks, total_input_tokens, total_output_tokens = process_relevant_chunks(prompts, chunks, total_input_tokens, total_output_tokens, model_id)
        final_relevant_chunks_kb["documents"] += docs
        final_relevant_chunks_kb["distances"] += distances
    
    return final_relevant_chunks_cases, final_relevant_chunks_kb, total_input_tokens, total_output_tokens

def generate_answer_from_prompt(prompt, model_name):
    
    api_key = config["GENAI_KEY"]
    endpoint = "https://bam-api.res.ibm.com/v2/text/generation?version=2024-01-10"
    headers = {'Authorization': f"Bearer {api_key}"}
    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json={
                "model_id": model_name,
                "input": prompt,
                "parameters": {
                    "decoding_method": DecodingMethod.GREEDY, 
                    "max_new_tokens": 512,
                    "min_new_tokens": 2,
                    "repetition_penalty": 1.05,
                    "stop_sequences": ['<endoftext>','END_KEY','####','\n\nUser:','\n\nAssistant:','\n\n--\n\n'],
                    "temperature": 0,
                    "top_k": 50,
                    "top_p": 1,
                    "typical_p": 1,
                    "include_stop_sequence": False
        }})
        if response.status_code == 200:
            generated_content = response.json()['results'][0]
            return generated_content['generated_text'], generated_content['input_token_count'], generated_content['generated_token_count']
        else:
            print(response.json())
    except Exception as e:
        print("==="*20)
        print("Error in generate_answer_from_prompt")
        print(e)       
        print("==="*20)

def get_answer_from_question(account_id, chunks, question_text, params, assignment_group = None, model_name=model1_name):
    
    prompt = generate_prompt_from_chunks(chunks, question_text, model, params, model_name)
    
    ans, in_tokens_l2, out_tokens_l2 = generate_answer_from_prompt(prompt, model_name)
    
    return ans, in_tokens_l2, out_tokens_l2

def get_answer_from_question_and_relevant_chunks(cases, qna, question_text, params, assignment_group = None, model_prompt_function = None, model_name = None, model_token_limit = 4096):
    
    prompt = generate_prompt_from_final_chunks(cases, qna, question_text, params, model_prompt_function, model_name, model_token_limit = model_token_limit)

    print("==="*20)
    print(prompt)
    print("==="*20)
    
    ans, in_tokens_l2, out_tokens_l2 = generate_answer_from_prompt(prompt, model_name)
    
    return ans, in_tokens_l2, out_tokens_l2

def compute_and_store_answer(account, query_id, user_message):

    print(f"Query ID: {query_id}")
    print("==="*20)
    
    # get the relevant chunks one time for usage with all 3 models
    raw_cases, raw_qna = get_relevant_chunks(account, user_message)
    
    print("Got relevant chunks...")
    print("==="*20)
    
    cases, qna, in_tokens_l1, out_tokens_l1 = filter_relevant_chunks_individual(user_message, raw_cases, raw_qna)

    print(cases)
    print(qna)
    print("==="*20)

    prompt_constructors_by_model = {
        model1_name: make_mistral_prompt,
        model2_name: make_granite_prompt,
        model3_name: make_llama2_prompt
    }

    token_limits_by_model = {
        model1_name: config["watsonx_models"]["model1_token_limit"],
        model2_name: config["watsonx_models"]["model2_token_limit"],
        model3_name: config["watsonx_models"]["model3_token_limit"]
    }

    print("Starting parallel processing...")
    print("==="*20)

    # do a sequential processing of the below loop instead

    '''
    # code snippet to do sequential processing
    model_1_response = ""
    model_2_response = ""
    model_3_response = ""

    print("Calling model 1")
    if model1_name in prompt_constructors_by_model:
        model_1_response, model_1_in_tokens_l2, model_1_out_tokens_l2 = get_answer_from_question_and_relevant_chunks(cases, qna, user_message, model1_params, model_prompt_function=prompt_constructors_by_model[model1_name], model_name=model1_name)
    else:
        model_1_response, model_1_in_tokens_l2, model_1_out_tokens_l2 = get_answer_from_question_and_relevant_chunks(cases, qna, user_message, model1_params, model_name=model1_name)

    print("Calling model 2")
    if model2_name in prompt_constructors_by_model:
        model_2_response, model_2_in_tokens_l2, model_2_out_tokens_l2 = get_answer_from_question_and_relevant_chunks(cases, qna, user_message, model2_params, model_prompt_function= prompt_constructors_by_model[model2_name], model_name=model2_name)
    else:
        model_2_response, model_2_in_tokens_l2, model_2_out_tokens_l2 = get_answer_from_question_and_relevant_chunks(cases, qna, user_message, model2_params, model_name=model2_name)

    print("Calling model 3")
    if model3_name in prompt_constructors_by_model:
        model_3_response, model_3_in_tokens_l2, model_3_out_tokens_l2 = get_answer_from_question_and_relevant_chunks(cases, qna, user_message, model3_params, model_prompt_function= prompt_constructors_by_model[model3_name], model_name=model3_name)
    else:
        model_3_response, model_3_in_tokens_l2, model_3_out_tokens_l2 = get_answer_from_question_and_relevant_chunks(cases, qna, user_message, model3_params, model_name=model3_name)'''

    # parallelly process the models
    with ThreadPoolExecutor() as executor:
        if model1_name in prompt_constructors_by_model:
            future1 = executor.submit(get_answer_from_question_and_relevant_chunks, cases, qna, user_message, model1_params, model_prompt_function=prompt_constructors_by_model[model1_name], model_name=model1_name, model_token_limit = token_limits_by_model[model1_name])
        else:
            future1 = executor.submit(get_answer_from_question_and_relevant_chunks, cases, qna, user_message, model1_params, model_name=model1_name, model_token_limit = token_limits_by_model[model1_name])
        if model2_name in prompt_constructors_by_model:
            future2 = executor.submit(get_answer_from_question_and_relevant_chunks, cases, qna, user_message, model2_params, model_prompt_function=prompt_constructors_by_model[model2_name], model_name=model2_name, model_token_limit = token_limits_by_model[model2_name])
        else:
            future2 = executor.submit(get_answer_from_question_and_relevant_chunks, cases, qna, user_message, model2_params, model_name=model2_name, model_token_limit = token_limits_by_model[model2_name])
        if model3_name in prompt_constructors_by_model:
            future3 = executor.submit(get_answer_from_question_and_relevant_chunks, cases, qna, user_message, model3_params, model_prompt_function=prompt_constructors_by_model[model3_name], model_name=model3_name, model_token_limit = token_limits_by_model[model3_name])
        else:
            future3 = executor.submit(get_answer_from_question_and_relevant_chunks, cases, qna, user_message, model3_params, model_name=model3_name, model_token_limit = token_limits_by_model[model3_name])
        
        try:
            model_1_response, model_1_in_tokens_l2, model_1_out_tokens_l2 = future1.result()
        except Exception as e:
            print(e)
            model_1_response = str(e)
            model_1_in_tokens_l2 = 0
            model_1_out_tokens_l2 = 0
        try:
            model_2_response, model_2_in_tokens_l2, model_2_out_tokens_l2 = future2.result()
        except Exception as e:
            print(e)
            model_2_response = str(e)
            model_2_in_tokens_l2 = 0
            model_2_out_tokens_l2 = 0
        try:
            model_3_response, model_3_in_tokens_l2, model_3_out_tokens_l2 = future3.result()
        except Exception as e:
            print(e)
            model_3_response = str(e)
            model_3_in_tokens_l2 = 0
            model_3_out_tokens_l2 = 0
   
    # ingest data into mongoDB

    mongo_collection = db[account]

    print("Adding mongoDB data in the background")
    print(query_id)

    print(model_3_response)
    print("==="*20)
    print(model_2_response)
    print("==="*20)
    print(model_1_response)
    print("==="*20)

    # ingest data into mongo
    result = mongo_collection.update_one(
        {
            "_id": ObjectId(str(query_id))
        },
        {
            "$set": {
                "responses": [
                    {"model_name": "Model 1", "model_id": model1_name, "response": model_1_response},
                    {"model_name": "Model 2", "model_id": model2_name, "response": model_2_response},
                    {"model_name": "Model 3", "model_id": model3_name, "response": model_3_response},
                ],
                "usage": {
                    "in_tokens_l1": in_tokens_l1,
                    "out_tokens_l1": out_tokens_l1,
                    "in_tokens_l2": {
                        model1_name: model_1_in_tokens_l2,
                        model2_name: model_2_in_tokens_l2,
                        model3_name: model_3_in_tokens_l2
                    },
                    "out_tokens_l2": {
                        model1_name: model_1_out_tokens_l2,
                        model2_name: model_2_out_tokens_l2,
                        model3_name: model_3_out_tokens_l2
                    }
                }
            }
        }
    )

    print("==="*20)
    print("Updated below data into MongoDB: ")
    print("==="*20)
    print(result)
    print("==="*20)

def get_status_for_account(account_id):
    mongo_collection = db[account_id + "_status"]
    status = mongo_collection.find()
    current_jobs = []
    completed_jobs = []
    for job in status:

        print(job)

        if job['status'] == 'ready' or job['status'] == 'completed':
            completed_jobs.append({
                "batch_id": str(job['_id']),
                "status": job['status'],
            })
        elif job['status'] == 'failed':
            completed_jobs.append({
                "batch_id": str(job['_id']),
                "status": job['status'],
                "error": job['error']
            })
        else:
            current_jobs.append({
                "batch_id": str(job['_id']),
                "status": job['status'],
            })
    return current_jobs, completed_jobs

@app.route('/v1/uploadBatch', methods=['PUT'])
def upload_batch_data():

    secret_token = request.headers.get('X-Auth-Token')
    
    if secret_token != config['secret']:
        return jsonify({
            'account_id': 'None',
            'status': 'failed',
            'fail_reason': 'Unauthorized'
            }), 401

    account_id = request.form.get('accountId')
    
    type_param = "batchData"

    # Check if request has a file
    if 'uploadFile' not in request.files:
        return jsonify({
            'account_id': account_id,
            'status': 'failed',
            'fail_reason': 'No file part'
        }), 400

    file = request.files['uploadFile']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({
            'account_id': account_id,
            'status': 'failed',
            'fail_reason': 'No selected file'
        }), 400

    '''# Check if the file is a CSV
    if not file.filename.endswith('.csv') and not file.filename.endswith('.xlsx'):
        return jsonify({
            'account_id': account_id,
            'status': 'failed',
            'fail_reason': 'Invalid file format, only CSV/XLSX files are allowed'
        }), 400'''

    NEEDED_COLUMNS = ["short_description", "long_description", "assignment_group"]

    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)
    elif file.filename.endswith('.pdf'):
    # Extract text from the PDF file
        pdf_reader = PdfReader(file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
    
    # Example: Convert extracted text to a DataFrame (adjust logic as needed)
    data = {"short_description": [], "long_description": [], "assignment_group": []}
    # Add logic here to parse `pdf_text` and populate `data`
    # For simplicity, this is just an example structure.
    df = pd.DataFrame(data)
    

    missing_columns = [x for x in NEEDED_COLUMNS if x not in df.columns]
    if len(missing_columns) > 0:
        return jsonify({
            'account_id': account_id,
            'status': 'failed',
            'fail_reason': 'Required columns: ' + str(missing_columns) + ' are missing in the headers. Please rename/add required columns to the prescribed format.'
        }), 400

    save_path = "data/" + account_id + "/batchData"

    if not (os.path.exists(save_path)):
        os.makedirs(save_path)

    df.to_csv(os.path.join(save_path, "batchData.csv"), index=False)
    
    p = subprocess.Popen([sys.executable, 'processBatchWorker.py', account_id, os.path.join(save_path, "batchData.csv")])

    return jsonify({
        'account_id': account_id,
        'status': 'processing',
        'fail_reason': ''
    })

@app.route('/v1/submitBulkFeedback', methods=['PUT'])
def upload_feedback():

    secret_token = request.headers.get('X-Auth-Token')
    if secret_token != config['secret']:
        return jsonify({'error': 'Unauthorized'}), 401

    account_id = request.form.get('accountId')

    # Check if request has a file
    if 'uploadFile' not in request.files:
        return jsonify({
            'account_id': account_id,
            'status': 'failed',
            'fail_reason': 'No file part'
        }), 400

    file = request.files['uploadFile']

    # Check if the file is empty
    if file.filename == '':
        return jsonify({
            'account_id': account_id,
            'status': 'failed',
            'fail_reason': 'No selected file'
        }), 400

    # Check if the file is a CSV/XLSX
    if not file.filename.endswith('.csv') and not file.filename.endswith('.xlsx'):
        return jsonify({
            'account_id': account_id,
            'status': 'failed',
            'fail_reason': 'Invalid file format, only CSV/XLSX files are allowed'
        }), 400

    # Read and verify column names
    NEEDED_COLUMNS = ["model1_score", "model2_score", "model3_score"]

    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)

    missing_columns = [x for x in NEEDED_COLUMNS if x not in df.columns]

    if len(missing_columns) > 0:
        return jsonify({
            'account_id': account_id,
            'status': 'failed',
            'fail_reason': 'Required columns: ' + str(missing_columns) + ' are missing in the headers. Please rename/add required columns to the prescribed format.'
        }), 400

    save_path = "data/" + account_id + "/feedback"

    if not (os.path.exists(save_path)):
        os.makedirs(save_path)

    count = 0
    while os.path.exists(os.path.join(save_path, "feedback_" + str(count) + ".csv")):
        count += 1

    df.to_csv(os.path.join(save_path, "feedback_" + str(count) + ".csv"), index=False)

    return jsonify({
        'account_id': account_id,
        'success': True,
        'fail_reason': ''
    })

@app.route('/v1/uploadData', methods=['PUT'])
def upload_data():

    secret_token = request.headers.get('X-Auth-Token')
    
    if secret_token != config['secret']:
        return jsonify({'error': 'Unauthorized'}), 401

    account_id = request.form.get('accountId')
    type_param = request.form.get('type')

    # Check if type_param is valid
    valid_types = ["ticketData", "kbData"]
    if type_param not in valid_types:
        return jsonify({
            'account_id': account_id,
            'type': type_param,
            'status': 'failed',
            'fail_reason': 'Invalid type parameter'
        }), 400

    # Check if request has a file
    if 'uploadFile' not in request.files:
        return jsonify({
            'account_id': account_id,
            'type': type_param,
            'status': 'failed',
            'fail_reason': 'No file part'
        }), 400

    file = request.files['uploadFile']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({
            'account_id': account_id,
            'type': type_param,
            'status': 'failed',
            'fail_reason': 'No selected file'
        }), 400

    # Check if the file is a CSV
    if not file.filename.endswith('.csv') and not file.filename.endswith('.xlsx'):
        return jsonify({
            'account_id': account_id,
            'type': type_param,
            'status': 'failed',
            'fail_reason': 'Invalid file format, only CSV/XLSX files are allowed'
        }), 400

    # Read and verify column names
    if type_param == "ticketData":
        NEEDED_COLUMNS = ["short_description", "long_description", "assignment_group", "resolution", "additional_comments"]
    else:
        NEEDED_COLUMNS = ["question", "answer","tags"]

    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        df = pd.read_excel(file)

    missing_columns = [x for x in NEEDED_COLUMNS if x not in df.columns]
    if len(missing_columns) > 0:
        return jsonify({
            'account_id': account_id,
            'type': type_param,
            'status': 'failed',
            'fail_reason': 'Required columns: ' + str(missing_columns) + ' are missing in the headers. Please rename/add required columns to the prescribed format.'
        }), 400
    
    save_path = "data/" + account_id + "/" + type_param
    try:
        os.makedirs(save_path)
    except:
        print("Directory already exists, skipping step.")
    
    df.to_csv(os.path.join(save_path, type_param + ".csv"))

    # call ingestionWorker.py in the background if type_param is ticketData and ingestionWorkerKb if type_param is kbData

    if type_param == "ticketData":
        p = subprocess.Popen([sys.executable, 'ingestionWorker.py', account_id, os.path.join(save_path, type_param + ".csv")])
    else:
        p = subprocess.Popen([sys.executable, 'ingestionWorkerKb.py', account_id, os.path.join(save_path, type_param + ".csv")])

    return jsonify({
        'account_id': account_id,
        'type': type_param,
        'status': 'processing'
    })

@app.route('/v1/checkFileExists', methods=['GET'])
def check_file_exists():
    try:

        secret_token = request.headers.get('X-Auth-Token')
        
        if secret_token != config['secret']:
            return jsonify({'error': 'Unauthorized'}), 401

        account_id = request.args.get('accountId')

        valid_types = ["ticketData", "kbData"]

        response = {}

        for valid_type in valid_types:
            save_path = "data/" + account_id + "/" + valid_type
            if os.path.exists(save_path) and len([x for x in os.listdir(save_path) if x.endswith('.csv')]) > 0:
                response[valid_type] = str([x for x in os.listdir(save_path) if x.endswith('.csv')][0])
        
        return jsonify(
            response
        )

    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "traceback": ''.join(traceback.TracebackException.from_exception(e).format())
            }
        ), 400

@app.route('/v1/viewBatch', methods=['GET'])
def view_batch():
    try:
        
        secret_token = request.headers.get('X-Auth-Token')
        
        if secret_token != config['secret']:
            return jsonify({'error': 'Unauthorized'}), 401

        account_id = request.args.get('accountId')

        current_jobs, completed_jobs = get_status_for_account(account_id)

        return jsonify(
            {
                "current_jobs": current_jobs,
                "completed_jobs": completed_jobs 
            }
        ), 200

    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "traceback": ''.join(traceback.TracebackException.from_exception(e).format())
            }
        ), 400

@app.route('/v1/downloadResult', methods=['GET', 'OPTIONS'])
def download_result():
    # provide a download to "batch_result.csv"
    try:

        secret_token = request.headers.get('X-Auth-Token')
        
        if secret_token != config['secret']:
            return jsonify({'error': 'Unauthorized'}), 401

        account_id = request.args.get('accountId')
        batch_id = request.args.get('jobId')

        #save_path = "data/" + account_id + "/" + type_param + "/" + batch_id + "_result.csv"
        save_path = f'data/{account_id}/{batch_id}.csv'

        if os.path.exists(save_path):
            return send_file(save_path, as_attachment=True), 200
        else:
            return jsonify({
                'account_id': account_id,
                'status': 'failed',
                'fail_reason': 'File does not exist'
            }), 400
    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "traceback": ''.join(traceback.TracebackException.from_exception(e).format())
            }
        ), 400

@app.route('/v1/getPredictSingleSplitResult', methods=['POST'])
def get_predict_single_split_result():
    try:
        
        secret_token = request.headers.get('X-Auth-Token')
        
        if secret_token != config['secret']:
            return jsonify({'error': 'Unauthorized'}), 401
        data = request.json

        account = data['account']
        ticketId = data['ticketId']

        mongo_collection = db[account]
        query_result = mongo_collection.find_one(
            {
                "_id": ObjectId(ticketId)
            }
        )

        print(query_result)

        return jsonify(
            {
                "query_id": str(ticketId),
                "output": [
                    {"model_str": query_result['responses'][0]['model_name'], "model_output": query_result['responses'][0]['response']},
                    {"model_str": query_result['responses'][1]['model_name'], "model_output": query_result['responses'][1]['response']},
                    {"model_str": query_result['responses'][2]['model_name'], "model_output": query_result['responses'][2]['response']}
            ]}), 200
    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "traceback": ''.join(traceback.TracebackException.from_exception(e).format())
            }
        ), 400

@app.route('/v1/predictSingleSplit', methods=['POST'])
def predict_single_split():
    try:
        
        secret_token = request.headers.get('X-Auth-Token')
        
        if secret_token != config['secret']:
            return jsonify({'error': 'Unauthorized'}), 401
        data = request.json

        if 'shortDescription' in data and len(data['shortDescription']) > 5:
            user_message = f"Subject: {preprocess_text_input(data['shortDescription'])}\n\nDescription: {preprocess_text_input(data['longDescription'])}"
        else:
            user_message = 'Description: ' + preprocess_text_input(data['longDescription'])

        print(user_message)
        print("==="*20)

        account = data['account']

        mongo_collection = db[account]
        query_id = mongo_collection.insert_one(
            {
                "short_description": data['shortDescription'],
                "long_description": data['longDescription'],
                "timestamp": datetime.now().utcnow()
            }
        )

        print(query_id.inserted_id)
        print("==="*20)

        # execute a function in a background thread but don't block for its result using a thread
        executor = ThreadPoolExecutor()
        future = executor.submit(compute_and_store_answer, account, query_id.inserted_id, user_message)

        return jsonify(
            {
                "query_id": str(query_id.inserted_id)
            }), 200
    except Exception as e:
        return jsonify(
            {
                "error": str(e),
                "traceback": ''.join(traceback.TracebackException.from_exception(e).format())
            }
        ), 400

@app.route('/v1/recordPositiveFeedback', methods=['POST'])
def record_positive_feedback():
    try:
        secret_token = request.headers.get('X-Auth-Token')
        
        if secret_token != config['secret']:
            return jsonify({'error': 'Unauthorized'}), 401

        data = request.get_json()
        
        mongo_collection = db[data['account']]
        query_id = data['combinedModelId'].split('_')[0]
        model_index = int(data['combinedModelId'].split('_')[1])

        mongo_collection.update_one(
            {"_id": ObjectId(query_id)},
            {"$set": {f"responses.{model_index}.{'score'}": int(data['score'])}}
        )

        return {"success": True}, 200
    except Exception as e:
        print(e)
        print(''.join(traceback.TracebackException.from_exception(e).format()))
        return {"success": False}, 400

@app.route('/v1/recordNegativeFeedback', methods=['POST'])
def record_negative_feedback():
    try:

        secret_token = request.headers.get('X-Auth-Token')
        if secret_token != config['secret']:
            return jsonify({'error': 'Unauthorized'}), 401

        data = request.get_json()
        
        mongo_collection = db[data['account']]
        query_id = data['combinedModelId'].split('_')[0]
        model_index = int(data['combinedModelId'].split('_')[1])

        mongo_collection.update_one(
            {"_id": ObjectId(query_id)},
            {"$set": {f"responses.{model_index}.{'score'}": int(data['score'])}}
        )

        mongo_collection.update_one(
            {"_id": ObjectId(query_id)},
            {"$set": {f"responses.{model_index}.{'negative_feedback_types'}": data['negative_feedback_types']}}
        )

        if 'negative_feedback_reason' in data:
            mongo_collection.update_one(
            {"_id": ObjectId(query_id)},
            {"$set": {f"responses.{model_index}.{'negative_feedback_reason'}": data['negative_feedback_reason']}}
        )

        return {"success": True}, 200
    except Exception as e:
        print(e)
        print(''.join(traceback.TracebackException.from_exception(e).format()))
        return {"success": False}, 400

if __name__ == '__main__':
    # generate_answer_from_prompt("test", "ibm/granite-13b-instruct-v2")
    app.run(host='0.0.0.0', debug=False)