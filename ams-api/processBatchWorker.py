from pymongo import MongoClient
from utils.config import get_configuration

import json
import os, sys
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import re
import requests
import tqdm
from dotenv import load_dotenv
from genai import Model
from genai.model import Credentials
from genai.schemas import GenerateParams
from datasets import Dataset

import torch
from torch.nn.functional import normalize
from torch import clamp, sum
from transformers import AutoTokenizer, AutoModel

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from milvus import default_server
from pymilvus import connections, utility

import logging

import traceback

from utils.helpers import create_status_tracker_in_db, convert_to_lower, load_data_v1, createAccountCollection, embed, EMBEDDING_BATCH_SIZE, remove_html_tags, remove_extra_spaces, cap_consecutive_newlines, preprocess_text_input, get_answer_from_question_and_relevant_chunks, get_relevant_chunks, filter_relevant_chunks_individual, make_llama2_prompt, make_granite_prompt
from utils.config import get_configuration

def processBatchData(account, filepath, jobid):
    
    if account in config['milvus']:
        COLLECTION_NAME = config['milvus'][account]['collection']
    else:
        COLLECTION_NAME = f"AMS_{account}"

    connections.connect(host='127.0.0.1', port=default_server.listen_port)

    # Check if the server is ready.
    print(utility.get_server_version())

    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    
    print("Step 1 of 5: Connected to existing account vector DB")
    
    challenge_question_df = load_data_v1(filepath)
    challenge_question_df['assignment_group'] = challenge_question_df['assignment_group'].apply(convert_to_lower)
    challenge_question_df.head()

    print("Step 2 of 5: KBs chunked.")

    cases_by_index = []
    kb_by_index = []

    total_in_tokens_l1 = 0
    total_out_tokens_l1 = 0

    for index, row in challenge_question_df.iterrows():
        question_text = preprocess_text_input(f'''Subject: {row['short_description']}\n\nDescription: {row['long_description']}''')
        cases, qna = get_relevant_chunks(collection, question_text)
        filtered_cases, filtered_qna, in_tokens_l1, out_tokens_l1 = filter_relevant_chunks_individual(question_text, cases, qna)
        cases_by_index.append(filtered_cases)
        kb_by_index.append(filtered_qna)
        total_in_tokens_l1 += in_tokens_l1
        total_out_tokens_l1 += out_tokens_l1
    
    nonzeroindices = []
    for i in range(len(challenge_question_df)):
        if len(cases_by_index[i]) or len(kb_by_index[i]):
            nonzeroindices.append(i)

    print("Step 3 of 5: Fetched relevant chunks from the vector DB for context.")

    model_names = [
        'ibm/mpt-7b-instruct',
        'ibm/granite-13b-chat-v1',
        'meta-llama/llama-2-70b-chat'
    ]

    params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=512,
        min_new_tokens=1,
        stream=False,
        repetition_penalty=1.2,
        stop_sequences=['<endoftext>','END_KEY','####','\n\nUser:','\n\nAssistant:','\n\n--\n\n']
    )

    prompt_constructors_by_model = {
        'ibm/granite-13b-chat-v1': make_granite_prompt,
        'meta-llama/llama-2-70b-chat': make_llama2_prompt
    }

    combined_answers_by_model = {}

    token_counts_by_model = {}

    for model_name in model_names:

        answers = []

        token_counts_by_model[model_name] = {
            "in_total": total_in_tokens_l1,
            "out_total": total_out_tokens_l1,
            "in_l1": total_in_tokens_l1,
            "out_l1": total_out_tokens_l1,
            "in_l2": 0,
            "out_l2": 0
        }

        for i, row in challenge_question_df.iterrows():

            if i in nonzeroindices:
                if model_name in prompt_constructors_by_model:
                    ans, in_tokens, out_tokens, in_l1, out_l1, in_l2, out_l2 = get_answer_from_question_and_relevant_chunks(cases_by_index[i], kb_by_index[i], preprocess_text_input(f'''Subject: {row['short_description']}\n\nDescription: {row['long_description']}'''), params, assignment_group=row['assignment_group'], model_prompt_function=prompt_constructors_by_model[model_name], model_name=model_name)
                else:
                    ans, in_tokens, out_tokens, in_l1, out_l1, in_l2, out_l2 = get_answer_from_question_and_relevant_chunks(cases_by_index[i], kb_by_index[i], preprocess_text_input(f'''Subject: {row['short_description']}\n\nDescription: {row['long_description']}'''), assignment_group=row['assignment_group'], params=params, model_name=model_name)
            
                # update token counts to track usage
                token_counts_by_model[model_name]["in_total"] += in_tokens
                token_counts_by_model[model_name]["out_total"] += out_tokens
                token_counts_by_model[model_name]["in_l1"] += in_l1
                token_counts_by_model[model_name]["in_l2"] += in_l2
                token_counts_by_model[model_name]["out_l1"] += out_l1
                token_counts_by_model[model_name]["out_l2"] += out_l2

                # append the answer
                answers.append(cap_consecutive_newlines(ans))
                print(f'{i} processed with model {model_name}')
                
            else:
                #print(e)
                #print(f'{i} encountered error: {e}')
                answers.append("")
                print(f'{i} skipped')
            
        combined_answers_by_model[model_name] = answers

    print("Step 4 of 5: Obtained answers from all models.")

    questions_with_answers_df = challenge_question_df.copy()

    for i in range(len(model_names)):
        questions_with_answers_df[f'model{i+1}_ans'] = combined_answers_by_model[model_names[i]]

    for i in range(len(model_names)):
        questions_with_answers_df[f'model{i+1}_score'] = ""
    
    questions_with_answers_df.to_csv(f'data/{account}/{jobid}.csv', index=False)

    print("Step 5 of 5: Answers saved to CSV file. Process complete.")

if __name__ == "__main__":
    account = sys.argv[1]
    filepath = sys.argv[2]

    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.DEBUG)

    config = get_configuration()

    client = MongoClient(config['mongodb']['uri'])
    db = client['AMS']

    mongo_collection = db[f'{account.lower().strip()}_status']

    # check if there is already a kb processing job in progress, and if yes, reject this one
    if mongo_collection.find_one({'type':"batchData", 'status': {'$in': ['processing']}}):
        print("Processing batch data failed.")
        print("There is already a batch processing job in progress.")
        sys.exit(1)
    
    jobid = create_status_tracker_in_db(mongo_collection, 'batchData')
    print(jobid)

    print("Batch processing started")

    try:
        # process batch data
        processBatchData(account, filepath, jobid)
        mongo_collection.update_one({'type':'batchData', 'status': {'$in': ['processing']}}, {'$set': {'status': 'ready', 'filepath': f'data/{account}/{jobid}.csv'}})
        print("Batch process completed")

    except Exception as e:
        print(f"Processing batch with id {jobid} failed.")
        print(e)
        # record error info
        mongo_collection.update_one({'type':"batchData"}, {'$set': {'status': 'failed', 'error': str(e), 'traceback': ''.join(traceback.TracebackException.from_exception(e).format())}})