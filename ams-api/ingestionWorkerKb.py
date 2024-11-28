import json
import os, sys
from typing import Any, Dict, Iterable, List, Optional
import unicodedata

import pandas as pd
import requests
import tqdm
import re

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

from utils.config import get_configuration
from utils.helpers import remove_html_tags, remove_emails, preprocess_text_input, remove_non_ascii, load_data_v1, convert_to_lower, BertTokenEstimator, createAccountCollection, tokenize_kb_chunk_data, embed, create_status_tracker_in_db
from utils.helpers import EMBEDDING_BATCH_SIZE, TOKENIZATION_BATCH_SIZE, INGESTION_BATCH_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MAX_SIZE

from pymongo import MongoClient

bert_token_estimator = BertTokenEstimator()

text_chunker = TextChunker(EMBEDDING_MAX_SIZE, tokens=True, token_estimator=BertTokenEstimator(), overlap_percent=0.3)

def createIndexStrKb(row):
    result = 'Topic: ' + remove_html_tags(str(row['question'])) + '\n\n'
    result += 'Answer: ' + remove_html_tags(str(row['answer'])) + '\n\n'
    result += 'Tags: ' + remove_html_tags(str(row['tags']))
    re.sub(r"\+91\s\d+\b", "", result, flags=re.DOTALL)
    result = remove_emails(result)
    result = preprocess_text_input(result)
    return remove_non_ascii(result.replace('\r',''))

def chunkKbData(row):
    text = createIndexStrKb(row)
    return text_chunker.chunk(text)

def processkbData(account, kb_path):

    knowledge = load_data_v1(kb_path)
    
    print("Step 1 of 4: KBs loaded from XLSX/CSV file")
    
    knowledge_chunks = knowledge.apply(chunkKbData, axis=1).explode().reset_index()

    knowledge_chunks.columns = ['ID', 'text']

    class_mapping = knowledge['tags'].to_dict()

    knowledge_chunks['tags'] = knowledge_chunks['ID'].map(class_mapping)

    knowledge_dataset = Dataset.from_pandas(knowledge_chunks)

    # Generate the tokens for each entry.
    knowledge_dataset = knowledge_dataset.map(tokenize_kb_chunk_data, batch_size=TOKENIZATION_BATCH_SIZE, batched=True)
    
    # Set the ouput format to torch so it can be pushed into embedding model
    knowledge_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask'], output_all_columns=True)

    print("Step 2 of 4: KBs chunked.")

    knowledge_dataset = knowledge_dataset.map(embed, remove_columns=['input_ids', 'token_type_ids', 'attention_mask'], batched = True, batch_size=EMBEDDING_BATCH_SIZE)

    ingestion_collection = createAccountCollection(account, 'kbData')

    print("Step 3 of 4: KBs embedded and collection created.")

    def insert_kb(batch):
        insertable = [
            [-1 for x in batch['ID']],
            [x.lower() for x in batch['tags']],
            ['kb' for _ in range(len(batch['text']))],
            batch['text'],
            normalize(torch.FloatTensor(batch['question_embedding']), dim=1).tolist()
        ]    
        ingestion_collection.insert(insertable)

    # Ingest the data into Milvus.
    knowledge_dataset.map(insert_kb, batched=True, batch_size=64)
    ingestion_collection.flush()

    print("Step 4 of 4: KB ingested. Process complete.")

# fetch filepath of file to process from command line argument

if __name__ == '__main__':
    account = sys.argv[1]
    filepath = sys.argv[2]

    config = get_configuration()

    client = MongoClient(config['mongodb']['uri'])
    db = client['AMS']

    mongo_collection = db[f'{account.lower().strip()}_status']

    # check if there is already a kb processing job in progress, and if yes, reject this one
    if mongo_collection.find_one({'type':"kbData", 'status': {'$in': ['processing']}}):
        print("Processing kbs failed.")
        print("There is already a kb processing job in progress.")
        sys.exit(1)
    
    create_status_tracker_in_db(mongo_collection, 'kbData')
    
    print("Processing kbs")
    try:
        processkbData(account, filepath)

        mongo_collection.update_one({'type':"kbData"}, {'$set': {'status': 'completed'}})

        print("Processing kbs completed.")

    except Exception as e:
        print("Processing kbs failed.")
        print(e)

        mongo_collection.update_one({'type':"kbData"}, {'$set': {'status': 'failed', 'error': str(e)}})