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
from utils.helpers import remove_html_tags, remove_emails, preprocess_text_input, remove_non_ascii, refine_ticket_data, load_data_v1, convert_to_lower, BertTokenEstimator, createAccountCollection, tokenize_ticket_data, embed, create_status_tracker_in_db
from utils.helpers import EMBEDDING_BATCH_SIZE, TOKENIZATION_BATCH_SIZE, INGESTION_BATCH_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MAX_SIZE

from pymongo import MongoClient

bert_token_estimator = BertTokenEstimator()

text_chunker = TextChunker(EMBEDDING_MAX_SIZE, tokens=True, token_estimator=BertTokenEstimator(), overlap_percent=0.3)

banned_resolution = []
banned_additional_comments = []

def chunkTicketData(row):
    text = createIndexStrTickets(row)
    return text_chunker.chunk(text)

def createIndexStrTickets(row):

    global banned_resolution, banned_additional_comments
    
    result = 'Subject: ' + str(row['short_description']) + '\n\n'
    result += 'Description: ' + re.sub(r"\+91\s?\d+$", "", str(row['long_description']).strip(), flags=re.DOTALL) + '\n\n'
    
    if pd.notnull(row['resolution']) and row['resolution'] not in banned_resolution:
        result += 'Resolution notes: ' + re.sub(r"\+91\s?\d+$", "", str(row['resolution']).strip(), flags=re.DOTALL) + '\n\n'
    if pd.notnull(row['additional_comments']) and row['additional_comments'] not in banned_additional_comments:
        result += 'Additional Comments: ' + re.sub(r"\+91\s?\d+$", "", str(row['additional_comments']).strip(), flags=re.DOTALL)
    
    result = preprocess_text_input(result)
    result = remove_emails(result)
    return remove_non_ascii(result.replace('\r',''))

def processTicketData(account, ticket_path):

    global banned_resolution, banned_additional_comments

    tickets = load_data_v1(ticket_path)
    
    print("Step 1 of 5: Tickets loaded.")
    
    tickets = refine_ticket_data(tickets)

    print("Step 2 of 5: Tickets preprocessed.")

    tickets = tickets[['assignment_group', 'short_description', 'long_description', 'resolution', 'additional_comments']]
    tickets['assignment_group'] = tickets['assignment_group'].apply(convert_to_lower)
    tickets = tickets.reset_index()
    tickets = tickets.rename(columns={'index': 'ID'})
    tickets.set_index('ID', inplace=True)

    banned_additional_comments += [x for x in tickets['additional_comments'].unique() if len(str(x)) < 10]
    banned_resolution += [x for x in tickets['resolution'].unique() if len(str(x)) < 10]

    ticket_chunks = tickets.apply(chunkTicketData, axis=1).explode().reset_index()

    ticket_chunks.columns = ['ID', 'text']

    # First, let's create a mapping from 'ID' to 'CLASS' in the `documents_copy` DataFrame.
    class_mapping = tickets['assignment_group'].to_dict()

    # Now, let's add the 'CLASS' column to the `chunks_df` DataFrame using this mapping.
    ticket_chunks['assignment_group'] = ticket_chunks['ID'].map(class_mapping)

    ticket_dataset = Dataset.from_pandas(ticket_chunks)

    # Generate the tokens for each entry.
    ticket_dataset = ticket_dataset.map(tokenize_ticket_data, batch_size=TOKENIZATION_BATCH_SIZE, batched=True)
    # Set the ouput format to torch so it can be pushed into embedding model
    ticket_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask'], output_all_columns=True)

    ticket_dataset = ticket_dataset.map(embed, remove_columns=['input_ids', 'token_type_ids', 'attention_mask'], batched = True, batch_size=EMBEDDING_BATCH_SIZE)

    print("Step 3 of 5: Tickets chunked.")

    ingestion_collection = createAccountCollection(account, 'ticketData')

    print("Step 4 of 5: Tickets collection created.")

    def insert_tickets(batch):
        insertable = [
            batch['ID'].tolist(),
            [x.lower().strip() for x in batch['assignment_group']],
            ['td' for _ in range(len(batch['text']))],
            batch['text'],
            normalize(batch['question_embedding'], dim=1).tolist()
        ]
        ingestion_collection.insert(insertable)

    # Ingest the data into Milvus.
    ticket_dataset.map(insert_tickets, batched=True, batch_size=64)
    ingestion_collection.flush()

    print("Step 5 of 5: Tickets ingested. Process complete.")

# fetch filepath of file to process from command line argument

if __name__ == '__main__':
    account = sys.argv[1]
    filepath = sys.argv[2]

    config = get_configuration()

    client = MongoClient(config['mongodb']['uri'])
    db = client['AMS']

    mongo_collection = db[f'{account.lower().strip()}_status']

    # check if there is already a ticket processing job in progress, and if yes, reject this one
    if mongo_collection.find_one({'type':"ticketData", 'status': {'$in': ['processing']}}):
        print("Processing tickets failed.")
        print("There is already a ticket processing job in progress.")
        sys.exit(1)
    
    create_status_tracker_in_db(mongo_collection, 'ticketData')
    
    print("Processing tickets")
    try:
        processTicketData(account, filepath)

        mongo_collection.update_one({'type':"ticketData"}, {'$set': {'status': 'completed'}})

        print("Processing tickets completed.")

    except Exception as e:
        print("Processing tickets failed.")
        print(e)

        mongo_collection.update_one({'type':"ticketData"}, {'$set': {'status': 'failed', 'error': str(e)}})