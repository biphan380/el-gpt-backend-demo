import os
from dotenv import load_dotenv
from llama_index.callbacks import LlamaDebugHandler, CallbackManager

load_dotenv()

import boto3
from multiprocessing.managers import BaseManager

boto3.set_stream_logger("botocore", level="DEBUG")

PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_REGION = os.environ["PINECONE_REGION"]

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

index = None
stored_docs = {}
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

from ingestion.ingest import create_index

def initialize_indexes(): 
    print("start to initialize indexes")
    """Create a new global index, or load one from the pre-set path."""
    # Need a way to distinguish the indexes
    global cases_index, stored_docs, docstore, index_store

    # arg1 is directory to read from and arg2 is a namespace prefix for the directory that's persisted to disk 
    cases_index = create_index("cases/", "cases_index")

def query_cases(query_text):
    """Query the global cases_index."""
    print("querying cases index...")
    global cases_index
    response = cases_index.as_query_engine().query(query_text)
    return response

if __name__ == "__main__":
    # init the global index
    print("initializing indexes...")
    initialize_indexes()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(("", 5602), b"password")
    manager.register("initialize_indexes", initialize_indexes)
    manager.register("query_cases", query_cases)

    server = manager.get_server()

    print("server started...")
    server.serve_forever()