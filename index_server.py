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

from ingestion.index_manager import IndexManager

def initialize_indexes():
    print("start to initialize indexes")
    # We might consider using the singleton pattern to create one IndexManager in the future
    # as to avoid using too many global variables
    # but if we already know our globals beforehand, why would we ever let the client dynamically define
    # our globals?
    global cases_index_manager, regs_index_manager, cases_index, regs_index
    
    cases_index_manager = IndexManager("cases/", "cases_index")
    regs_index_manager = IndexManager("regs/", "regs_index")
    
    cases_index = cases_index_manager.create_index("cases/")
    regs_index = regs_index_manager.create_index("regs/")

def query_cases(query_text):
    """Query the global cases_index."""
    print("querying cases index...")
    response = cases_index.as_query_engine().query(query_text)
    return response

def query_regs(query_text):
    """Query the global regs_index."""
    print("querying regs index...")
    response = regs_index_manager.index.as_query_engine().query(query_text)
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
    manager.register("query_regs", query_regs)

    server = manager.get_server()

    print("server started...")
    server.serve_forever()