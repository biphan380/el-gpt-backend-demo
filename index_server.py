from queue import Queue
import os
from threading import Thread
from dotenv import load_dotenv
from llama_index.callbacks import LlamaDebugHandler, CallbackManager

load_dotenv()

import boto3
from multiprocessing.managers import BaseManager

from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
)

boto3.set_stream_logger("botocore", level="DEBUG")

AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_REGION = os.environ["PINECONE_REGION"]

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

index = None
stored_docs = {}
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])

from llama_index import VectorStoreIndex

from llama_index import SimpleDirectoryReader
from utils.new_reader import CustomDirectoryReader
from utils.to_file import write_documents_to_file
from llama_index.text_splitter import SentenceSplitter
from utils.to_file import write_text_chunks_to_file
from llama_index.schema import TextNode
from utils.to_file import write_nodes_to_file
from ingestion.ingest import create_index

# TO DO: rename this to initialize_indexes
def initialize_index(): 
    print("start to initialize indexes")
    """Create a new global index, or load one from the pre-set path."""
    # Need a way to distinguish the indexes
    global cases_index, stored_docs, docstore, index_store

    cases_index = create_index("cases/", "cases_index")

def query_index(query_text):
    """Query the global index."""
    print("querying index...")
    global cases_index
    response = cases_index.as_query_engine().query(query_text)
    return response

if __name__ == "__main__":
    # init the global index
    print("initializing index...")
    initialize_index()

    # setup server
    # NOTE: you might want to handle the password in a less hardcoded way
    manager = BaseManager(("", 5602), b"password")
    manager.register("initialize_index", initialize_index)
    manager.register("query_index", query_index)

    server = manager.get_server()

    print("server started...")
    server.serve_forever()