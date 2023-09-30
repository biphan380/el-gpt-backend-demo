from queue import Queue
import os
from threading import Thread
from dotenv import load_dotenv
from llama_index.callbacks import LlamaDebugHandler, CallbackManager

load_dotenv()

import boto3
import pinecone
from multiprocessing.managers import BaseManager

from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from llama_index import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
)
from llama_index import download_loader

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

def initialize_index(): 
    print("start to initialize index")
    """Create a new global index, or load one from the pre-set path."""
    global index, stored_docs, docstore, index_store

    reader = CustomDirectoryReader(
    input_dir="cases/"
    )

    documents = reader.load_data()

    write_documents_to_file(documents)

    text_splitter = SentenceSplitter(
    chunk_size=1024,
    # separator=" ", ...no separator??
    )

    text_chunks = []

    doc_indexes = []
    for doc_index, doc in enumerate(documents):
        cur_text_chunks = text_splitter.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)
        doc_indexes.extend([doc_index] * len(cur_text_chunks))

    write_text_chunks_to_file(text_chunks)

    nodes = [] 
    # used 'i' to mean 'index' because in a lot of languages, the index position that the iterator returns (e.g enumerate) is declared as 'i'
    for i, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        src_doc = documents[doc_indexes[i]] # We can see why doc_idxs is structured the way it is, to ensure
        node.metadata = src_doc.metadata   # That the node will always find which src_doc it came from.  
        nodes.append(node)                 # Only thing is, I don't think the src_doc has any metadata right now
                                            # Look at documents.txt and you will see the Metadata output is blank 

    write_nodes_to_file(nodes)


    from llama_index.node_parser.extractors import (
        MetadataExtractor,
        QuestionsAnsweredExtractor,
        TitleExtractor,
    )
    from llama_index.llms import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo")

    metadata_extractor = MetadataExtractor(
        extractors=[
            # TitleExtractor(nodes=5, llm=llm),
            QuestionsAnsweredExtractor(questions=3, llm=llm),
        ],
        in_place=False,
    )

    import os
    import pickle


    # Define the path to the cache file 
    cache_file = 'processed_nodes.pk1'

    if os.path.exists(cache_file):
        # If cache file exists, load the processed nodes from the file 
        with open(cache_file, 'rb') as f:
            nodes = pickle.load(f)
    else:
        # If cache file does not exist, process the nodes and save the result to the file
        nodes = nodes
        nodes = metadata_extractor.process_nodes(nodes)
        with open(cache_file, 'wb') as f:
            pickle.dump(nodes, f)

    # print out the nodes with their new metadata 
    write_nodes_to_file(nodes)


    from llama_index.embeddings import OpenAIEmbedding

    embed_model = OpenAIEmbedding()

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    from vector_store.vector_store_3b import VectorStore3B
    vector_store = VectorStore3B()
    # load nodes created from the cases into the vector stores
    vector_store.add(nodes)

    from llama_index.vector_stores import VectorStoreQuery, VectorStoreQueryResult

    from llama_index import VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(vector_store)

    from llama_index.storage import StorageContext
    index.storage_context.persist(persist_dir="storage")
    print("index initialized")

def query_index(query_text):
    """Query the global index."""
    print("querying index...")
    global index
    response = index.as_query_engine().query(query_text)
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