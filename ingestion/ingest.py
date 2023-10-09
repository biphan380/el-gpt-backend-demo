'''
We use these functions in index_serve.py to create as many indexes as we need
'''

import string

from utils.new_reader import CustomDirectoryReader
from utils.to_file import write_documents_to_file
from llama_index.text_splitter import SentenceSplitter
from utils.to_file import write_text_chunks_to_file
from llama_index.schema import TextNode
from utils.to_file import write_nodes_to_file

def create_index(input_dir: string, namespace: string):
    print("creating index")

    # not really sure why we need global variables at the moment
    global index, stored_docs, docstore, index_store

    reader = CustomDirectoryReader(
        input_dir=input_dir
    )
    # create Document interfaces from our documents
    documents = reader.load_data()

    # logging to txt file for debugging if needed
    write_documents_to_file(documents)

    # split document interface into N number of node objects
    text_splitter = SentenceSplitter(
        chunk_size=1024,
    )

    text_chunks = []

    doc_indexes = []
    # Our CustomDirectoryReader converts 1 pdf into 1 Document interface
    # Each Document interface is then split into N number of node 
    # objects, each containing <chunk_size> tokens
    # e.g. our nodes currently contain 1024 tokens each, because that's
    # our set chunk_size above.
    # refer to the specific tokenization strategy for each llm 
    # for the exact number of words/characters per token 
    for i, doc in enumerate(documents):
        cur_text_chunks = text_splitter.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)

        # Keep track of the index position of each Document interface
        # from our documents list so we can map
        # each node object to its parent Document 
        doc_indexes.extend([i]*len(cur_text_chunks))

    # logging for debugging if needed
    write_text_chunks_to_file(text_chunks)

    nodes = []
    # used 'i' to mean 'index' because in a lot of languages, the index position that the iterator returns (e.g enumerate) is declared as 'i'
    for i, text_chunk in enumerate(text_chunks):
        node = TextNode(
            text=text_chunk,
        )
        # determine the parent Document interface 
        # from which the text_chunk was created
        src_doc = documents[doc_indexes[i]]

        # Why is appending metadata from the source Document so important?
        # Extracting metadata at the Document creation stage, i.e., when
        # the CustomDirectoryReader is called could mean 
        # we don't need to call the metadata extractor from llama_index,
        # saving us the cost of llm calls 
        # also, more metadata seems to be very helpful for the llm
        # when retrieving relevant document nodes
        node.metadata = src_doc.metadata
        nodes.append(node)

    # logging for debugging if needed
    write_nodes_to_file(nodes)

    from llama_index.node_parser.extractors import (
        MetadataExtractor,
        QuestionsAnsweredExtractor,
        TitleExtractor, # title extractor currently doesn't work
    )
    from llama_index.llms import OpenAI

    llm = OpenAI(model="gpt-3.5-turbo")

    # NOTE: I wonder if increasing the number of questions extracted
    # helps with top_k retrieval?
    metadata_extractor = MetadataExtractor(
        extractors=[
            QuestionsAnsweredExtractor(questions=3, llm=llm),
        ],
        in_place=False, # what happens when set to true?
    )

    import os
    import pickle

    # Let's cache what's already been ingested so
    # we don't keep spending $$ extracting metadata for the same
    # document! 

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

    # logging for debugging purposes
    write_nodes_to_file(nodes)

    from llama_index.embeddings import OpenAIEmbedding

    # we create each node into a numerical embedding 
    # to later put into our vector store. 
    # NOTE: haven't explored this API at the low level yet
    embed_model = OpenAIEmbedding()

    for node in nodes:
        node_embedding = embed_model.get_text_embedding(
            node.get_content(metadata_mode="all")
        )
        node.embedding = node_embedding

    # NOTE: haven't explored what it looks like just before
    # the node embedding is injected into the vector store
    from vector_store.vector_store_3b import VectorStore3B
    vector_store = VectorStore3B()
    vector_store.add(nodes)

    from llama_index.vector_stores import VectorStoreQuery, VectorStoreQueryResult
    from llama_index import VectorStoreIndex
    index = VectorStoreIndex.from_vector_store(vector_store)

    from llama_index.storage import StorageContext
    index.storage_context.persist(persist_dir="storage: " + namespace) #TODO: distinguish between the two indexes
    print("index initialized")
    return index




