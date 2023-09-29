from llama_index.schema import Document
from typing import List

def write_documents_to_file(documents: List[Document], filename: str = 'documents.txt'):
    """
    Write the content and metadata of a list of Document objects to a file.

    :param documents: List of Document objects to be written to file
    :param filename: Name of the file to which the documents will be written
    """
    with open(filename, 'w') as file:
        for doc in documents:
            # Write page_content
            file.write("Page Content:\n")
            file.write(doc.get_content())
            file.write("\n\n")

            # Write metadata
            file.write("Metadata:\n")
            for key, value in doc.metadata.items():
                file.write(f"{key}: {value}\n")

            # Add a separator between documents
            file.write("\n" + "-"*50 + "\n\n")

# Example usage:
# documents = [Document(...), Document(...), ...]  # Assuming you have a list of Document objects
# write_documents_to_file(documents)

from llama_index.schema import TextNode

def write_nodes_to_file(nodes: List[TextNode], filename: str = 'nodes.txt'):
    """
    Write the text and metadata of a list of TextNode objects to a file.

    :param nodes: List of TextNode objects to be written to file
    :param filename: Name of the file to which the nodes will be written
    """
    separator = "\n" + "separator_starts" + "@" * 34 + "separator_ends" + "\n\n"
    
    with open(filename, 'w') as file:
        for node in nodes:
            # Write node text
            file.write("Node Text:\n")
            file.write(node.text)
            file.write("\n\n")

            # Write node metadata
            file.write("Metadata:\n")
            for key, value in node.metadata.items():
                file.write(f"{key}: {value}\n")

            # Add a separator between nodes
            file.write(separator)

# Usage:
# Assuming you have created a list of TextNode objects in the 'nodes' variable
# write_nodes_to_file(nodes, 'output_nodes.txt')

def write_text_chunks_to_file(text_chunks: List[str], filename: str = 'text_chunks.txt'):
    """
    Write a list of text chunks to a file with a separator between each chunk.

    :param text_chunks: List of text chunks to be written to file
    :param filename: Name of the file to which the text chunks will be written
    """
    separator = "\n" + "separator_starts" + "@" * 34 + "separator_ends" * 50 + "\n"
    
    with open(filename, 'w') as file:
        for chunk in text_chunks:
            # Write text chunk
            file.write(chunk)
            
            # Add a separator between text chunks
            file.write(separator)
    
    print(f"Number of text chunks written: {len(text_chunks)}")

# Usage:
# Assuming you have created a list of text chunks in the 'text_chunks' variable
# write_text_chunks_to_file(text_chunks, 'output_text_chunks.txt')

from vector_store.vector_store_3b import VectorStore3B
from llama_index.vector_stores.types import VectorStoreQuery

def write_query_results_to_file(VectorStore3B, VectorStoreQuery, filename='output.txt'):
    """
    A helper function to query the vector store with the given query object,
    and write the results to a file.

    Args:
        vector_store: The vector store to be queried.
        query_obj: The query object.
        filename (str, optional): The name of the file to write the results to. 
            Defaults to 'output.txt'.
    """

    # Open the file in write mode, if the file does not exist, it will be created.
    with open(filename, 'w') as file:

        # Execute the query on the vector store
        query_result = VectorStore3B.query(VectorStoreQuery)

        for similarity, node in zip(query_result.similarities, query_result.nodes):
            # Construct the string to be written to the file
            result_string = (
                "\n----------------\n"
                f"[Node ID {node.node_id}] Similarity: {similarity}\n\n"
                f"{node.get_content(metadata_mode='all')}"
                "\n----------------\n\n"
            )

            # Write the string to the file
            file.write(result_string)

            # If you also want to print the result to the console
            print(result_string)
