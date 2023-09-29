'''
The next extension is adding metadata filter support. 
This means that we will first filter the candidate set with documents that pass the metadata filters, 
and then perform semantic querying.

For simplicity we use metadata filters for exact matching with an AND condition.'''

from llama_index.vector_stores import MetadataFilters
from llama_index.schema import BaseNode
from typing import cast, List

def filter_nodes(nodes: List[BaseNode], filters: MetadataFilters):
    filtered_nodes = []
    for node in nodes:
        matches = True
        for f in filters.filters:
            if f.key not in node.metadata:
                matches = False
                continue
            if f.value != node.metadata[f.key]:
                matches = False
                continue
        if matches:
            filtered_nodes.append(node)
    return filtered_nodes