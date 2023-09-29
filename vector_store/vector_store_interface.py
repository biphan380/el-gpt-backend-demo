'''
Now we’ll build our in-memory vector store. We’ll store Nodes within a simple Python dictionary. 
We’ll start off implementing embedding search, and add metadata filters.

'''

from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from typing import List, Any, Optional, Dict
from llama_index.schema import TextNode, BaseNode
import os

class BaseVectorStore(VectorStore):
    """Simple custom Vector Store.

    Stores documents in a simple in-memory dict.
    
    """

    stores_text: bool = True

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        pass

    def add(
        self,
        nodes: List[BaseNode]
    ) -> List[str]:
        """Add nodes to index."""
        pass

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.
        
        Args:
            ref_doc_id (str): The doc_id of the document to delete.
            
        """
        pass

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""
        pass

    def persist(self, persist_path, fs=None) -> None:
        """Persist the SimpleVectorStore to a directory.

        NOTE: we are not implementing this for now.
        
        """
        pass




