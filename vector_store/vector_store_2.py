from typing import List, Any, Dict
from .vector_store_interface import BaseVectorStore
from llama_index.schema import BaseNode

# Not sure why docs decided to call this VectorStore2, could've just called it VectorStore
class VectorStore2(BaseVectorStore):
    """VectorStore2 (add/get/delete implemeneted)."""

    stores_text: bool = True

    def __init__(self) -> None: 
        """Init params. """
        self.node_dict: Dict[str, BaseNode] = {}

    def get(self, text_id: str) -> List[float]:
        """Get embedding."""
        return self.node_dict[text_id]
    
    def add(
        self,
        nodes: List[BaseNode]
    ) -> List[str]:
        """Add nodes to index."""
        for node in nodes:
            self.node_dict[node.node_id] = node

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with node_id.

        Args:
            node_id: str

        """
        del self.node_dict[node_id]

        
