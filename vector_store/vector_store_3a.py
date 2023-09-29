from vector_store_2 import VectorStore2
from llama_index.vector_stores.types import (
    VectorStore,
    VectorStoreQuery,
    VectorStoreQueryResult,
)
from typing import List, Any, Dict, cast
from utils.get_top_k import get_top_k_embeddings

class VectorStore3A(VectorStore2):
    """Implements semantic/dense search."""

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Get nodes for response."""

        query_embedding = cast(List[float], query.query_embedding)
        doc_embeddings = [n.embedding for n in self.node_dict.values()]
        doc_ids = [n.node_id for n in self.node_dict.values()]

        similarities, node_ids = get_top_k_embeddings(
            query_embedding,
            doc_embeddings,
            doc_ids,
            similarity_top_k=query.similarity_top_k,
        )
        result_nodes = [self.node_dict[node_id] for node_id in node_ids]

        return VectorStoreQueryResult(
            nodes=result_nodes, similarities=similarities, ids=node_ids
        )