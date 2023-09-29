from typing import Tuple
import numpy as np
from typing import List, Any, Dict

'''
One component of the generative transformer model that we need to understand:

The query is first converted (by us) into a vector embedding--a store of text that's represented as numbers. 
How many dimensions this vector store contains is unknown right now.
The document embedding is also stored as a vector embedding.

The model then calculates cos(theta) where theta == the angle between a document embedding <> vector embedding pair.
the result of this calculation is what's called 'cosine similarity' and this is a method
the gpt model uses to return contextually relevant document text given a query.

a vector space can have infinite dimensions, so the proof of this calculation is difficult to formulate and visualize, 
but suffice to say we have an abstract understanding at this time. 
'''

def get_top_k_embeddings(
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        doc_ids: List[str],
        similarity_top_k: int = 5,
) -> Tuple[List[float], List]:
    """Get top nodes by similarity to the query. I.e., the top 5 doc_embedding nodes with the highest cosine similarity score to
    the query embedding nodes"""
    # dimensions: D
    qembed_np = np.array(query_embedding) #store query embeddings in an array
    # dimensions: N x D
    dembed_np = np.array(doc_embeddings)  #store doc embeddings in an array
    # dimensions: N
    dproduct_arr = np.dot(dembed_np, qembed_np)
    # dimensions: N
    norm_arr = np.linalg.norm(qembed_np) * np.linalg.norm(
        dembed_np, axis=1, keepdims=False
    )
    # dimensions: N
    cos_sim_arr = dproduct_arr / norm_arr

    # now we have the N cosine similarities for each document
    # sort by top k cosine similarity, and return ids
    tups = [(cos_sim_arr[i], doc_ids[i]) for i in range(len(doc_ids))]
    sorted_tups = sorted(tups, key=lambda t: t[0], reverse=True)

    sorted_tups = sorted_tups[:similarity_top_k]

    result_similarities = [s for s, _ in sorted_tups]
    result_ids = [n for _, n in sorted_tups]
    return result_similarities, result_ids
