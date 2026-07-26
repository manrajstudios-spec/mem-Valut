import numpy as np
from node import Node
from call_model import embed_msg
from sentence_transformers import util

sentences = [
    "I decided to use Granite 4.1 as the default extraction model for mem-vault.",
    "The mem-vault system should preserve raw conversations as evidence.",
    "I want to compare my greedy graph search against FAISS.",
    "My transformer training slowed down because the model was running partly on CPU.",
    "I fixed Ollama GPU support by installing the CUDA package.",
    "I plan to learn reinforcement learning after finishing mem-vault.",
    "The Unity agent should learn to manage hunger, energy, and money.",
    "I bought Sony WH-CH720N headphones because I do not like heavy bass.",
    "Hyprland feels less comfortable than GNOME for my normal workflow.",
    "I want to apply for a machine learning internship in Germany during university."
]

def make_graph(embeddings,graph_name="ABC",k=5):
    if embeddings.ndim == 1: embeddings = embeddings.reshape(1,-1)
    if len(embeddings) == 1:
        graph = [Node(value=embeddings[0])]
    
    scores = embeddings @ embeddings.T
    
    idx = np.arange(len(embeddings))
    scores[idx, idx] = -np.inf
    
    k = min(k,len(scores))
    print(k)
    similars = np.argpartition(scores,axis=-1,kth=-k)[::-1][:,-k:]
    print(similars)

    graph = [Node(neighbours=similar,value=embedding) for embedding,similar in zip(embeddings,similars)]

    print(graph)
if __name__ == "__main__":
    make_graph(embed_msg(sentences))