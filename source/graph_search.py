import numpy as np
import random
from classes import Node
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

def make_graph(embeddings,graph_name="ABC",k=5,min_samples_per_leaf=5,threshold=0.5):
    if embeddings.ndim == 1: embeddings = embeddings.reshape(1,-1)
    if len(embeddings) == 1:
        graph = [Node(value=embeddings[0])]
        return graph
    
    scores = embeddings @ embeddings.T
    
    idx = np.arange(len(embeddings))
    scores[idx, idx] = -np.inf
    
    k = min(k,len(scores)-1)

    similars = scores[scores >= threshold]

    graph = [Node(neighbours=[graph[i] for i in similar],value=embedding) for embedding,similar in zip(embeddings,similars)]

    return graph

def add_to_graph(to_add,graph=[]):
    visited = set()
    similars = set()

    max_depth = 5
    min_samples_per_leaf = 7
    
    random_rods = random.sample(graph,k=min(3,len(graph)-1)) 
    
    stack = [(node,0) for node in random_rods]
    
    offset = 0.15
    min_threshold = -np.inf
    
    while stack:
        cur_node,depth = stack.pop()
        
        if cur_node in visited or depth > max_depth:
            visited.add(cur_node)
            continue
        
        visited.add(cur_node)
        
        sim = to_add @ cur_node.value
        
        if sim + offset >= min_threshold:
            if sim > min_threshold:
                min_threshold = sim
                if min_threshold + offset < sim:
                    similar.clear()
            
            similars.add(cur_node)
            
            for neighbour in cur_node.neighbours:
                stack.append((neighbour,depth+1))
    
    new_node = Node(neighbours=list(similars),value=to_add)
    
    for similar in similars:
        if len(cur_node.neighbours) >= min_samples_per_leaf: continue
        
        similar.neighbour.append(new_node)
         
    graph.append(new_node)
       
def get_similar(query_embed,graph=[]):
    visited = set()
    similar = set()
    
    offset = 0.15
    max_depth = 5
    min_threshold = -np.inf
    
    stack = [(node,0) for node in random.sample(graph,min(3,len(graph)))]
    
    while stack:
        cur_node,depth = stack.pop()
                
        if cur_node in visited or depth > max_depth:
            continue
                
        visited.add(cur_node)
        
        sim = query_embed @ cur_node.value
                
        if sim + offset >= min_threshold:
            if sim > min_threshold:
                min_threshold = sim
                if min_threshold + offset < sim:
                    similar.clear()
                
            similar.add(cur_node)
            
            for neighbour in cur_node.neighbours:
                stack.append((neighbour,depth+1))
    
    return list(similar)

if __name__ == "__main__":
    make_graph(embed_msg(sentences))