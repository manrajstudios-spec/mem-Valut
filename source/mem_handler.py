import json
import pickle
import numpy as np
from call_model import embed_msg
from sentence_transformers import util
from call_model import embed_msg,ask_model
from graph_search import make_graph,get_similar,add_to_graph

test_exchanges = []

with open("Data/test_exchanges.json",'r') as file:
    test_exchanges = json.load(file)
    

def make_groups(exchanges:list[any]):
    to_embed = []
    
    for exchange in exchanges:
        to_embed.append(f"user: {exchange["user"]}\nassistant: {exchange["assistant"]}")
    
    embeddings = embed_msg(exchanges)
    
    grouped = util.community_detection(embeddings,threshold=0.6)
    
    grouped_embeddings = [[embeddings[i] for i in group] for group in grouped]
    grouped_exchanges = [[exchanges[i] for i in group] for group in grouped]
    
    return grouped_exchanges,grouped_embeddings
    
def save_to_mem(exchanges:list[any]):
    pass
     
    
        