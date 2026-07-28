import json
import pickle
import numpy as np
import pandas as pd
from classes import Group
from call_model import embed_msg
from sentence_transformers import util
from call_model import embed_msg,ask_model,make_keywords
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
    
def load_mem():
    old_chats = pd.read_csv("Data/chat_mem.csv")
    chat_groups = pickle.load("Data/mem_groups.pickle")
    old_embeddings = np.load("Data/mem_embeddings.npy")
    mean_old_groups = np.load("Data/mem_mean.npy")
    
    
    return chat_groups.to_list(),old_chats,old_embeddings,mean_old_groups
    
def save_to_mem(exchanges:list[any]):
    threshold = 0.45
    old_groups,old_chats,old_embeddings,mean_old_groups = load_mem() # mem groups array of arrays # mem chats pandas df cols chats,topics,mem_embeddings array of embeddings
    
    new_grouped_chats,new_grouped_embeddings = make_groups(exchanges=exchanges)
    
    new_group_mean = []
    
    for ge in new_grouped_embeddings:
        if ge.ndim == 1:
            ge = ge.reshape(1,-1)
        
        new_group_mean.append(ge.mean(axis=0))        
    
    np.vstack(new_group_mean)
    
    grouped_sims = new_group_mean @ mean_old_groups.T
    
    for new_group_chats,sim_mit_old_group_embeddings in zip(new_grouped_chats,grouped_sims):
        new_keywords = make_keywords(new_grouped_chats)