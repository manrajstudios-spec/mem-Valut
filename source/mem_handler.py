import json
import pickle
import numpy as np
import pandas as pd
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
    
def load_mem():
    old_chats = pd.read_csv("Data/chat_mem.csv")
    chat_groups = pickle.load("Data/mem_groups.pickle")
    old_embeddings = np.load("Data/mem_embeddings.npy")
    mean_old_groups = np.load("Data/mem_mean.npy")
    
    
    return chat_groups.to_list(),old_chats,old_embeddings,mean_old_groups
    
def save_to_mem(exchanges:list[any]):
    threshold = 0.45
    chat_groups,old_chats,old_embeddings,mean_old_groups = load_mem() # mem groups array of arrays # mem chats pandas df cols chats,topics,mem_embeddings array of embeddings
    
    new_grouped_chats,new_grouped_embeddings = make_groups(exchanges=exchanges)
    
    new_group_mean = [em.mean(axis=0) for em in new_grouped_embeddings]
    
    new_group_mean = np.vstack(new_group_mean)
    
    sims = util.cos_sim(new_group_mean @ mean_old_groups)
    
    for sim,new_group_embeddings in zip(sims,new_grouped_embeddings):
        selected_group_ids = np.argsort(sim)[::-1]
        selected_groups = [chat_groups[i] for i in selected_group_ids]
        selected_embeddings = [old_embeddings[i] for i in selected_group_ids]
        
def add_new_group(grouped_chats,grouped_embeddings):
    