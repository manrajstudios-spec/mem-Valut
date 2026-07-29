import json
import pickle
import numpy as np
import pandas as pd
from call_model import make_embeddings
from sentence_transformers import util
from call_model import make_embeddings,ask_model,make_keywords
from graph_search import make_graph,get_similar,add_to_graph

test_exchanges = []

with open("Data/test_exchanges.json",'r') as file:
    test_exchanges = json.load(file)
    
def get_old_data():
    stored_groups = pickle.load("groups.pickel")
    stored_embeddings = np.load("all_embeddings.npy")
    stored_keywords = pd.read_csv("all_keywords.csv")
    stored_chats = pd.read_csv("all_chats.csv")
    
    return stored_groups,stored_embeddings,stored_keywords,stored_chats

def add_to_mem(exchanges):
    embedded_exchanges = make_embeddings(exchanges,normalize=False)
    tuple_keywords = make_keywords(exchanges)
    
    keywords = []
    
    for k in tuple_keywords:
        to_add = {}
        for keyword,value in k:
            to_add[keyword] = value
        keywords.append(to_add)
    
    groups = util.community_detection(embedded_exchanges,threshold=0.45,min_community_size=1,show_progress_bar=True)
    
    grouped_exchanges = [[exchanges[g] for g in group] for group in groups]
    grouped_embeddings = [[embedded_exchanges[g] for g in group] for group in groups]
    grouped_keywords = [[keywords[g] for g in group] for group in groups]
    
    group_mean = [] 
    
    for group_e in grouped_embeddings:
        group_mean.append(np.vstack(group_e))
    
    grouped_keywords_scored = []
    
    for group_k in grouped_keywords:
        to_add = {}
        
        for keyword,value in zip(group_k.keys(),group_k.values()):
            stored = to_add.get(keyword, 0)
            to_add[keyword] = stored + value * (1 - stored)
            
        grouped_keywords_scored.append(to_add)

    stored_groups,stored_embeddings,stored_keywords,stored_chats = get_old_data()

    
    def save_new_group(embeddings,chats,keywords,stored_groups,stored_embeddings,stored_keywords,stored_chats):
        new_group_id = len(stored_groups)
        old_len = len(chats)
        
        new_embeddings = np.vstack([stored_embeddings,embeddings])
        
        new_keywords = stored_keywords + keywords
        new_chats = stored_chats + chats
        
        new_group = {"group_id":new_group_id,"members":[i + old_len for i in range(len(chats))]}
        