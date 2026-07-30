import json
import pickle
import numpy as np
import pandas as pd
from classes import Group
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
    stored_mean = np.load("all_means.npy")
    
    return stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean

def save_new_group(embeddings,chats,keywords,stored_groups,stored_embeddings,stored_keywords,stored_chats):
    new_group_id = len(stored_groups)
    old_len = len(stored_chats)
    
    new_embeddings = np.vstack([stored_embeddings,embeddings])
    
    new_keywords = stored_keywords + keywords
    new_chats = stored_chats + chats
    

    new_group = Group(group_id=new_group_id,members=[i+old_len for i in range(len(chats))])

    new_groups = stored_groups + new_group
    
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
    
    groups_mean = [] 
    grouped_embeddings_stacked = []
    
    for group_e in grouped_embeddings:
        stacked= np.vstack(group_e)
        grouped_embeddings_stacked.append(stacked)
        group_mean.append(np.mean(stacked,axis =0))
        
        
    grouped_keywords_scored = []
    
    for group_k in grouped_keywords:
        to_add = {}
        
        for keyword,value in zip(group_k.keys(),group_k.values()):
            stored = to_add.get(keyword, 0)
            to_add[keyword] = stored + value * (1 - stored)
            
        grouped_keywords_scored.append(to_add)

    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()

    threshold = 0.45
    
    for group_mean,new_keywords,new_embedding,new_chats in zip(groups_mean,grouped_keywords,grouped_embeddings,grouped_exchanges):
        selected_groups = []
        
        for old_group in stored_groups:
            old_mean = stored_mean[old_group.group_id]
            overall_keywords = stored_keywords[old_group.group_id]

            embedding_sim = group_mean @ old_mean

            keyword_sim = 0
            
            for keyword,value in zip(new_keywords.keys(),new_keywords.values()):
                if keyword in overall_keywords.keys():
                    keyword_sim += overall_keywords[keyword] + value   
            
            if keyword_sim:
                sim = embedding_sim + np.log1p(keyword_sim)
            else:
                sim = embedding_sim
            
            if sim >= threshold:
                selected_groups.append(old_group)
            
        if not selected_groups:
            stored_embeddings.extend()
            stored_chats.extend()
            continue
            
        for selected_group in selected_groups:
            pass
        