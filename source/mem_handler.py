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

    new_sim = embedded_exchanges @ embedded_exchanges.T
    np.fill_diagonal(new_sim, -np.inf)
    
    keyword_score = []
    
    for i,outer in enumerate(keywords):
        to_Add = []
        for k,inner in enumerate(keywords):
            score = 0
            
            if k == i:
                to_Add.append(0)
                continue
            
            for keyword,value in zip(outer.keys(),outer.values()):
                if keyword in inner.keys():
                    score += value + inner[keyword]

            to_Add.append(score)
        
        keyword_score.append(to_Add)
        
    for i,row in enumerate(keyword_score):
        for k ,sim in enumerate(row):
            new_sim[i,k] *= np.log1p(sim)
            
    groups = []
    
    threshold = 0.45
    
    for i,sim in enumerate(new_sim):
        to_add = np.argwhere(sim>= threshold)
        
        to_add = np.argwhere(to_add >=i)
        
    
    threshold = 0.5
    last_groups = []

    for i,sim in enumerate(new_sim):
        sim = np.argwhere(sim>=threshold).flatten()
        sim = sim[sim > i].tolist()
        
        sim.append(i)
        
        if not last_groups:
            last_groups.append(set(sim))
            continue    
        
        if last_groups:
            sim = set(sim)
            founded = sim.copy()
            groups = []
            
            for last in last_groups:
                if last & founded:
                    founded |= last
                else:
                    groups.append(last)
            
            if not founded:
                groups.append(sim)
            else:
                groups.append(founded)  
                    
            last_groups = groups
    
    groups = last_groups
        
    grouped_exchanges = [[exchanges[g] for g in group] for group in groups]
    grouped_embeddings = [[embedded_exchanges[g] for g in group] for group in groups]
    grouped_keywords = [[keywords[g] for g in group] for group in groups]
    
    groups_mean = [] 
    grouped_embeddings_stacked = []
    
    for group_e in grouped_embeddings:
        stacked= np.vstack(group_e)
        grouped_embeddings_stacked.append(stacked)
        groups_mean.append(np.mean(stacked,axis =0))
        
    grouped_keywords_scored = []
    
    for row in grouped_keywords:
        to_add = {}
        for group_k in row: 
        
            for keyword,value in zip(group_k.keys(),group_k.values()):
                stored = to_add.get(keyword, 0)
                to_add[keyword] = stored + value * (1 - stored)
            
        grouped_keywords_scored.append(to_add)

    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()

    threshold = 0.45
    
    groups_to_add = []
    
    for group_mean,new_keywords,new_embedding,new_chats in zip(groups_mean,grouped_keywords,grouped_embeddings_stacked,grouped_exchanges):
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
                
        stored_embeddings = np.concat(stored_embeddings,new_embedding)    
        old_len = len(stored_chats)
        stored_chats.extend(new_chats)
        
        if not selected_groups:
            new_group = Group(group_id=len(groups)+len(groups_to_add),members=[i + old_len for i in range(len(new_chats))]) 
            groups_to_add.append(new_group)
            continue 
        
        for selected_group in selected_groups:
            selected_group.members.extend([i+old_len for i in range(len(new_chats))])
        