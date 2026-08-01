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
    stored_embeddings = np.load("all_embeddings.npy")
    
    stored_groups,stored_keywords,stored_exchanges = pickle.load("exchanges_keywords.pickle")
    stored_mean = np.load("all_means.npy")
    
    return stored_groups,stored_embeddings,stored_keywords,stored_exchanges,stored_mean
    

def save_data(groups,embeddings,exchanges,mean,keywords):
    pickle.dump((groups,exchanges,keywords),"exchanges_keywords.pickle")
    np.save("all_embeddings",embeddings)
    np.save("all_means",mean)

def make_groups(exchanges):
    embeddings = make_embeddings(exchanges)
    
    embedding_sim = embeddings @ embeddings.T    
    tuple_keywords = make_keywords(exchanges)
    
    keywords = [dict(exchange_keywords) for exchange_keywords in tuple_keywords]    

    n = len(exchanges)
    keywords_score = np.zeros((n,n),dtype=float)

    for i,outer in enumerate(keywords):
        for j in range(i+1,n):
            inner = keywords[j]

            score = sum((value+inner[keyword]) for keyword,value in outer.items() if keyword in inner)

            keywords_score[i,j] = score
            keywords_score[j,i] = score

    
    sims = 0.6 * embedding_sim + 0.4 * np.log1p(keywords_score)
    
    np.fill_diagonal(sims,float("-inf"))
    
    threshold = 0.5
    groups = []
    last_groups = []

    for i,sim in enumerate(sims):
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
    
    grouped_keywords_unpacked = [[keywords[g] for g in group] for group in groups]
    grouped_keywords = []
    
    for group_k in grouped_keywords_unpacked:
        to_add = {}
        
        for keyword,value in group_k.items():
            to_add[keyword] = to_add.get(keyword,0) + value
        
        grouped_keywords.append(to_add)
    
    grouped_exchanges = [[exchanges[g] for g in group] for group in groups]
    grouped_embeddings = [np.vstack([embeddings[i] for i in group]) for group in groups]
    grouped_mean = [group_e.mean(axis=0) for group_e in grouped_embeddings]    
    
    return groups,grouped_exchanges,grouped_embeddings,grouped_mean,grouped_keywords

def save_to_mem(exchanges):
    groups,grouped_exchanges,grouped_embeddings,grouped_mean,grouped_keywords = make_groups(exchanges)
    
    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()
    
    threshold = 0.56
    
    groups_to_add = []
    
    for group,group_mean,group_embedidngs,group_keywords,group_exchanges in zip(groups,grouped_mean,grouped_embeddings,grouped_keywords,grouped_exchanges):
        selected_groups = []
        
        for old_group in stored_groups:
            old_id = old_group.grouop_id
            old_mean = stored_mean[old_id]
            old_keywords = stored_keywords[old_id]
            
            mean_sim = group_mean @ old_mean
            
            keyword_sim = sum(value + old_keywords[keyword]  for keyword,value in group_keywords.items() if keyword in old_keywords)
            
            sim = mean_sim * np.log1p(keyword_sim)
            
            if sim >= threshold:
                selected_groups.append(old_group)
        
        new_group_id = len(stored_groups)
        old_len = len(stored_chats)
        stored_embeddings = np.concat(stored_embeddings,group_embedidngs)
        stored_chats = stored_chats + group_exchanges
        
        new_memebers = [i+old_len for i in range(old_len+len(group_exchanges))]
        
        if not selected_groups:
            new_group = Group(group_id=new_group_id,members=new_memebers)
            groups_to_add.append(new_group)
            continue
        
        for group in selected_groups:
            group.memebers.extend(new_memebers)
            
    stored_groups.extend(groups_to_add)
    
    save_data()