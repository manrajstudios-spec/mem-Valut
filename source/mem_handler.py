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
    test_exchanges = json.load(file)["exchanges"]
    
def get_old_data():
    try:
        stored_embeddings = np.load("Data/all_embeddings.npy")
    except:
        stored_embeddings = None
        
    try:
        stored_mean = np.load("Data/all_means.npy")
    except:
        stored_mean = None
    
    try:
        with open("Data/exchanges_keywords.pickle",'rb') as f:
            stored_groups,stored_keywords,stored_exchanges = pickle.load(f)
    except:
        stored_groups,stored_keywords,stored_exchanges = [],[],[]
        
    return stored_groups,stored_embeddings,stored_keywords,stored_exchanges,stored_mean
    

def save_data(groups,embeddings,exchanges,mean,keywords):
    with open("Data/exchanges_keywords.pickle","wb") as f:
        pickle.dump(obj=(groups,keywords,exchanges),file=f,protocol=pickle.HIGHEST_PROTOCOL)
    
    with open("Data/json_keys.json",'w') as f:
        json.dump({"ids":[(group.group_id,group.members) for group in groups],"exchanges":exchanges,"keywords":keywords},f,indent=4)
                            
    np.save("Data/all_embeddings",embeddings)
    np.save("Data/all_means",mean)

def make_groups(exchanges):
    embeddings = make_embeddings(exchanges,normalize=True)
    
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
    
    threshold = 0.7
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
    
    # make 1 dict per group 
    grouped_keywords_unpacked = [[keywords[g] for g in group] for group in groups]
    grouped_keywords = []
    
    for group_k in grouped_keywords_unpacked:
        to_add = {}
        
        for dictt in group_k:
            for keyword,value in dictt.items():
                to_add[keyword] = to_add.get(keyword,0) + value
        
        grouped_keywords.append(to_add)
    
    grouped_exchanges = [[exchanges[g] for g in group] for group in groups]
    grouped_embeddings = [np.vstack([embeddings[i] for i in group]) for group in groups]
    grouped_mean = [group_e.mean(axis=0) for group_e in grouped_embeddings]    
    
    return groups,grouped_exchanges,grouped_embeddings,grouped_mean,grouped_keywords

def save_to_mem(exchanges):
    groups,grouped_exchanges,grouped_embeddings,grouped_mean,grouped_keywords = make_groups(exchanges)
    print(groups)
    
    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()
    
    threshold = 0.56
    
    groups_to_add = []
    
    for group,group_mean,group_embedidngs,group_keywords,group_exchanges in zip(groups,grouped_mean,grouped_embeddings,grouped_keywords,grouped_exchanges):
        selected_groups = []
        
        for old_group in stored_groups:
            old_id = old_group.group_id
            old_mean = stored_mean[old_id]
            old_keywords = stored_keywords[old_id]
            
            print(f"new: {group_mean.shape}; old: {old_mean.shape}")
            mean_sim = group_mean @ old_mean
            
            keyword_sim = sum(value + old_keywords[keyword]  for keyword,value in group_keywords.items() if keyword in old_keywords)
            sim = mean_sim * 0.6 + np.log1p(keyword_sim) * 0.4
            print(sim)
            if sim >= threshold:
                selected_groups.append(old_group)
        
        old_len = len(stored_chats)
        print(f"groupembedidngs {group_embedidngs.shape}")
        
        if stored_embeddings is None:
            stored_embeddings = group_embedidngs.reshape(1,-1)
        else:
            stored_embeddings = np.concatenate((stored_embeddings,group_embedidngs))
      
        stored_chats = stored_chats + group_exchanges
        
        new_members = list(range(old_len, old_len + len(group_exchanges)))
                
        if not selected_groups:
            new_group_id = len(stored_groups) + len(groups_to_add)
            stored_keywords = [group_keywords]
            
            if stored_mean is None:
                stored_mean = group_mean.reshape(1,-1)
            else:
                group_mean = group_mean.reshape(1,-1)
                stored_mean = np.concatenate((stored_mean,group_mean))
    
            new_group = Group(group_id=new_group_id,members=new_members)
            groups_to_add.append(new_group)
            continue
        
        for group in selected_groups:
            group.members.extend(new_members)
            
            idd = group.group_id
            mean = stored_mean[idd]
            stored_mean[idd] = (mean * mean.size + group_mean * group_mean.size)/ (mean.size + group_mean.size) 
            
            for keyword,value in group_keywords:
                stored_keywords[old_id] = stored_keywords[old_id].get(keyword,0) + value
            
    stored_groups.extend(groups_to_add)
    
    save_data(groups=stored_groups,embeddings=stored_embeddings,exchanges=stored_chats,mean=stored_mean,keywords=stored_keywords)

def retrieve_major_groups(queries):
    embeddings = make_embeddings(queries,True)
    
    tuple_keywords = make_keywords(queries)
    keywords = [dict(exchange_keywords) for exchange_keywords in tuple_keywords]     
    
    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()

    selected_groups = []
    
    threshold = 0.5
    
    for embedding,keyword_dict in zip(embeddings,keywords):
        for group in stored_groups:
            group_id = group.group_ids
            group_mean = stored_mean[group_id]
            group_keywords = stored_embeddings[group_id]
            
            keyword_score = sum(value + group_keywords[keyword] for value,keyword in keyword_dict if keyword in group_keywords)
            
            embedding_score = group_mean @ embedding
            
            final_score = embedding_score * 0.7 + np.log1p(keyword_score) * 0.2
            
            print(final_score)        
            if final_score >= threshold:
                selected_groups.append(group)

save_to_mem(test_exchanges)

def rerank(groups,query_embeds):
    retrived_info = []
    stored_groups,stored_embeddings,stored_keywords,stored_chats,stored_mean = get_old_data()

    for group in groups:
        if group.graph is None:
            retrived_info