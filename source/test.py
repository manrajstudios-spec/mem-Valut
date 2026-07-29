import numpy as np
from call_model import make_embeddings,make_keywords
from sentence_transformers import util 

group1 = [
    "I trained a small transformer on the TinyStories dataset.",
    "The model uses rotary positional embeddings inside attention.",
    "Training loss decreased quickly during the first epoch.",
    "My laptop became very hot while training the model.",
    "The CPU temperature reached nearly ninety degrees Celsius.",
    "I stopped training to avoid damaging the laptop.",
    "I may rent a cloud GPU for the full training run.",
    "A larger GPU would allow me to increase the batch size.",
    "I want to compare local training with cloud training.",
    "The final model will be tested on longer text sequences."
]

group1_keys = make_keywords(group1)

g1_keys = {}

for i in group1_keys:
    for keyword,value in i:
        old = g1_keys.get(keyword,0)
        
        g1_keys[keyword] = old + value
                     
group2 = [
    "My university hostel rooms were already fully booked.",
    "I started searching for a paying guest room near campus.",
    "The new room should be within walking distance of the university.",
    "My parents were worried about the housing situation.",
    "They wanted to make sure the area was safe.",
    "I was less stressed because I already had a backup plan.",
    "University classes will begin in the middle of August.",
    "I need to finish several coding projects before classes begin.",
    "My university email account has already been activated.",
    "The student email may provide access to free developer tools."
]
group_2_keys = make_keywords(group2)

g_2_keys = {}

for i in group_2_keys:
    for keyword,value in i:
        old = g_2_keys.get(keyword,0)
        g_2_keys[keyword] = old + value

group3 = [
    "I am building a browser extension for YouTube Music.",
    "The extension reads the songs currently present in the queue.",
    "It can reshuffle the playlist when the song order looks bad.",
    "A classifier will judge whether the shuffle is enjoyable.",
    "The model will use information about the first ten songs.",
    "Song order may be as important as the songs themselves.",
    "The backend of the application will use FastAPI.",
    "The browser extension will send queue data to the backend.",
    "The backend will return a score for the current playlist.",
    "Later I may add gesture controls for playback and volume."
]

group3_keys = make_keywords(group3)

g3_keys = {}

for i in group3_keys:
    for keyword,value in i:
        g3_keys[keyword] = g3_keys.get(keyword,0) + value

group4 = [
    "The walls in my room were recently covered with putty.",
    "I want to use wallpaper instead of normal wall paint.",
    "A false ceiling may improve the overall appearance of the room.",
    "The gaming table will be placed near the front window.",
    "The bed will remain against the left wall.",
    "I learned how to connect simple switches and electrical plugs.",
    "A switch can route a plug between inverter and main power.",
    "The live and neutral wires must be identified carefully.",
    "Incorrect wiring can cause electric shock or a short circuit.",
    "Complicated electrical work should still be handled by an electrician."
]
group_4_keys = make_keywords(group4)

g4_keys = {}

for i in group_4_keys:
    for keyword,value in i:
        old = g4_keys.get(keyword,0)
        g4_keys[keyword] = old + value
        
group5 = [
    "I am developing a memory system called Mem Vault.",
    "The system stores facts, preferences, decisions, and goals.",
    "Each saved memory will also contain an embedding vector.",
    "Similar memories will be grouped into topic clusters.",
    "A mean embedding can represent the overall meaning of a cluster.",
    "BM25 can detect important keywords inside stored conversations.",
    "Dense embeddings are better at finding semantic similarities.",
    "Sparse and dense retrieval scores can be combined.",
    "A router will decide whether the query needs personal memory.",
    "The retrieved memories will be passed to the main answering model."
]

group_5_keys = make_keywords(group5)

g_5_keys = {}

for i in group_5_keys:
    for keyword,value in i:
        old = g_5_keys.get(keyword,0)
        g_5_keys[keyword] = old + value

new_sents = [
    "I trained a small transformer on the TinyStories dataset.",
    "The model uses rotary positional embeddings in its attention layers.",
    "Training stopped because my laptop became too hot.",
    "I may use a cloud GPU for the next training run.",

    "My university hostel was already fully booked.",
    "I started looking for a paying guest room near campus.",
    "My parents were worried about the safety of the new area.",

    "I am building a browser extension for YouTube Music.",
    "The extension reads the first ten songs in the current queue.",
    "A classifier decides whether the shuffled song order is good.",
    "The backend sends the playlist score using FastAPI.",

    "The walls of my room were recently covered with putty.",
    "I want to install wallpaper and a false ceiling.",
    "The gaming table will be placed near the window.",

    "A solar eclipse happens when the Moon blocks sunlight from reaching Earth.",
    "Elephants can recognize themselves in mirrors.",
    "The bakery near my house sells fresh bread every morning.",
    "A broken keyboard key can sometimes be fixed by cleaning underneath it.",
    "Rainwater collected in open containers can attract mosquitoes.",
    "Chess players often plan several moves before touching a piece."
]

old_groups = [group1,group2,group3,group4,group5]

old_group_embeds = [make_embeddings(group,normalize=False) for group in old_groups]

old_group_mean = [group.mean(axis=0) for group in old_group_embeds]

old_group_mean = np.vstack(old_group_mean)

new_embeds = make_embeddings(new_sents,normalize=False)

groups = util.community_detection(new_embeds,min_community_size=1,threshold=0.29)
print(groups)
grouped_embeddings = [[new_embeds[i] for i in group] for group in groups]
group_means = []

for group in grouped_embeddings:
    group = np.stack(group)
    group_means.append(group.mean(axis=0))

group_means = np.vstack(group_means)

print(group_means.shape)

sims = util.dot_score(group_means,old_group_mean)
print(sims.shape)

threshold = 0.35

old_keywords = [g1_keys,g_2_keys,g3_keys,g4_keys,g_5_keys]

grouped_new_sents = [[new_sents[i] for i in g] for g in groups]
print(grouped_new_sents)

grouped_keywords_new = []

print(len(grouped_new_sents) == len(groups))

for g in grouped_new_sents:
    keys = make_keywords(g)
    keywords = {}

    if isinstance(keys[0],tuple):
        for keyword,value in keys:
            keywords[keyword] = keywords.get(keyword,0) + value
        
        grouped_keywords_new.append(keywords)
        continue
    
    for sent in keys:
        for keyword,value in sent:
            stored = keywords.get(keyword,0)
            keywords[keyword] = stored + value
    
    grouped_keywords_new.append(keywords)

final_score = []

for sim,keywords in zip(sims,grouped_keywords_new):
    listt = []
    for g_sim,old_keyword in zip(sim,old_keywords):
        to_Add = 0
        for keyword,value in zip(keywords.keys(),keywords.values()):
            if keyword in old_keyword.keys():
                to_Add += old_keyword[keyword] + value
        
        if to_Add:
            print(to_Add)
            listt.append(g_sim * np.log1p(to_Add))
        else:
            listt.append(g_sim * 0.6)
            
    final_score.append(listt)
            
print(final_score)