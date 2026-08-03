import numpy as np
from call_model import make_embeddings,make_keywords

groups = [
    "I decided to use Granite 4.1 as the default extraction model for mem-vault.",
    "The mem-vault system should preserve raw conversations as evidence.",
    "I want to compare my greedy graph search against FAISS.",
    "My transformer training slowed down because the model was running partly on CPU."
]

new_sents = ["I fixed Ollama GPU support by installing the CUDA package.",
    "I plan to learn reinforcement learning after finishing mem-vault.",
    "The Unity agent should learn to manage hunger, energy, and money."]

new_emebddings = make_embeddings(new_sents)
new_emebddings = np.stack(new_emebddings)

group_embeddings = np.stack(make_embeddings(groups))

embedding_sim = new_emebddings @ group_embeddings.T

new_keywords = make_keywords(new_sents)
sent_keys = make_keywords(groups)

keywords_group = [dict(x) for x in sent_keys]
keywords_me = [dict(x) for x in new_keywords]

keyword_score = []

for new_keyword in keywords_me:
    score = [sum(value+group_key[keyword] for keyword,value in new_keyword.items() if keyword in group_key) for group_key in keywords_group]
    keyword_score.append(score)
    
keyword_score = np.array(keyword_score)

sims = embedding_sim * 0.7 + 0.3 * np.log1p(keyword_score)

print(sims)

selected_ids = []

for sim in sims:
    ids = np.argwhere(sim>=0.4).flatten()
    selected_ids.append(ids)

print(selected_ids)