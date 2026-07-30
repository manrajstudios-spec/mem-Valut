import numpy as np
from sentence_transformers import util
from call_model import make_embeddings

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


embeddings = make_embeddings(new_sents)

embeddings = np.vstack(embeddings)
print(embeddings.shape)
    
    
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


g5_embeddings = make_embeddings(group5)

g5_embeddings = np.vstack(g5_embeddings)

print(g5_embeddings.shape)

h = np.concat([embeddings,g5_embeddings])

print(f"V stack: {h.shape}\nH stack: {h.shape}")