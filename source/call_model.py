import numpy
from ollama import ChatResponse,chat
from sentence_transformers import SentenceTransformer

chat_model = "granite4.1:3b"
embeddor = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
# chat(model=chat_model,messages=[],keep_alive=-1)             

def ask_model(hist,schema=None):
    response:ChatResponse = chat(model=chat_model,messages=hist,format=schema)
    
    return response.message.content

def embed_msg(messages,normalize=False):
    return embeddor.encode(messages,convert_to_numpy=True,normalize_embeddings=True)
