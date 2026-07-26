import numpy
from ollama import ChatResponse,chat
from sentence_transformers import SentenceTransformer

chat_model = "granite4.1:3b"
embeddor = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
# chat(model=chat_model,messages=[],keep_alive=-1)             

def ask_model(hist):
    response:ChatResponse = chat(model=chat_model,messages=hist)
    
    return response.message.content

def embed_msg(messages):
    return embeddor.encode(messages,normalize_embeddings=False,convert_to_numpy=True)

