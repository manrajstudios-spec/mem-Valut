import numpy
from keybert import KeyBERT
from ollama import ChatResponse,chat
from sentence_transformers import SentenceTransformer

key_bert = KeyBERT()

chat_model = "granite4.1:3b"
embeddor = SentenceTransformer("multi-qa-distilbert-cos-v1")
# chat(model=chat_model,messages=[],keep_alive=-1)             

chat_stop_words = [
    'screen', 'page', 'button', 'click', 'clicked', 'clicking', 'link',
    'issue', 'problem', 'error', 'try', 'tried', 'trying', 'happened',
    'showing', 'working', 'failed', 'check', 'checked', 'checking',
    'said', 'told', 'see', 'look', 'want', 'need', 'user', 'assistant'
]

def ask_model(hist,schema=None):
    response:ChatResponse = chat(model=chat_model,messages=hist,format=schema)
    
    return response.message.content

def make_embeddings(messages,normalize=False):
    return embeddor.encode(messages,convert_to_numpy=True,normalize_embeddings=normalize)

def make_keywords(data):
    return key_bert.extract_keywords(data,diversity=0.4,stop_words='english')
