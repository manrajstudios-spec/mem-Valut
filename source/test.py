import ollama
from ollama import ChatResponse,chat
from sentence_transformers import SentenceTransformer

chat_model = "granite4.1:3b"
embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

chat(model=chat_model,messages=[],keep_alive=-1)

response: ChatResponse = chat(model=chat_model,messages=[{"role":'user',"content":"HI"}])

print(response.message.content)

print(embedding_model.encode("HI"))