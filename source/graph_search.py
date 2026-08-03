import numpy as np
import random
from classes import Node
from call_model import make_embeddings
from sentence_transformers import util

sentences = [
    "I decided to use Granite 4.1 as the default extraction model for mem-vault.",
    "The mem-vault system should preserve raw conversations as evidence.",
    "I want to compare my greedy graph search against FAISS.",
    "My transformer training slowed down because the model was running partly on CPU.",
    "I fixed Ollama GPU support by installing the CUDA package.",
    "I plan to learn reinforcement learning after finishing mem-vault.",
    "The Unity agent should learn to manage hunger, energy, and money.",
    "I bought Sony WH-CH720N headphones because I do not like heavy bass.",
    "Hyprland feels less comfortable than GNOME for my normal workflow.",
    "I want to apply for a machine learning internship in Germany during university."
]

def make_graph(embeddings,threshold):
    sims = embeddings @ embeddings.T

    