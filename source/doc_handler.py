import json
import time
import spacy
import pickle
import pymupdf
import pymupdf4llm
import numpy as np
# from call_model import make_embeddings,make_keywords

make_sentences = spacy.load("en_core_web_sm")

path = "Data/doc_data/attention.pdf"

def add_doc(path):
    start_time = time.monotonic()
    doc = pymupdf.open(path)
    
    text_per_page = {}
    tabel_pages = []
        
    for i,page in enumerate(doc):
        text_per_page[i] = page.get_text("text")
        
        tabel_finder = page.find_tables()
    
        if len(tabel_finder.tables) > 0:
            tabel_pages.append(i)
    
    tabels = pymupdf4llm.to_text(path,pages=tabel_pages,use_ocr=False)
    
    text = "\n".join(text_per_page.values())

    
    
add_doc(path)