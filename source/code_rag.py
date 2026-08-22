import ast
import time
import numpy as np
from call_model import embed_code

def save_code(path):
    code=''
    start_time = time.monotonic()
    with open(path,'r',encoding='utf-8')as f:    
        code=f.read()
    
    parsed = ast.parse(code)
    blocks = []
    
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            blocks.append(ast.get_source_segment(code, node))
    
    split_ids = []
     
    stack = blocks
    
    final_blocks = []

    while stack:
        cur = stack.pop()
        
        if len(cur)>2000:
            n = len(cur)
            k = (n + 2000 - 1) // 2000
            
            chunks = [cur[i * n // k:(i + 1) * n // k] for i in range(k)]
            split_ids.append((len(final_blocks)-1,k))
            final_blocks.extend(chunks)
            
    print(f"parsing done: {time.monotonic() - start_time}")
    start_time = time.monotonic()
    emebddings = embed_code(blocks)

    final_embeddings = []
    i=0
    
    while i < len(emebddings):
        split_id = []
        
        for s in split_ids:
            if i in s[0]:
                split_id = s
                break
                
        if split_id:
            ...
            # do mean of split tokens
            

    
if __name__ == "__main__":
    save_code("call_model.py")