import json
from call_model import ask_model

with open("Data/prompts_schema/router_prompt.txt",'r') as file:
    sys_prompt = file.read()

with open("Data/prompts_schema/router_output_schema.json",'r') as file:
    json_schema = json.load(file)

prompt = [{"role":"system","content":sys_prompt}]

def route_msg(hist):
    json_out = ask_model(prompt+hist,schema=json_schema)
    parsed = json.loads(json_out)
    
    web = parsed["websearch"]
    doc = parsed["doc_retrieval"]
    rag = parsed["rag_retrieval"]
    user_prefrences = parsed["user_prefrences"]
    
    user_events = parsed["user_events"]
    user_decisions = parsed["user_decisions"]
    user_tasks = parsed["user_tasks"]
    
