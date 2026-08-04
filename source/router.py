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
    
    return parsed

while True:
    user_input =input("Enter: ")
    
    print(route_msg([{"role":"user","content":user_input}]))    