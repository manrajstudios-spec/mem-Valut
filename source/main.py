from call_model import ask_model

def ask_user(to_ask,options=[],empty=False):
    while True:
        user_input = input(to_ask)
        
        if empty and not user_input:
            return ""
        
        if options:
            if user_input in options:
                return user_input    
            
            continue
        
        if user_input:
            return user_input
                
chat_hist = [] # list of dicts goes aas context


while True:
    