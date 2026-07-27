from call_model import embed_msg
from sentence_transformers import util


sentences = [
    # Group 1: Language-model training
    "I am training a small transformer model on a dataset of short stories.",
    "The neural network learns language patterns from sequences of tokens.",
    "I am experimenting with attention layers to improve text generation.",
    "The language model needs more training to produce coherent responses.",

    # Group 2: Conversation memory system
    "I am building a Python application that stores user conversations.",
    "The program retrieves old messages when the user refers to past discussions.",
    "I am organizing saved information into projects and memory categories.",
    "The application uses conversation history to provide personalized replies."
]

main = ["The AI assistant processes text before deciding what information it needs.",
    "A language model can analyze new messages and route them to the correct system.",
    "The application uses embeddings to compare the meaning of different conversations.",
    "Past messages can provide useful context for generating more relevant responses.",
    "The model identifies whether a user message belongs to an existing project.",
    "Stored conversation data can be retrieved and added to the model's prompt.",
    "Semantic similarity helps the assistant locate information related to the current topic.",
    "The system combines language understanding with structured memory retrieval.",
    "A smaller model can classify messages before a larger model generates the response.",
    "The assistant improves its replies by combining recent context with saved knowledge."
]


embeddings = embed_msg(sentences)

groups = util.community_detection(embeddings=embeddings,threshold=0.53,min_community_size=1)

print(groups)

print(f"shape: {embeddings.shape} \nmean: {embeddings.mean(axis=0,keepdims=True).shape}")



