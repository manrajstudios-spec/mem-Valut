import numpy as np
from call_model import make_embeddings
from sentence_transformers import util

sentences = [
    "The cat sat on the wooden fence.",
    "The dog sat near the wooden fence.",
    "A black cat chased the small mouse.",
    "The small dog chased a red ball.",
    "The teacher explained the math lesson.",
    "The student understood the math lesson quickly.",
    "She enjoys drinking hot coffee every morning.",
    "He enjoys drinking hot tea every morning.",
    "The programmer fixed the login bug.",
    "The developer fixed the database bug yesterday.",
    "The car stopped at the traffic light.",]

sent2 =[
    "The bus stopped at the traffic signal.",
    "The little girl read a mystery book.",
    "The little boy read a science book.",
    "Heavy rain flooded the city streets.",
    "Heavy wind damaged the city park.",
    "The chef cooked a delicious pasta dish.",
    "The chef prepared a delicious pizza.",
    "Artificial intelligence is transforming healthcare.",
    "Artificial intelligence is transforming education."
]

embedidngs = make_embeddings(sentences)
e2 = make_embeddings(sent2)

embedidngs = np.stack(embedidngs)
e2 = np.stack(e2)

combined = np.concat((embedidngs, e2))
print(combined.shape)