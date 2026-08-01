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
    "The car stopped at the traffic light.",
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

gorupss = util.community_detection(embeddings=embedidngs,min_community_size=1,threshold=0.2)

grouped_embeddings = [np.vstack([embedidngs[i] for i in group]) for group in gorupss]

mean = [group_e.mean(axis=0) for group_e in grouped_embeddings]

print(mean[0])