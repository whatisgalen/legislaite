# import google_pa
import pickle
import joblib

embeddings = joblib.load(open("/Users/galenmancino/repos/legislaite/vdb/embeddings2.pkl", "rb"))
from sklearn.metrics.pairwise import cosine_similarity



# Compute the cosine similarity between the embeddings
cosine_similarities = cosine_similarity(embeddings)

# Find the top 10 most similar embeddings to the query embedding
top_10_similarities = cosine_similarities.argsort()[:-10:-1]

# Retrieve the text sections associated with the top 10 most similar embeddings
text_sections = [texts[i] for i in top_10_similarities]
print(embeddings[0])

# similar_terms = google_pa.query(embeddings, query)

# for term in similar_terms:
#     print(term)