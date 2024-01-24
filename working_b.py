import spacy
import os
import time
import google.generativeai as genai
os.environ['G_API_KEY'] = "AIzaSyDQkU1_oF9wOrNgNGgg4C822UFvfXYI2N0"
os.environ['P_API_KEY'] = "13f78aa3-99c3-45a7-b0f7-ee636e5161de"
genai.configure(api_key=os.environ['G_API_KEY'])
# import pprint
# for model in genai.list_models():
#     pprint.pprint(model)
# model = genai.get_model('models/embedding-gecko-001') # Load the embedding-gecko-001 model
file = open("/Users/galenmancino/repos/legislaite/workingoutput.txt", "w")
# Load the spaCy English language model
# nlp = spacy.load('en_core_web_sm')
text_filepath = "/Users/galenmancino/repos/markdown-of-the-california-legislation/markdown/CONS.md" # path to text file
with open(text_filepath, 'r') as f: # Open the text file
    text = str(f.read())
# tokens = nlp(text) # # Tokenize the text


# Load the text file into the spacy model
# doc = nlp(text)

# Tokenize the text file
# tokens = doc.tokens

import time
import joblib
# for i, t in enumerate(tokens[0:1000]):
#     print(i, t)

def generate_embeddings(tokens, batch_size=299, timer=61):
    """Generates embeddings for the given tokens.

    Args:
    tokens: A list of tokens.
    batch_size: The number of tokens to batch together.

    Returns:
    A list of embeddings.
    """
    embeddings = []
    start_time = time.time()
    for i in range(0, len(tokens)):
        # print(tokens[i])
        embeddings.append(genai.generate_embeddings(
            model='models/embedding-gecko-001',
            text=tokens[i].text
        ))
        if (i > 0 and i % batch_size == 0) or i == len(tokens) - 1:
            print(f"processed {i/len(tokens)}")
            time.sleep(timer)
    return embeddings

def break_text_into_sections(text):
    """Breaks the given text into sections every time the word 'SEC' appears.

    Args:
    text: The text to break into sections.

    Returns:
    A list of strings, where each string is a section of the text.
    """
    sections = []
    current_section = []
    world_list = text.split()
    # print(len(world_list))
    for word in world_list:
        if "SEC" in word:
            # print("SUCCESS")
            if current_section:
                sections.append(" ".join(current_section))
                current_section = []
            else:
                current_section.append(word)
    if current_section:
        sections.append(" ".join(current_section))
    return sections

sections = break_text_into_sections(text)
# print(sections[1])
# for s in sections:
#    print(s)
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# if True is False:
# embedding_model = joblib.load("embedding_model.pkl")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
embeddings = vectorizer.fit_transform(sections)
print(embeddings[0])

cosine_similarities = cosine_similarity(embeddings)

# Find the top 10 most similar embeddings to the query embedding
top_10_similarities = cosine_similarities.argsort()[:-10:-1]
top_10_similarities = top_10_similarities.tolist()
# print(top_10_similarities)

# Retrieve the text sections associated with the top 10 most similar embeddings
text_sections = [sections[i] for i in top_10_similarities[0]]
print(text_sections)
# print(text_sections)
# print(embeddings[0])





# vectors = vectorizer.transform(tokens)
# new_text = "racial discrimination"
# vectors = vectorizer.transform(text)
# joblib.dump(vectors, "vectors.pkl")

# vectors = scikit_learn.externals.joblib.load("vectors.pkl")

# sklearn.metrics.pairwise.cosine_similarity

# vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000) # Create a new vectorizer
# new_vectors = vectorizer.transform(new_text) # Vectorize the new text
# all_embeddings = embedding_model.predict(new_vectors) # Create the embeddings


# from scikit_learn.feature_extraction.text import TfidfVectorizer
import pickle

# vectorizer = TfidfVectorizer()

# embeddings = vectorizer.fit_transform(["The quick brown fox jumps over the lazy dog", "The cat sat on the mat"])

# from google.ai.generativelanguage import TextServiceClient

# text_service_client = TextServiceClient(project_id='my-project-1510178609167', ) 429
# all_embeddings = generate_embeddings(tokens[0:(299*2)])
# for token in tokens:
#     google_embeddings = genai.generate_embeddings(
#         model='models/embedding-gecko-001', text=token.text
#     )
#     all_embeddings.append(google_embeddings)

# import scikit_learn.feature_extraction.text as text_
# import sklearn.externals.joblib as joblib
# vectorizer = text_.TfidfVectorizer(ngram_range=(1, 2)) # Create a vectorizer
# vectorizer.fit(all_embeddings) # Fit the vectorizer to the embeddings
# X = vectorizer.transform(all_embeddings) # Transform the embeddings into a vector representation
if True is False:
    joblib.dump(embeddings, '/Users/galenmancino/repos/legislaite/vdb/embeddings2.pkl') # Save the vector representation to a vector database

# import google_pa
# Write the string to the file
# string = str(google_embeddings)
# file.write(string)

# def main():
#   tokens = ['This is a test.', 'This is another test.', 'This is the last test.']
#   embeddings = generate_embeddings(tokens)
#   print(embeddings)

# if __name__ == '__main__':
#   main()

# Add the embeddings to the document
# for token in tokens:
#     document.add_embeddings(model.embed(token))

# # Save the document to the index
# index.save_document(document)

