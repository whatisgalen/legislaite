



import gensim
import spacy
from sklearn.metrics.pairwise import cosine_similarity


text_filepath = "/Users/galenmancino/repos/markdown-of-the-california-legislation/markdown/CONS.md" # path to text file
with open(text_filepath, 'r') as f: # Open the text file
    text = str(f.read())

def create_phrases(text):
    phrases = []
    for sentence in text.split('\n'):
        for phrase in sentence.split(' '):
            phrases.append(phrase)

    for phrase in text.split('.'):
        phrases.append(phrase)
    return phrases

def create_vectors(phrases, model):
    """Creates a vector representation for each phrase."""
    vectors = []
    for phrase in phrases:
        vectors.append(model.wv[phrase])
    return vectors

def measure_similarity(vectors, user_input):
  """Measures the similarity between the user's input and each phrase."""
  similarities = []
  for vector in vectors:
    similarities.append(cosine_similarity(vector, user_input))
  return similarities

def get_best_match(similarities):
  """Returns the phrase that has the highest similarity score."""
  best_match = None
  best_score = 0
  for similarity in similarities:
    if similarity > best_score:
      best_match = phrases[similarities.index(similarity)]
      best_score = similarity
  return best_match


# Create a list of all the phrases in the text.
phrases = create_phrases(text)
from tensorflow import keras

# Create a vector representation for each phrase.
# vectors = create_vectors(phrases, gensim.models.Word2Vec.load('google_bert'))

from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
# unmasker("Hello I'm a [MASK] model.")
vectors = create_vectors(phrases, gensim.models.Word2Vec.load('bert-base-uncased'))

# Measure the similarity between the user's input and each phrase.
similarities = measure_similarity(vectors, input('Enter a phrase: '))

# Get the best match.
best_match = get_best_match(similarities)

# Print the best match.
print('The best match is:', best_match)