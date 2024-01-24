# import google.generativeai as genai
# import os




# # response.reply("Can you tell me a joke?")
# file = open("/Users/galenmancino/repos/markdown-of-the-california-legislation/markdown/CONS.md", "r")

# # import pprint
# # for model in genai.list_models():
# #     pprint.pprint(model)

# # model = genai.get_model('embedding-gecko-001')
# from spacy import pipeline
# import spacy

# nlp = pipeline("ner")

# spacy.load('en_core_web_sm')
# for doc in nlp(file):
#     tokens = []
#     for token in doc:
#         tokens.append({
#             'text': token.text,
#             'lemma_': token.lemma_,
#             'pos_': token.pos_,
#             'embeddings': token.embeddings
#         })

#     print(tokens)


# import pinecone
# client = pinecone.Client()
# dataset = client.get_dataset("my_dataset")
# column = dataset.get_column("embeddings")
# for embedding in embeddings:
#     column.upsert(embedding)