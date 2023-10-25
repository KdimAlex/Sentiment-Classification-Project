import spacy
import nltk
nltk.download('punkt')
import numpy as np
from nltk.tokenize import word_tokenize

# Load the spaCy model with GloVe embeddings
nlp = spacy.load("en_core_web_md")

def get_tok_embedding(paragraph):
  tokens = word_tokenize(paragraph)
  tok_embeddings = [nlp(token).vector for token in tokens]
  return tok_embeddings

def get_pos_embedding (paragraph):
  tokens = word_tokenize(paragraph)
  position = np.arange(len(tokens))[:, np.newaxis]
  div_term = np.exp(np.arange(0, 300, 2) * -(np.log(10000.0) / 300))
  pos_embeddings = np.zeros((len(tokens), 300))
  pos_embeddings[:, 0::2] = np.sin(position * div_term)
  pos_embeddings[:, 1::2] = np.cos(position * div_term)
  return pos_embeddings

def get_seg_embedding (paragraph):
  return seg_embeddings

def get_final_embedding (tok_embeddings, pos_embeddings, seg_embeddings):
  return pre_self_attention_embeddings

def get_self_attention_embeddings (pre_self_attension_embeddings):
  return input_embeddings_for_neural_net



paragraph = "My name is Tom, and I am unbelievably hungry."

token_embeddings = get_tok_embedding (paragraph)
position_embeddings = get_pos_embedding (paragraph)
# segment_embeddings = get_seg_embedding (paragraph)
# final_embedding = get_final_embedding (tok_embeddings, pos_embeddings, seg_embeddings)
# transformer_input = get_self_attention_embeddings(final_embedding)

#Tester for while I am coding:
def print_token_embeddings(tokens, embeddings, num_values=10, decimal_places=4):
    for token, embedding in zip(tokens, embeddings):
        limited_embedding = embedding[:num_values]
        formatted_embedding = [round(value, decimal_places) for value in limited_embedding]
        print(f"Token: {token}")
        print(f"Token Embedding (first {num_values} values of {len(embedding)} values): {formatted_embedding}\n")
tokens = word_tokenize(paragraph)
print_token_embeddings(tokens, token_embeddings)

def print_position_embeddings(tokens, position_embeddings, num_values=10, decimal_places=4):
    for token, embedding in zip(tokens, position_embeddings):
        limited_embedding = embedding[:num_values]
        formatted_embedding = [round(value, decimal_places) for value in limited_embedding]
        print(f"Position: {token}")
        print(f"Position Embedding (first {num_values} values of {len(embedding)} values): {formatted_embedding}\n")
tokens = word_tokenize(paragraph)
print_position_embeddings(tokens, position_embeddings)
