import csv

import os
from vocabulary import Vocabulary
# import spacy
import numpy as np
from numpy import asarray
from numpy import zeros



# https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
def get_text_embedding(filepath):

    MAX_NUM_WORDS = 100000

    EMBEDDING_SIZE = 200



    # Next, we'll create the tokenizers. A tokenizer is used to turn a string containing a sentence into a list of
    # individual tokens that make up that string, e.g. "good morning!" becomes ["good", "morning", "!"]. We'll start
    # talking about the sentences being a sequence of tokens from now, instead of saying they're a sequence of words.
    # What's the difference? Well, "good" and "morning" are both words and tokens, but "!" is a token, not a word.

    # spaCy has model for each language ("de" for German and "en" for English) which need to be loaded so we can access the
    # tokenizer of each model.
    # spacy_en = spacy.load('en')

    # Next, we create the tokenizer functions. These can be passed to TorchText and will take in the sentence as a string
    # and return the sentence as a list of tokens.

    # def tokenize_en(text):
    #     """
    #     Tokenizes English text from a string into a list of strings (tokens)
    #     """
    #     return [tok.text for tok in spacy_en.tokenizer(text)]
    #
    #
    #
    # SRC = Field(tokenize = tokenize_en,
    #             init_token = '<sos>', # start of padding short sentences
    #             eos_token = '<eos>', # end of sentence token
    #             lower = True)

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~\n\t'''
    sentences = []
    modelIdList = []
    categoryList = []
    caption_lengths = []
    with open(filepath) as f:
        data = csv.DictReader(f)
        for row in data:
            filepath = os.path.join(row["category"],row["modelId"]+".ply")
            if os.path.exists(filepath):
                    no_punct = ""
                    for char in row["description"]:
                        if char not in punctuations:
                            no_punct = no_punct + char
                    sentences.append(no_punct)
                    modelIdList.append(row["modelId"])
                    categoryList.append(row["category"])
            else:
                print(filepath,"does not exist")
    # print(sentences)

    voc = Vocabulary('test')
    # print(voc)

    for sent in sentences:
        voc.add_sentence(sent)


    # print('Token 4 corresponds to token:', voc.to_word(4))
    # print('Token "this" corresponds to index:', voc.to_index('table'))
    # print(voc.word2index)
    # print(voc.longest_sentence)
    max_input_len = voc.longest_sentence
    encoder_input_sequences = []
    for sentence in sentences:
        sent_tkns = ["sos"]
        sent_idxs = [0]
        for word in sentence.split(' '):
            sent_tkns.append(word)
            sent_idxs.append(voc.to_index(word))
        caption_lengths.append(len(sent_idxs))
        sent_idxs += ([2] +[0] * (max_input_len + 1 - len(sent_idxs)) ) # padding
        # print(sent_tkns)
        # print(sent_idxs)
        encoder_input_sequences.append(sent_idxs)
    encoder_input_sequences = np.array(encoder_input_sequences) # transformed to a numpy ndarray

    # print(encoder_input_sequences[231])

    #embedding
    embeddings_dictionary = dict()
    glove_file = open(r'glove.6B.200d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = asarray(records[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    # Recall that we have 3523 unique words in the input.
    # We will create a matrix where the row number will represent the integer value for the word and the columns will
    # correspond to the dimensions of the word. This matrix will contain the word embeddings for the words in our
    # input sentences.
    num_words = min(MAX_NUM_WORDS, voc.num_words + 1)
    
    embedding_matrix = zeros((num_words, EMBEDDING_SIZE))
    for word, index in voc.word2index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector

    # print(len(embeddings_dictionary["this"]))
    return embedding_matrix.astype(np.float32), encoder_input_sequences, modelIdList, categoryList,caption_lengths


