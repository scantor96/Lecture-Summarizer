#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 00:40:20 2021

@author: Martina
"""

import nltk
from nltk import word_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from pprint import pprint
import numpy as np

def listify(arg):
    return arg if isinstance(arg, list) else [arg] 
#open csv file
my_tokens = []
def read_csvfile(filename):
    import csv
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        datalist = list(reader)
    return datalist

in_confirm = read_csvfile('/home/cantors2/Downloads/transcript.csv')

def confirm_obs(in_confirm):

    for row in in_confirm:
        col1 = row[2]
        my_tokens.append(word_tokenize(str(col1)))

confirm_obs(in_confirm)




#preprocess
tokens = [j for i in my_tokens for j in i]
my_text = [each_string.lower() for each_string in tokens]
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(each_word) for each_word in my_text]

#remove stopwords
filtered_text = []
symbols = ["!", "?", ",", ".", "..", "--", "``", "\'\'"]
stop_words = list(set(stopwords.words('english')))
newStopWords = ['would','get', 'actually', 'think', 'need', 'based', 'back', 'understand', 'fit', '\'ve', 'bird',
                'kind', 'n\'t', '\'re', 'us', 'go', 'want', 'know', '\'s', 'next', 'tomorrow', 'case', 'different',
                'example', 'okay', 'one', 'like', '\'m', 'take', 'say', 'last', 'going', 'anybody', 'new', 'um',
                'way', 'john', 'start', 'much', 'use', 'thing', 'something', 'another', 'week', 'also', 'put', 'sign',
                'basically', 'given', 'make', 'maybe', 'well', 'beginning', 'seen', 'producing', 'many', 'oh', 'came'
                'could', 'two', 'let', 'right', 'right', 'doe', 'simply', '\'ll', 'mary', 'wa', 'verb', 'whatever',
                'come', 'choice', 'essentially', 'give', 'thank', 'ha', 'lot', 'good', 'yeah', 'happens', 'bank', 'felt',
                'see', 'everybody', 'look', 'name', 'three', 'choose', 'word', 'noun', 'yes', 'idea', 'meaning', 'feel',
                'chinese', 'number', 'perplexity', 'every', 'talked', 'combination', 'night', 'line', 'proper', 'enough',
                'said', 'even', 'thinking', 'talking', 'swift', 'point', 'saying', 'part', 'tell', 'else', 'age',
                'using', 'taylor', 'used', 'similar', 'first', 'guess', 'close', 'stuff', 'play', 'step', 'might', 'day',
                'really', 'turn', 'water', 'apart', 'book', 'star', 'work', 'behind ', 'exam', '\'d', 'buy', 'yesterday',
                'hmm', 'mm', 'love', 'hate', 'clue', 'comma', 'appears', 'whether', 'tease', '15', 'hope', 'end', 'kind',
                'standpoint', 'york', 'making', 'better', 'provide', 'old', 'true', 'dive', 'people', 'half', 'ca', 'could',
                'obviously', 'came', 'trying', 'thought', 'sort', 'ahead', 'everything', 'ad', 'run', 'top']

stop_words.extend(newStopWords)
for x in lemmas:
    if x not in stop_words and x not in symbols and len(x)>1:
        filtered_text.append(x)

#splits filtered text into 20 chunks, makes topic model for each chunk
chunks = np.array_split(filtered_text, 20)

topics = []
for array in chunks:
    print("")
    print("Chunk:")
#doc2bow only accepts lists
    data_words = []
    for text in array:
        data_words.append(listify(text))
    
# Create Dictionary
    id2word = corpora.Dictionary(data_words)# Create Corpus
    texts = data_words# Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]


# number of topics
    num_topics = 1# Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics)# Print the Keyword in the topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]
    topics.append(lda_model.print_topics())

#put the words in topics into lists
n_topic = []
for topic in topics:
    s_topic = str(list(topic))
    no_digits = []
    bad_chars = [".", "*", "+", '"', '[', '(', ']', ')', ',', "'"]
    for c in s_topic:
        if not c.isdigit() and c not in bad_chars:
            no_digits.append(c)
            
    result = ''.join(no_digits)
    l_topic = result.split()
    n_topic.append(l_topic)

#map chunks to topic    
final = list(zip(chunks, n_topic))
my_dict = {}

for i in range(0, len(final)):   #why not 0?
    my_dict[i] = final[i]

#map updated chunks to time line
new_chunks = [my_dict[0][1]]
new_text = [my_dict[0][0]]  #added
idx = 0
for chunk in range(len(my_dict)):
    current_topics = my_dict[chunk][1]
    last_topic = new_chunks[idx]
    current_strings = my_dict[chunk][0]  #added
    last_strings = new_text[idx]       #added
    if len(list(set(current_topics) & set(last_topic))) > 2: #greater than or >=?
        new_chunks[idx]= list(set(current_topics + last_topic))
        new_text[idx]= (last_strings, current_strings)
    else: 
        new_chunks.append(current_topics)
        new_text.append(current_strings)
        idx+=1
        
'''for first word, we would take the first word of each chunk/
for the last word, we would take the last word, give that it only appears
in the chunk.'''
new_final = list(zip(new_chunks,new_text))
new_dict = {}
for i in range(0,len(new_final)):
    new_dict[i] = new_final[i]

#map chunks to time stamp
all_time_stamps = []
idx = 1
for entry in new_dict:
    ct_first = 0
    ct_last = 0
    first_word = new_dict[entry][0][0]
    last_word = new_dict[entry][0][0]
    time_stamp_start = []
    time_stamp_end = []
    for lst in in_confirm[idx:]:
        if ct_first < 1:
            if first_word in lst[2]:
                time_stamp_start.append(lst[0])
                ct_first+=1
        if ct_last <1:
            if last_word in lst[2]:
                time_stamp_end.append(lst[1])
                ct_last+=1
                idx= in_confirm.index(lst)
            

                
    all_time_stamps.append((time_stamp_start, time_stamp_end))
    
really_new_dict = []
for item in new_dict.values():
    really_new_dict.append((item[0],item[1]))            

mega_final = list(zip(all_time_stamps, really_new_dict))


    
        





