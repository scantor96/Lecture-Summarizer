import nltk
from nltk import word_tokenize 
import csv
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import gensim
import gensim.corpora as corpora
from pprint import pprint
import numpy as np

def listify(arg):
    return arg if isinstance(arg, list) else [arg] 

my_tokens = []
with open('/Users/averyfield/Downloads/my_csv.csv', 'r') as csvfile:
   reader = csv.DictReader(csvfile)
   for col in reader:
       line = col["Words"]
       my_tokens.append(word_tokenize(line))
tokens = [j for i in my_tokens for j in i]
my_text = [each_string.lower() for each_string in tokens]
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(each_word) for each_word in my_text]

#remove stopwords
filtered_text = []
symbols = ["!", "?", ",", ".", "..", "--", "``", "\'\'"]
stop_words = list(set(stopwords.words('english')))
newStopWords = ['would','get', 'actually', 'think', 'need', 'based', 'back', 'understand', 'fit', '\'ve',
                'kind', 'n\'t', '\'re', 'us', 'go', 'want', 'know', '\'s', 'next', 'tomorrow', 'case', 'different',
                'example', 'okay', 'one', 'like', '\'m', 'take', 'say', 'last', 'going', 'anybody', 'new', 'um',
                'way', 'john', 'start', 'much', 'use', 'thing', 'something', 'another', 'week', 'also', 'put',
                'basically', 'given', 'make', 'maybe', 'well', 'beginning', 'seen', 'producing', 'many', 'oh',
                'could', 'two', 'let', 'right', 'right', 'doe', 'simply', '\'ll', 'mary', 'wa', 'verb', 'whatever',
                'come', 'choice', 'essentially', 'give', 'thank', 'ha', 'lot', 'good', 'yeah', 'happens', 'bank',
                'see', 'everybody', 'look', 'name', 'three', 'choose', 'word', 'noun', 'yes', 'idea', 'meaning',
                'chinese', 'number', 'perplexity', 'every', 'talked', 'combination', 'night', 'line', 'proper',
                'said', 'even', 'thinking', 'talking', 'swift', 'point', 'saying', 'part', 'tell', 'else', 'age',
                'using', 'taylor', 'used', 'similar', 'first', 'guess', 'close', 'stuff', 'play', 'step', 'might',
                'really', 'turn', 'water', 'apart', 'book', 'star', 'work', 'behind ', 'exam', '\'d']

stop_words.extend(newStopWords)
for x in lemmas:
    if x not in stop_words and x not in symbols and len(x)>1:
        filtered_text.append(x)

#splits filtered text into 10 chunks, makes topic model for each chunk
chunks = np.array_split(filtered_text, 10)

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