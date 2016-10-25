from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import re
import logging
import lxml.html
import csv

###################################_CONFIGURATION_#########################################
# toggle between article title [1] and article content [0]
TOGGLE = 0
NUM_TOPICS = 4
NUM_WORDS_IN_TOPIC = 15
NUM_ITERATIONS = 1000
NAME_OF_INPUT_FILE = 'INPUT/Feb27Wkndrss.csv'
NAME_OF_OUTPUT_FILE = 'RESULTS/CONTENT/Feb27Wknd.csv'
# NAME_OF_OUTPUT_FILE = 'RESULTS/CONTENT/output.csv'
##################################_END_CONFIGURATION_######################################


# uncomment for detailed logging info
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# remove tag fucntion to remove html tags (for pictures etc.)
def remove_tags(text):
    return lxml.html.fromstring(text).text_content()

# tokenizer rules
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()



# document list
message_list = []

# freq counter for number of lines
freq = 0

# read in file, remove tags, filter special characters and build corpus list
with open(NAME_OF_INPUT_FILE, 'r') as f:
    for line in f:
        data_list = line.strip().split(",")

        message = data_list[TOGGLE]
        if message:
            message = remove_tags(message)
            message = re.sub(r'[^a-zA-Z0-9 ]',r'',message)

        message_list.append(message)
        freq = freq + 1


# list for tokenized documents in loop
texts = []


# loop through document list
for i in message_list:

    # clean and tokenize document string
    raw = i.lower()
    tokens = tokenizer.tokenize(raw)

    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]

    # stem tokens
    # stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

    # add tokens to list
    texts.append(stopped_tokens)


dictionary = corpora.Dictionary(texts)


# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]


# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word = dictionary, passes=NUM_ITERATIONS)


lda_results = (ldamodel.print_topics(num_topics=NUM_TOPICS, num_words=NUM_WORDS_IN_TOPIC))


# open file for ouput, make sure the line terminator is newline (\n) and then print field names
f = open(NAME_OF_OUTPUT_FILE, 'wt')
writer = csv.writer(f, lineterminator='\n')
writer.writerow(('Topic', 'Word', 'Score'))

# Data transformation and print to csv file
for topic in lda_results:
    topic_num = topic[0]
    topic_words = topic[1]
    topic_words = topic_words.replace("+", " ")
    topic_words = topic_words.split()
    for word_score in topic_words:
        word_score = word_score.split('*')
        word = word_score[1]
        score = word_score[0]
        print topic_num, ':', word, ':', score
        writer.writerow((topic_num, word, score))