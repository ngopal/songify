import sys
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
from Helpers.SQLhelper import SQL
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import spacy
from nltk.corpus import words
from collections import defaultdict
import logging
from nltk.stem import *
from nltk.stem.wordnet import WordNetLemmatizer

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Load Database
DBhelper = SQL()

all_tables = ["19PgP2QSGPcm6Ve8VhbtpG", "37i9dQZF1DX1ewVhAJ17m4", "37i9dQZF1DX5bjCEbRU4SJ","3vxotOnOGDlZXyzJPLFnm2","49oW3sCI91kB2YGw7hsbBv","4tZSI7b1rnGVMdkGeIbCI4","68bXT1MZWZvLOJc0FZrgf7","6wz8ygUKjoHfbU7tB9djeS","7eHApqa9YVkuO6gELsju2j"]
# all_tables = ["19PgP2QSGPcm6Ve8VhbtpG", "37i9dQZF1DX1ewVhAJ17m4"]

lyrics = []

for p in all_tables:
    pdf = pd.read_sql_query("SELECT * FROM \"" + p +"\"", DBhelper.engine)
    lyrics = lyrics + list(pdf["lyrics"].dropna())

print(lyrics)

#### Form the right data structures
from gensim import corpora

documents = lyrics
stoplist = set('for a of the and to in yeah who don like got want know baby let hey come tell need said way thing cause little look'.split())
# texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# Use NLTK tokenizer to correctly split contracted words
tokenizer = RegexpTokenizer(r'\w+')
# Use NLTK stopwords to make sure common words are not being stored in list
#stopset = set(stopwords.words('english'))
stopset = set(spacy.en.STOP_WORDS)
en_words = set(words.words())
stemmer = SnowballStemmer("english")


# Should I make the top N ranked words stopwords in the spacy dataset?
def notStopWord(word):
    if word not in stopset:
        if word not in stoplist:
            return word

# Ensure words are lowercase
texts = [list(map(lambda word: word.lower(), tokenizer.tokenize(document))) for document in documents]

# Ensure words are in the same tense
texts = [list(map(lambda word: WordNetLemmatizer().lemmatize(word, "v"), tokenizer.tokenize(document))) for document in documents]

# Ensure words are stemmed
texts = [list(map(lambda word: stemmer.stem(word), tokenizer.tokenize(document))) for document in documents]

# Ensure words are not stopwords
texts = [list(filter(notStopWord, document)) for document in texts]

# Ensure words are at least 3 characters long
texts = [list(filter(lambda word: len(word) >= 3, document)) for document in texts]

# Ensure word is a real word in english dictionary
texts = [list(filter(lambda word: word in en_words, document)) for document in texts]

# Remove empty lists from list
print(len(texts))
texts = [document for document in texts if document]
print(len(texts))

print(texts[1:2])


# remove words that appear only once
from collections import defaultdict

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# Distribution of words and frequencies
for k, v in sorted(frequency.items(), key=lambda x: x[1]):
    print(k, v)

texts = [[token for token in text if frequency[token] > 1] for text in texts]


from pprint import pprint  # pretty-printer

# pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')
print("PRINTING DICTIONARY")
print(dictionary)

# print(dictionary.token2id)

new_doc = "forget the pain"
new_vec = dictionary.doc2bow(new_doc.lower().split())
new_vec

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)
# print(corpus)


#### TF IDF
from gensim import corpora, models, similarities

tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
corpus_tfidf = tfidf[corpus]

# for doc in corpus_tfidf:
#     print(doc)

ntopics = 10

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=ntopics)  # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

print("Latent Semantic Indexing")
lsi.print_topics(ntopics)
# for l in lsi.print_topics(ntopics):
#     print(l)

# for doc in corpus_lsi:  # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
#     print(doc)

# LDA
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=ntopics)

print("Latent Dirichlet Allocation")
lda_model.print_topics(ntopics)


# for l in lda_model.print_topics(ntopics):
#     print(l)