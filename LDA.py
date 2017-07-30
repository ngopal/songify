from gensim.models.word2vec import Word2Vec
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_20newsgroups
import re
import matplotlib.pyplot as plt
import sys

sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
from Helpers.SQLhelper import SQL
import pandas as pd

# Load Database
DBhelper = SQL()

# all_tables = ["19PgP2QSGPcm6Ve8VhbtpG", "37i9dQZF1DX1ewVhAJ17m4", "37i9dQZF1DX5bjCEbRU4SJ","3vxotOnOGDlZXyzJPLFnm2","49oW3sCI91kB2YGw7hsbBv","4tZSI7b1rnGVMdkGeIbCI4","68bXT1MZWZvLOJc0FZrgf7","6wz8ygUKjoHfbU7tB9djeS","7eHApqa9YVkuO6gELsju2j"]
all_tables = ["19PgP2QSGPcm6Ve8VhbtpG", "37i9dQZF1DX1ewVhAJ17m4"]


def multipleTableNames(list_of_tables):
    qc = "SELECT \'lyrics\' FROM "
    for k in list_of_tables:
        qc += "\"" + k + "\", "
    # qc += k+", "
    qc = qc[:-2]
    qc += " WHERE \'lyrics\' IS NOT NULL"
    return qc


qc = multipleTableNames(all_tables)
print(qc)
pdf = pd.read_sql_query(qc, DBhelper.engine)
print(pdf.head(10))
# lyrics = list(pdf["lyrics"])
print("SHaboy", lyrics)

#### Form the right data structures
from gensim import corpora

documents = lyrics
stoplist = set('for a of the and to in'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]

# remove words that appear only once
from collections import defaultdict

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]

from pprint import pprint  # pretty-printer

# pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('/tmp/deerwester.dict')
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

ntopics = 2

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=ntopics)  # initialize an LSI transformation
corpus_lsi = lsi[corpus_tfidf]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

lsi.print_topics(ntopics)

for doc in corpus_lsi:  # both bow->tfidf and tfidf->lsi transformations are actually executed here, on the fly
    print(doc)

# LDA
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=ntopics)

lda_model.print_topics(ntopics)
