import sys
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
import pandas as pd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from Helpers.SQLhelper import SQL

DBhelper = SQL()

table_name = "aa_6wz8ygUKjoHfbU7tB9djeS"
table_name2 = "6wz8ygUKjoHfbU7tB9djeS"

#pdf = pd.read_sql_query(" SELECT * FROM \""+table_name+"\" JOIN ", DBhelper.engine)

# pdf = pd.read_sql_query("SELECT * FROM \"aa_6wz8ygUKjoHfbU7tB9djeS\" JOIN \"6wz8ygUKjoHfbU7tB9djeS\" ON \"aa_6wz8ygUKjoHfbU7tB9djeS\".uri=\"6wz8ygUKjoHfbU7tB9djeS\".SpotifySongURI", DBhelper.engine)

# pdf = pd.read_sql_query("SELECT \"uri\" FROM \""+table_name+"\" ", DBhelper.engine)
# pdf = pd.read_sql_query("SELECT \"SpotifySongURI\" FROM \""+table_name+"\" ", DBhelper.engine)

## JOIN
## I could also do a pandas merge, if I don't want to do this in SQL

## Query to join two tables and return all data
# pdf = pd.read_sql_query("SELECT * FROM \""+table_name+"\", \""+table_name2+"\" WHERE \"SpotifySongURI\" = \"uri\"", DBhelper.engine)

## Query to join two tables and return all data that have lyrics
pdf = pd.read_sql_query("SELECT * FROM \""+table_name+"\", \""+table_name2+"\" WHERE \"SpotifySongURI\" = \"uri\" AND \"lyrics\" IS NOT NULL", DBhelper.engine)

# print(pdf)

ls = []
for tup in pdf[["commonArtistName","lyrics"]].itertuples():
    print(tup[2].split())
#     ls.append(  LabeledSentence(words=tup[2].split(), tags=[tup[1]])  )
# print(ls)
#
# # ls = LabeledSentence(words="nikhil is part of insight".split(" "), tags=["nikhil"])
# # print(ls)
#
# model = Doc2Vec(ls, size=100, window=8, min_count=5, workers=4)
# print(model)
# print(model.most_similar("tears and whiskey"))