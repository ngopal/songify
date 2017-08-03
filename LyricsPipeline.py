import sys
sys.path.append('/Users/nikhilgopal/Documents/Insight/vid2song/')
from Helpers.SQLhelper import SQL
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import spacy
from nltk.corpus import words
from collections import defaultdict
import logging
from nltk.stem import *
from nltk.stem.wordnet import WordNetLemmatizer
from collections import defaultdict
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
import lda
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import bokeh.plotting as bp
from bokeh.models import HoverTool, BoxSelectTool
from bokeh.plotting import figure, show, output_notebook

import logging
logging.getLogger("lda").setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LyricsPipeline:
    def __init__(self):
        # Initialize parameters
        self.num_clusters = 20
        self.num_topics = self.num_clusters

        # Vectorizers
        self.vectorizer = TfidfVectorizer(min_df=10, max_features=10000, ngram_range=(1, 2)) # set binary flag to True?
        self.cvectorizer = CountVectorizer(min_df=4, max_features=10000, ngram_range=(1,2))

        # Models
        self.lda_model = lda.LDA(n_topics=self.num_topics, n_iter=2000)
        self.tsne_model = TSNE(n_components=2, verbose=1, random_state=0, method='exact')
        self.kmeans_model = MiniBatchKMeans(n_clusters=self.num_clusters, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
        self.svd = TruncatedSVD(n_components=50, random_state=0)
        self.dbscan = DBSCAN(eps=0.10, min_samples=5)
        self.nmf_model = NMF(n_components=self.num_topics, random_state=1, alpha=.1, l1_ratio=.5)
        self.neigh_model = KNeighborsClassifier(n_neighbors=3)
        # self.lda = LatentDirichletAllocation(n_topics=self.num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)

        # Pipeline
        self.pdf = self.load_from_SQL()
        self.text_cleaning()

        # Stored Variables
        self.colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
                     "#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
                     "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
                     "#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])
        self.vz = self.tfidf()
        self.cvec = self.countvector()

        # Building Shared Models
        self.kmeans = self.kmeans_model.fit(self.vz)
        self.svd_tfidf = self.svd.fit_transform(self.vz)

        self.resultsOne = self.pipelineOne()

        self.resres = self.customPoint(["harry truman"], self.resultsOne)
        print(self.resres)

        self.classifyTagWords([0,1,2,3], self.resres)

        # self.plotbk(self.resres)

        # self.resultsOne = self.pipelineOne()
        # self.classifyTagWords([1, 2, 3], self.resultsOne)
        # print("ACTUAL", self.resultsOne.loc[1,])
        # self.resultsTwo = self.pipelineTwo()
        # self.resultsThree = self.pipelineThree()
        # self.resultsFour = self.pipelineFour()
        # print(self.resultsOne.head())
        # print(self.resultsTwo.head())
        # self.display_topics(self.nmf, self.vectorizer, 10)
        # self.display_topics(self.lda_model, self.vectorizer, 10)

    def load_from_SQL(self):
        # Load Database
        DBhelper = SQL()
        all_tables = ["19PgP2QSGPcm6Ve8VhbtpG", "37i9dQZF1DWTJ7xPn4vNaz","37i9dQZF1DX1ewVhAJ17m4","37i9dQZF1DX5bjCEbRU4SJ","37i9dQZF1DXbTxeAdrVG2l","37i9dQZF1DXcBWIGoYBM5M","3vxotOnOGDlZXyzJPLFnm2","49oW3sCI91kB2YGw7hsbBv","4tZSI7b1rnGVMdkGeIbCI4","68bXT1MZWZvLOJc0FZrgf7","6wz8ygUKjoHfbU7tB9djeS","7eHApqa9YVkuO6gELsju2j","spotifyplaylistid", "3nrwJoFbrMKSGeHAxaoYSC", "37i9dQZF1DX1XDyq5cTk95"]
        frames = []
        for p in all_tables:
            pdf = pd.read_sql_query("SELECT \"lyrics\", \"SpotifyArtistURI\" FROM \"" + p +"\"", DBhelper.engine)
            frames.append(pdf)
        pdf = pd.concat(frames)
        del(frames)
        pdf = pdf[pdf["lyrics"] != ""]
        return(pdf)

    def text_cleaning(self):
        documents =  [(str(l[1]), str(l[2])) for l in self.pdf[["lyrics", "SpotifyArtistURI"]].itertuples()]
        stoplist = set('for a of the and to in yeah who don like got want know baby let hey come tell need said way thing cause little look'.split())

        # Use NLTK tokenizer to correctly split contracted words
        tokenizer = RegexpTokenizer(r'\w+')
        # Use NLTK stopwords to make sure common words are not being stored in list
        stopset = set(spacy.en.STOP_WORDS)
        en_words = set(words.words())
        stemmer = SnowballStemmer("english")

        def notStopWord(word):
            if word not in stopset:
                if word not in stoplist:
                    return word

        # Ensure words are lowercase
        texts = [(list(map(lambda word: word.lower(), tokenizer.tokenize(document[0]))), document[1]) for document in documents]

        # Ensure words are in the same tense
        texts = [(list(map(lambda word: WordNetLemmatizer().lemmatize(word, "v"), tokenizer.tokenize(document[0]))), document[1]) for document in documents]

        # Ensure words are stemmed
        texts = [(list(map(lambda word: stemmer.stem(word), tokenizer.tokenize(document[0]))), document[1]) for document in documents]

        # Ensure words are not stopwords
        texts = [(list(filter(notStopWord, document[0])), document[1]) for document in texts]

        # Ensure words are at least 3 characters long
        texts = [(list(filter(lambda word: len(word) >= 3, document[0])), document[1]) for document in texts]

        # Ensure words with spaces are filtered out
        texts = [(list(filter(lambda word: ' ' not in word, document[0])), document[1]) for document in texts]

        # Ensure words are unique
        texts = [(list(set(document[0])), document[1]) for document in texts]

        # Ensure word is a real word in english dictionary
        texts = [(list(filter(lambda word: word in en_words, document[0])), document[1]) for document in texts]

        # Remove empty lists from list
        texts = [(document[0], document[1]) for document in texts if document[0]]

        # remove words that appear only once
        frequency = defaultdict(int)
        for text in texts:
            for token in text[0]:
                frequency[token] += 1

        # Distribution of words and frequencies
        # for k, v in sorted(frequency.items(), key=lambda x: x[1]):
        #     print(k, v)

        text_data = texts
        self.texts = [[token for token in text[0] if frequency[token] > 1] for text in text_data]
        self.artist_ids = [text[1] for text in text_data]
        return

    def tfidf(self):
        vz = self.vectorizer.fit_transform((' '.join(t) for t in self.texts))
        print(len(self.artist_ids[1]))
        print(vz.shape)
        return vz

    def countvector(self):
        cvz = self.cvectorizer.fit_transform((' '.join(t) for t in self.texts))
        return cvz

    def pipelineOne(self):
        """ Vectorized TFIDF -> K-Means -> (for visualization: TSNE) """
        # Perform SVD on vectorized TFIDF data. svd_tfidf is a term-document matrix
        # svd_tfidf = self.svd.fit_transform(self.vz)

        # Perform K-means on vectorized TFIDF data
        self.lyrics_tfidfvect_kmeans_clusters = self.kmeans.predict(self.vz)
        self.lyrics_tfidfvect_kmeans_distances = self.kmeans.transform(self.vz)

        # Perform TSNE on Kmeans results for visualization
        self.tsne_kmeans = self.tsne_model.fit_transform(self.lyrics_tfidfvect_kmeans_distances)
        kmeans_df = pd.DataFrame(self.tsne_kmeans, columns=['x', 'y'])
        kmeans_df['cluster'] = self.lyrics_tfidfvect_kmeans_clusters

        return kmeans_df


    def pipelineTwo(self):
        """ Vectorized TFIDF -> SVD -> DBSCAN -> (for visualization: TSNE)"""

        # Perform SVD on vectorized TFIDF data
        self.svd_tfidf = self.svd.fit_transform(self.vz)
        # Perform TSNE on SVD data
        self.tsne_svd_tfidf = self.tsne_model.fit_transform(self.svd_tfidf)
        # Convert TSNE data to data frame (for visualization)
        tfidf_df = pd.DataFrame(self.tsne_svd_tfidf, columns=['x', 'y'])

        # Perform DBSCAN using SVD data
        X = StandardScaler().fit_transform(self.svd_tfidf)

        db = self.dbscan.fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        # print('Estimated number of clusters: %d' % n_clusters_)

        tfidf_df['cluster'] = labels
        return tfidf_df

    def pipelineThree(self):
        """ Vectorized TFIDF ->  NNMF  """
        # Perform NNMF on vectorized TFIDF data
        nmf = self.nmf_model.fit_transform(self.vz)
        print("Components ")
        print(self.nmf_model.components_)

        # Need to attach this grouping to SVD -> TSNE xy coords to plot

        # Print Results to screen
        self.display_topics(self.nmf_model, self.vectorizer.get_feature_names(), 10)

    def pipelineFour(self):
        """ Vectorized TFIDF -> LDA """
        X_topics = self.lda_model.fit_transform(self.cvec)

        # Print results to screen
        self.display_topics(self.lda_model, self.cvectorizer.get_feature_names(), 10)

    def display_topics(self, model, feature_names, no_top_words):
        for topic_idx, topic in enumerate(model.components_):
            print("Topic %d:" % (topic_idx))
            print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

    def classifyTagWords(self, tag_words, pd_points):
        """ Perform KNN on a dataframe containing XY coordinates and group assignment """
        knn = self.neigh_model.fit(pd_points[["x", "y"]], pd_points[["cluster"]])
        logging.log(logging.INFO, knn.predict(pd_points.loc[1,["x", "y"]]))
        logging.log(logging.INFO, knn.predict_proba(pd_points.loc[1,["x", "y"]]))
        logging.log(logging.INFO, knn.predict(pd_points.loc[pd_points.index[-1],["x", "y"]]))
        logging.log(logging.INFO, knn.predict_proba(pd_points.loc[pd_points.index[-1],["x", "y"]]))


    def customPoint(self, custom_line, pipeline):
        """ Custom Data vs Pipeline One """

        # Turn new tag words into vector
        new_word_vector = self.vectorizer.transform(custom_line)
        logging.log(logging.INFO, new_word_vector.toarray())

        # Perform TSNE on Kmeans results for visualization
        kmeans_new_point_cluster_prediction = self.kmeans.predict(new_word_vector)
        kmeans_distance_new_point = self.kmeans.transform(new_word_vector)
        logging.log(logging.INFO, kmeans_new_point_cluster_prediction)
        logging.log(logging.INFO, kmeans_distance_new_point)


        # Perform TSNE on Kmeans results for visualization
        tsne_kmeans_new_point = self.tsne_model.fit_transform(kmeans_distance_new_point)
        logging.log(logging.INFO, tsne_kmeans_new_point)
        prepipi = [{"x" : tsne_kmeans_new_point[0][0],
                    "y" : tsne_kmeans_new_point[0][1],
                    "cluster" : kmeans_new_point_cluster_prediction[0],
                    "novel" : 1}]
        pipi = pd.DataFrame().from_dict(prepipi)
        logging.log(logging.INFO, pipi)
        sLength = len(pipeline['x'])
        pipeline["novel"] = np.zeros(sLength)

        return pipeline.append(pipi, ignore_index=True)

    def plotDF(self, expected_df):
        """ Expecting df with x, y, cluster """

        markers = np.array(["star", "."])

        plt.scatter(expected_df["x"], expected_df["y"], c = self.colormap[ expected_df["cluster"] ])

        plt.title('Pipeline One')
        plt.show()

    def plotbk(self, expected_df):
        output_notebook()
        plot_tfidf = bp.figure(plot_width=700, plot_height=600, title="tf-idf clustering of the lyrics",
                               tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                               x_axis_type=None, y_axis_type=None, min_border=1)

        plot_tfidf.scatter(x='x', y='y', source=expected_df)
        hover = plot_tfidf.select(dict(type=HoverTool))
        hover.tooltips={"cluster": "@cluster", "input":"@novel"}
        show(plot_tfidf)






if __name__ == "__main__":
    LP = LyricsPipeline()