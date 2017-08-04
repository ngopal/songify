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
import math

import logging
import coloredlogs
coloredlogs.install()
logging.getLogger("lda").setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class LyricsPipeline:
    def __init__(self):
        # Initialize parameters
        LTP = LyricsTextProcessing()
        self.num_clusters = 20
        self.num_topics = self.num_clusters

        # Vectorizers
        self.vectorizer = TfidfVectorizer(min_df=10, max_features=10000, ngram_range=(1, 2)) # set binary flag to True?
        self.cvectorizer = CountVectorizer(min_df=4, max_features=10000, ngram_range=(1,2))

        # Models
        self.lda_model = lda.LDA(n_topics=self.num_topics, n_iter=2000)
        self.tsne_model = TSNE(n_components=2, verbose=1, random_state=0, method='exact')
        self.kmeans_model = MiniBatchKMeans(n_clusters=self.num_clusters, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
        self.svd = TruncatedSVD(n_components=2, random_state=0)
        self.dbscan = DBSCAN(eps=0.10, min_samples=5)
        self.nmf_model = NMF(n_components=self.num_topics, random_state=1, alpha=.1, l1_ratio=.5)
        self.neigh_model = KNeighborsClassifier(n_neighbors=3)
        # self.lda = LatentDirichletAllocation(n_topics=self.num_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0)

        # Pipeline
        self.pdf = LTP.pdf
        self.artist_ids = LTP.artist_ids
        self.texts = LTP.texts


        # print(list(zip(self.texts,self.artist_ids)))
        # print(len(self.artist_ids))
        # sys.exit()

        # Stored Variables
        self.colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
                     "#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
                     "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
                     "#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])
        self.tfidf_vector = self.tfidf()
        self.count_vector = self.countvector()

        # Building Shared Models
        self.kmeans = self.kmeans_model.fit(self.tfidf_vector)
        self.svd_tfidf = self.svd.fit_transform(self.tfidf_vector)

        ##### FINAL RUNS ####
        # Run Pipeline that does K-means and SVD  (Working)
        # resultsOne contains: x, y, cluster
        # self.resultsOne = self.pipeline_Kmeans_SVD()
        # logging.log(logging.INFO, self.resultsOne)
        # self.resres = self.customPoint(["I love you"], self.resultsOne)
        # for k in self.extractTopSongs(self.resres):
        #     print(k)

        # Run Pipeline that does SVD then Kmeans   (Working)
        # resultsTwo contains: x, y, cluster
        # self.resultsTwo = self.pipeline_SVD_Kmeans()
        # logging.log(logging.INFO, self.resultsTwo)
        # self.resres = self.customPoint(["I love you"], self.resultsTwo)
        # for k in self.extractTopSongs(self.resres):
        #     print(k)

        # Run Pipeline that does LDA
        # resultsThree contains: Topic_dim1, Topic_dim2,...
        self.resultsThree = self.pipeline_LDA()
        logging.log(logging.INFO, self.resultsThree)
        # TODO
        # I need to (1) run transform to figure out which topic (column) my new data point is part of,
        # then (2) get all of the rows for which that column is the max column, (3) then figure out which N
        # songs are the "closest" to the line in question. Then I won't need coords, but will have songs returned
        #
        # self.resres = self.customPoint(["I love you"], self.resultsTwo)
        # for k in self.extractTopSongs(self.resres):
        #     print(k)

        # Run Pipeline that does LDA and Kmeans
        # resultsFour contains: Topic_dim1, Topic_dim2,..., cluster
        # self.resultsFour = self.pipeline_LDA_Kmeans()
        # logging.log(logging.INFO, self.resultsFour)


        # Run random selection pipeline (Working)
        # self.resultsFive = self.randomSelection(10)
        # logging.log(logging.INFO, self.resultsFive)

        ####### FINAL RUNS ######







        # for k in self.extractTopSongs(self.resres):
        #     print(k)

        # self.plotDF(self.resres)

        # self.classifyTagWords([0,1,2,3], self.resres)

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
        svd_tfidf = self.svd.fit_transform(self.tfidf_vector)
        logging.log(logging.INFO, svd_tfidf)

        # Perform K-means on vectorized TFIDF data
        self.lyrics_tfidfvect_kmeans_clusters = self.kmeans.predict(self.tfidf_vector)
        self.lyrics_tfidfvect_kmeans_distances = self.kmeans.transform(self.tfidf_vector)

        # Perform TSNE on Kmeans results for visualization
        self.tsne_kmeans = self.tsne_model.fit_transform(self.lyrics_tfidfvect_kmeans_distances)
        kmeans_df = pd.DataFrame(self.tsne_kmeans, columns=['x', 'y'])
        kmeans_df['cluster'] = self.lyrics_tfidfvect_kmeans_clusters

        return kmeans_df

    def pipeline_Kmeans_SVD(self):
        """ Vectorized TFIDF -> K-Means -> SVD """
        # Perform SVD on vectorized TFIDF data. svd_tfidf is a term-document matrix
        # svd_tfidf = self.svd.fit_transform(self.vz)
        # logging.log(logging.INFO, svd_tfidf)

        # Perform K-means on vectorized TFIDF data
        self.lyrics_tfidfvect_kmeans_clusters = self.kmeans.predict(self.tfidf_vector)
        self.lyrics_tfidfvect_kmeans_distances = self.kmeans.transform(self.tfidf_vector)

        # Perform SVD on Kmeans results for visualization
        self.svd_kmeans = self.svd.fit_transform(self.lyrics_tfidfvect_kmeans_distances)
        logging.log(logging.INFO, self.svd_kmeans)
        kmeans_df = pd.DataFrame(self.svd_kmeans, columns=['x', 'y'])
        kmeans_df['cluster'] = self.lyrics_tfidfvect_kmeans_clusters

        return kmeans_df

    def pipeline_SVD_Kmeans(self):
        """ Vectorized TFIDF -> SVD -> Kmeans"""
        # Perform SVD on vectorized TFIDF data
        svd = TruncatedSVD(n_components=2, random_state=0)

        # Obtain SVD-ed TFIDF Vector
        svd_tfidf = svd.fit_transform(self.tfidf_vector)
        logging.log(logging.INFO, svd_tfidf)
        tfidf_df = pd.DataFrame(svd_tfidf, columns=['x', 'y'])
        logging.log(logging.INFO, tfidf_df)

        # Perform Kmeans using SVD data
        kmeans_svd_input_model = MiniBatchKMeans(n_clusters=self.num_clusters, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
        X = StandardScaler().fit_transform(svd_tfidf)
        kmeans_svd_input_model.fit(X)

        # Obtain K-means cluster assignments and distances
        kmeans_clusters = kmeans_svd_input_model.predict(X)
        kmeans_distances = kmeans_svd_input_model.transform(X)

        logging.log(logging.INFO, kmeans_clusters)
        logging.log(logging.INFO, kmeans_distances)

        # Add cluster column
        tfidf_df['cluster'] = kmeans_clusters

        logging.log(logging.INFO, tfidf_df)

        return tfidf_df

    def pipeline_LDA(self):
        """ Vectorized TFIDF -> LDA """
        from sklearn.decomposition import LatentDirichletAllocation

        # Build Model
        n_topics = 4
        lda_model = LatentDirichletAllocation(n_topics=n_topics)
        lda_df = lda_model.fit_transform(self.count_vector)
        logging.info("TRAINING:")
        logging.log(logging.INFO, lda_model.transform(self.count_vector))
        lda_df = pd.DataFrame(lda_df)
        logging.log(logging.INFO, lda_df)

        temp_columns = {}
        for i, x in enumerate(["x"+str(i) for i in range(n_topics)]):
            temp_columns[i] = x

        lda_df.rename(columns=temp_columns, inplace=True)
        logging.log(logging.INFO, lda_df)

        # I need to (1) run transform to figure out which topic (column) my new data point is part of,
        # then (2) get all of the rows for which that column is the max column, (3) then figure out which N
        # songs are the "closest" to the line in question. Then I won't need coords, but will have songs returned
        #

        # Example of how to call LDA model on new data
        new_word = self.cvectorizer.transform(["I love you"])
        new_vector = lda_model.transform(new_word)
        logging.info("NEW WORD:")
        logging.log(logging.INFO, new_vector)

        # Merge
        # Setup new data point for appending
        prepipi = {"x"+str(i) for i in range(n_topics)}
        prepipi = dict.fromkeys(prepipi, 0)
        for key, value in prepipi.items():
            print(key, value)
            print(key.split("x"))
            prepipi[key] = list(new_vector[0])[int(key.split("x")[1])]
        prepipi['novel'] = 1
        prepipi = [prepipi]
        logging.log(logging.INFO, prepipi)
        pipi = pd.DataFrame().from_dict(prepipi)
        logging.log(logging.INFO, pipi)

        # add 0 values to novel for current df
        sLength = len(lda_df['x0'])
        lda_df["novel"] = np.zeros(sLength)
        logging.log(logging.INFO, lda_df)

        lda_df = lda_df.append(pipi, ignore_index=True)

        logging.log(logging.INFO, lda_df)

        # Assign cluster column
        # lda_df contains Topic_dim1, Topic_Dim2, etc...
        logging.log(logging.INFO, lda_df)
        groupings = lda_df.apply(lambda l: 0 if np.asarray(l).argmax()-1 < 0 else np.asarray(l).argmax()-1, axis=1)
        logging.log(logging.INFO, groupings)
        lda_df['cluster'] = groupings
        logging.log(logging.INFO, lda_df)
        # lda_df now has # Topic Dim, novel, and cluster

        # Find the top N closest songs
        # Obtained the cluster group number of the novel point
        clusttemp = int(lda_df.loc[lda_df['novel'] == 1]['cluster'])
        logging.log(logging.INFO, clusttemp)

        # List of relevant df indices
        inds = list(lda_df.loc[lda_df['cluster'] == clusttemp].index)
        logging.log(logging.INFO, inds)

        # Find top 3 closest points in space
        # df = lda_df.loc[lda_df['cluster'] == clusttemp]
        df = lda_df.iloc[inds,:]
        logging.log(logging.INFO, df)

        return


    def pipeline_LDA_Kmeans(self):
        """ Vectorized TFIDF -> LDA -> Kmeans """
        from sklearn.decomposition import LatentDirichletAllocation

        # LDA Model
        n_topics = 4
        lda_model = LatentDirichletAllocation(n_topics=n_topics)
        lda_df = lda_model.fit_transform(self.count_vector)
        logging.info("TRAINING:")
        logging.log(logging.INFO, lda_model.transform(self.count_vector))

        lda_df = pd.DataFrame(lda_df, columns=[str(i) for i in range(1,(n_topics+1))])
        logging.log(logging.INFO, lda_df)

        # Kmeans
        kmeans_lda_input_model = MiniBatchKMeans(n_clusters=self.num_clusters, init='k-means++', n_init=1, init_size=1000, batch_size=1000, verbose=False, max_iter=1000)
        X = StandardScaler().fit_transform(lda_df)
        kmeans_lda_input_model.fit(X)

        # Obtain K-means cluster assignments and distances
        kmeans_clusters = kmeans_lda_input_model.predict(X)
        kmeans_distances = kmeans_lda_input_model.transform(X)

        logging.log(logging.INFO, kmeans_clusters)
        logging.log(logging.INFO, kmeans_distances)

        # Add cluster column
        lda_df['cluster'] = kmeans_clusters

        logging.log(logging.INFO, lda_df)

        return lda_df


        # Print results to screen
        # self.display_topics(lda_model, self.cvectorizer.get_feature_names(), 10)

    def randomSelection(self, num_songs):
        """ Takes in an argument for number of songs to return and returns a list of songs"""
        # Pull N random numbers from range rows of DF
        top_range = int(len(self.pdf.index))
        inds = [np.random.randint(0, top_range) for k in range(num_songs)]
        logging.log(logging.INFO, inds)

        # return df with cluster and novel columns
        logging.log(logging.INFO, self.pdf.iloc[inds,:])
        return self.pdf.iloc[inds,:]["commonSongName"]

    def pipelineTwo(self):
        """ Vectorized TFIDF -> SVD -> DBSCAN -> (for visualization: TSNE)"""

        # Perform SVD on vectorized TFIDF data
        self.svd_tfidf = self.svd.fit_transform(self.tfidf_vector)
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
        nmf = self.nmf_model.fit_transform(self.tfidf_vector)
        print("Components ")
        print(self.nmf_model.components_)

        # Need to attach this grouping to SVD -> TSNE xy coords to plot

        # Print Results to screen
        self.display_topics(self.nmf_model, self.vectorizer.get_feature_names(), 10)

    def pipelineFour(self):
        """ Vectorized TFIDF -> LDA """
        X_topics = self.lda_model.fit_transform(self.count_vector)

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
        # TODO: CHANGE THIS TO PCA AFTER KMEANS, BUT WILL NEED TO UPDATE PIPELINE AS WELL
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

    def extractClusterAssignment(self, expected_df):
        """ Returns a list of indexes """
        # Obtained the cluster group number of the novel point
        clusttemp = int(expected_df.loc[expected_df['novel'] == 1]['cluster'])

        # List of relevant df indices
        inds = list(expected_df.loc[expected_df['cluster'] == clusttemp].index)

        # Find top 3 closest points in space
        df = expected_df.loc[expected_df['cluster'] == clusttemp]
        #df.to_hdf('foo.h5', 'df') # Run this if I want to save the df
        df["distance"] = df.apply(lambda x: math.sqrt((df[df["novel"] == 1]["x"] - x["x"])**2 + (df[df["novel"] == 1]["y"] - x["y"])**2), axis=1)

        logging.log(logging.INFO, df.sort_values(["distance"]).head(n=11).tail(n=10).index)

        return df.sort_values(["distance"]).head(n=11).tail(n=10).index

    def extractTopSongs(self, expected_df):
        rel_ind = self.extractClusterAssignment(expected_df)
        return ((i,v) for i, v in enumerate(self.artist_ids) if i in rel_ind )

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



class LyricsTextProcessing():
    def __init__(self):
        # Do Text Processing
        self.pdf = self.load_from_SQL()
        k = self.text_cleaning()

        # Initialize null variables
        self.texts = k['texts']
        self.artist_ids = k['artist_ids']

    def load_from_SQL(self):
        # Load Database
        DBhelper = SQL()
        all_tables = ["19PgP2QSGPcm6Ve8VhbtpG", "37i9dQZF1DWTJ7xPn4vNaz","37i9dQZF1DX1ewVhAJ17m4","37i9dQZF1DX5bjCEbRU4SJ","37i9dQZF1DXbTxeAdrVG2l","37i9dQZF1DXcBWIGoYBM5M","3vxotOnOGDlZXyzJPLFnm2","49oW3sCI91kB2YGw7hsbBv","4tZSI7b1rnGVMdkGeIbCI4","68bXT1MZWZvLOJc0FZrgf7","6wz8ygUKjoHfbU7tB9djeS","7eHApqa9YVkuO6gELsju2j","spotifyplaylistid", "3nrwJoFbrMKSGeHAxaoYSC", "37i9dQZF1DX1XDyq5cTk95"]
        frames = []
        for p in all_tables:
            pdf = pd.read_sql_query("SELECT \"lyrics\", \"commonSongName\" FROM \"" + p +"\"", DBhelper.engine)
            frames.append(pdf)
        pdf = pd.concat(frames)
        del(frames)
        pdf = pdf[pdf["lyrics"] != ""]
        pdf.drop_duplicates(["commonSongName"], inplace=True)
        return pdf

    def text_cleaning(self):
        documents =  [(str(l[1]), str(l[2])) for l in self.pdf[["lyrics", "commonSongName"]].itertuples()]
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
        texts = [[token for token in text[0] if frequency[token] > 1] for text in text_data]
        artist_ids = [text[1] for text in text_data]
        return {"texts" : texts, "artist_ids" : artist_ids}


if __name__ == "__main__":
    LP = LyricsPipeline()
    # LTP = LyricsTextProcessing()
    # print(LTP.texts)
    # print(LTP.artist_ids)