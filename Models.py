import operator
import pandas as pd
import spacy
from LyricsPipeline import LyricsPipeline
import logging
import coloredlogs
from Helpers.SQLhelper import SQL


class DocModel:
    """A simple document model using word vectors and cosine similarity"""
    def __init__(self, pl_name):
        self.pl_name = pl_name
        self.helper = SQL()
        self.pdf = pd.read_sql_table(pl_name, self.helper.engine)
        self.LP = LyricsPipeline(self.pl_name)

    def kmeans_svd(self, image_keywords):
        """ Runs the kmeans -> SVD pipeline """
        # Run Pipeline that does K-means and SVD  (Working)
        # resultsOne contains: x, y, cluster
        resultsOne = self.LP.pipeline_Kmeans_SVD()
        logging.log(logging.INFO, resultsOne)
        resres = self.LP.customPoint(image_keywords, resultsOne)
        # I have commonSongNames printed. Now I need to modify LP to provide track data as well
        results = list(self.LP.extractTopSongs(resres))
        results_songs =  [k[1] for k in results]
        results_tracks = [k[2] for k in results]
        return {"uris" : results_tracks, "songs" : results_songs}

    def svd_kmeans(self, image_keywords):
        """ Runs the SVD -> Kmeans pipeline """
        # Run Pipeline that does K-means and SVD  (Working)
        # resultsOne contains: x, y, cluster
        resultsTwo = self.LP.pipeline_SVD_Kmeans()
        logging.log(logging.INFO, resultsTwo)
        resres = self.LP.customPoint(image_keywords, resultsTwo)
        # I have commonSongNames printed. Now I need to modify LP to provide track data as well
        results = list(self.LP.extractTopSongs(resres))
        results_songs =  [k[1] for k in results]
        results_tracks = [k[2] for k in results]
        return {"uris" : results_tracks, "songs" : results_songs}

    def lda(self, image_keywords):
        """ Runs LDA """
        resultsThree = self.LP.pipeline_LDA(image_keywords)
        logging.log(logging.INFO, resultsThree)
        results_songs =  [self.LP.artist_ids[k] for k in resultsThree]
        results_tracks = [self.LP.tracks[k] for k in resultsThree]
        return {"uris" : results_tracks, "songs" : results_songs}


    def nullModel(self, image_keywords):
        # create models
        doc_models = []
        nlp = spacy.load('en')
        for row in self.pdf.itertuples():
            # print(row)
            if row[6]:
                if row[5]:
                    if row[4]:
                        """The code expects a certain schema"""
                        # doc_models.append({"song": row[5], "model" : nlp(row[6])}) # SongName
                        # doc_models.append({"song": row[4], "model" : nlp(row[6])}) #TrackURI
                        doc_models.append({"songURI" : row[4],"song": row[5], "model" : nlp(row[6])})

        real_test = image_keywords
        real_test = [i.split(" ")[-1] for i in real_test]
        in_str = ' '.join(real_test)
        print(in_str)
        inModel = nlp(in_str)
        results = {}
        results_names = {}

        # Populate results and results_name dictionaries with track URI or song name as keys, and cosine score as values
        for i,d in enumerate(doc_models):
            simScore = inModel.similarity(d["model"])
            results_names[d["song"]] = simScore
            results[d["songURI"]] = simScore # Storing TrackURIs as keys (instead of song names as above)

        # for k, v in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
        #     print(k, v)

        # Returning a dictionary with keys "uris" and "songs", each of which contains a sorted list
        return {"uris" : sorted(results.items(), key=operator.itemgetter(1), reverse=True), "songs" : sorted(results_names.items(), key=operator.itemgetter(1), reverse=True)}

if __name__ == "__main__":
    Model = DocModel("6wz8ygUKjoHfbU7tB9djeS") # Can make Models per playlist
    keyws = ['n04162706 seat belt, seatbelt', 'n02965783 car mirror', 'n03452741 grand piano, grand', 'n04070727 refrigerator, icebox', 'n04162706 seat belt, seatbelt', 'n04200800 shoe shop, shoe-shop, shoe store', 'n03670208 limousine, limo', 'n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier', 'n04356056 sunglasses, dark glasses, shades', 'n04356056 sunglasses, dark glasses, shades', 'n03534580 hoopskirt, crinoline', 'n03032252 cinema, movie theater, movie theatre, movie house, picture palace']
    # spotifyplaylistid, 4SMubSJhL8oHG1RNa6RGkQ
    # Model.nullModel(keyws)
    # print(Model.nullModel(keyws))
    print(Model.kmeans_svd(keyws))


