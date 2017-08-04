import operator
import pandas as pd
import spacy

from Helpers.SQLhelper import SQL


class DocModel:
    """A simple document model using cosine similarity"""
    def __init__(self, pl_name):
        self.pl_name = pl_name
        self.helper = SQL()
        self.pdf = pd.read_sql_table(pl_name, self.helper.engine)

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

        for i,d in enumerate(doc_models):
            simScore = inModel.similarity(d["model"])
            results_names[d["song"]] = simScore
            results[d["songURI"]] = simScore # Storing TrackURIs as keys (instead of song names as above)

        # for k, v in sorted(results.items(), key=operator.itemgetter(1), reverse=True):
        #     print(k, v)

        return {"uris" : sorted(results.items(), key=operator.itemgetter(1), reverse=True), "songs" : sorted(results_names.items(), key=operator.itemgetter(1), reverse=True)}

if __name__ == "__main__":
    Model = DocModel("6wz8ygUKjoHfbU7tB9djeS") # Can make Models per playlist
    keyws = ['n04162706 seat belt, seatbelt', 'n02965783 car mirror', 'n03452741 grand piano, grand', 'n04070727 refrigerator, icebox', 'n04162706 seat belt, seatbelt', 'n04200800 shoe shop, shoe-shop, shoe store', 'n03670208 limousine, limo', 'n02687172 aircraft carrier, carrier, flattop, attack aircraft carrier', 'n04356056 sunglasses, dark glasses, shades', 'n04356056 sunglasses, dark glasses, shades', 'n03534580 hoopskirt, crinoline', 'n03032252 cinema, movie theater, movie theatre, movie house, picture palace']
    # spotifyplaylistid, 4SMubSJhL8oHG1RNa6RGkQ
    # Model.nullModel(keyws)
    print(Model.nullModel(keyws))


