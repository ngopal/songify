import gensim



# TODO: Since this takes a while to load, I should pre-load this in the (probably in the flask server)


class W2VModel:
    def __init__(self):
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./NNmodels/GoogleNews-vectors-negative300.bin', binary=True)

    def processSentence(self, sentence):
        return list(filter(lambda x: x in self.model.vocab, sentence))

    def similarity(self, sent1, sent2):
        return self.model.n_similarity(self.processSentence(sent1.split(' ')), self.processSentence(sent2.split(' ')))

    def test(self):
        sent1 = "hello this is nikhil"
        sent2 = "hello this is not nikhil"
        print("Sentence 1:", self.processSentence(sent1.split(' ')))
        print("Sentence 2:", self.processSentence(sent2.split(' ')))
        sim = self.model.n_similarity(self.processSentence(sent1.split(' ')), self.processSentence(sent2.split(' ')))
        print("Similarity:", sim)
        return sim


if __name__ == "__main__":
    W2V = W2VModel()
    W2V.test()