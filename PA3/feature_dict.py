from collections import defaultdict
import nltk

class FeatureDict:
    def __init__(self, vocabulary_path, feature_set):
        self.vocabulary_path = vocabulary_path
        self.feature_set = feature_set
        self.feature_dict = defaultdict(lambda : defaultdict(str))
        self.read_vocabulary()
        self.create_feature_dict()

    def read_vocabulary(self):
        with open(self.vocabulary_path, "r") as vocabulary_file:
            self.vocab = vocabulary_file.read().splitlines()

    def create_feature_dict(self):
        counter = 0
        for token in self.vocab:
            for feature in self.feature_set:
                self.feature_dict[token][feature] = str(counter)
                counter += 1
