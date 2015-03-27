import nltk

class FeatureDict:
    """
    Map features to indices. Initialize with a list of feature names.
    Heavily inspired by Plotnick's CodeBook class.
    """

    def __init__(self, vocabulary_path, feature_set):
        self.vocabulary_path = vocabulary_path
        self.feature_set = feature_set
        self.feature_dict =\
            dict((feature, index) for index, feature in enumerate(feature_set))
        self.read_vocabulary()
        for item in self.vocab:
            self.add(item)


    def __contains__(self, item):
        return item in self.feature_dict

    def __iter__(self):
        return iter(self.feature_dict)

    def __len__(self):
        return len(self.feature_dict)

    def read_vocabulary(self):
        with open(self.vocabulary_path, "r") as vocabulary_file:
            self.vocab = vocabulary_file.read().splitlines()

    def __getitem__(self, feature):
        """Return the index for given feature."""
        return self.feature_dict[feature]

    def add(self, item):
        """Add an item with a generated index."""
        if item not in self:
            index = len(self)
            self.feature_dict[item] = index
        return item       
    
    def get(self, item, default=None):
        """Return index associated with item."""
        return self.feature_dict.get(item, default)
 
