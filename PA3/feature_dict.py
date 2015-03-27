import nltk

class FeatureDict:
    """
    Map features to indices. Initialize with a list of feature names.
    Heavily inspired by Plotnick's CodeBook class.
    """
    def __init__(self):
        self.feature_dict = {}

    def __contains__(self, item):
        return item in self.feature_dict

    def __iter__(self):
        return iter(self.feature_dict)

    def __len__(self):
        return len(self.feature_dict)

    def __getitem__(self, feature):
        """Return the index for given feature."""
        return self.feature_dict[feature]

    def add(self, item):
        """Add an item with a generated index."""
        if item not in self:
            index = len(self)
            self.feature_dict[item] = index
        return len(self.feature_dict)       
