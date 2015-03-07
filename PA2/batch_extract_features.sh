#!/bin/sh

#extract features from the train set
python extract_features.py train.original train.features.labeled train

# extract features from the dev set
python extract_features.py dev.original dev.features.nolabel test

# extract features from the test set
python extract_features.py test.original test.features.nolabel test

# puts label first on a dev set with features
python extract_features.py dev.original dev.features.labeled train

# puts label first on a test set with features
python extract_features.py test.original test.features.labeled train
