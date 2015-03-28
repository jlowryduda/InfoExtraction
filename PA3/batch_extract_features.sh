#!/bin/sh

#extract features from the train set
python mallet_extract_features.py train.original train.features train

# extract features from the dev set
python mallet_extract_features.py dev.original dev.features test

# extract features from the test set
#python mallet_extract_features.py test.original test.features test
