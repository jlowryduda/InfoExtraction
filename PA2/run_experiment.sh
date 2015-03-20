#!/bin/sh

#Specify paths to train and dev files below

EXPERIMENT_DIR=/home/g/grad/libenson/ie
MALLET=/home/j/clp/chinese/bin/mallet-maxent-classifier.sh

# train the classifier
$MALLET -train \
	-model=$EXPERIMENT_DIR/model \
	-gold=$EXPERIMENT_DIR/train.features.labeled

# test model
$MALLET -classify  \
	-model=$EXPERIMENT_DIR/model \
	-input=$EXPERIMENT_DIR/dev.features.nolabel > $EXPERIMENT_DIR/dev.features.tagged

# evaluate
python coref-evaluator.py dev.features.labeled dev.features.tagged
