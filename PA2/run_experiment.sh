#!/bin/sh

#Specify paths to train and dev files below

DATA_DIR=/home/j/xuen/teaching/cosi137/spring-2015/projects/project2/data
EXPERIMENT_DIR=/home/g/grad/jduda/ie
MALLET=/home/j/clp/chinese/bin/mallet-maxent-classifier.sh

# extract features from the train set
python $EXPERIMENT_DIR/extract_features.py $DATA_DIR/coref-trainset.gold $EXPERIMENT_DIR/coref-trainset-features.txt train

# extract features from the test set
python $EXPERIMENT_DIR/extract_features.py $DATA_DIR/coref-devset.gold $EXPERIMENT_DIR/coref-devset-features.txt test

# train the classifier
$MALLET -train \
	-model=$EXPERIMENT_DIR/model \
	-gold=$EXPERIMENT_DIR/coref-trainset-features.txt

# test model
$MALLET -classify  \
	-model=$EXPERIMENT_DIR/model \
	-input=$EXPERIMENT_DIR/coref-devset-features.txt > $EXPERIMENT_DIR/coref-devset-tagged.txt

# evaluate
python $EXPERIMENT_DIR/coref-evaluator.py $DATA_DIR/coref-devset.gold \
	$EXPERIMENT_DIR/coref-devset-tagged.txt > $EXPERIMENT_DIR/evaluation.txt
