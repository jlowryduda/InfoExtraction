#!/bin/sh

#Specify paths to train and dev files below

input_train_gold=../Project_1/CRF++-0.58/train.gold
output_train_gold=../Project_1/CRF++-0.58/new_train.gold
input_dev_gold=../Project_1/CRF++-0.58/dev.gold
output_dev_gold=../Project_1/CRF++-0.58/new_dev.gold
crf_location=../Project_1/CRF++-0.58/

python extract_features.py $input_train_gold $output_train_gold
echo "Extracted features from a train file"
python extract_features.py $input_dev_gold $output_dev_gold 
echo "Extracted features from a test file"
$crf_location/crf_learn template.txt $output_train_gold model.data
echo "Learned the model"
$crf_location/crf_test -m model.data $output_dev_gold > output.txt
echo "Tested the model. Computing p, r, f"
python evaluate-head.py $output_dev_gold output.txt
