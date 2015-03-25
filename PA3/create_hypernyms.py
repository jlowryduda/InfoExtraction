import os
import json
from nltk.corpus import wordnet as wn

path = os.getcwd() + '/data/jsons/'

data_files = ['train.original.json', 'dev.original.json', 'test.original.json']

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename not in data_files:
            print filename
            with open(path + filename, 'r') as infile:
                f = json.load(infile)
            new_file = []
            for sentence in f:
                new_sentence = []
                for item in sentence:
                    token = item[0].lower()
                    synsets = wn.synsets(token)
                    hypernym = '*'
                    if synsets:
                        hyper_synsets = synsets[0].hypernyms()
                        if hyper_synsets:
                            hypernym = hyper_synsets[0].lemma_names()[0]
                    new_sentence.append([item[0], hypernym])
                new_file.append(new_sentence)
            with open(path[:-6] + 'hypernyms/' + filename, 'w') as outfile:
                json.dump(new_file, outfile)
