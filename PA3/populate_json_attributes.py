import json
import os
from nltk.corpus import wordnet as wn

path = os.getcwd() + '/data/jsons/'

data_files = ['train.original.json', 'dev.original.json', 'test.original.json']
with open(path + 'train.original.json', 'r') as infile:
    train_file = json.load(infile)
with open(path + 'dev.original.json', 'r') as infile:
    dev_file = json.load(infile)
with open(path + 'test.original.json', 'r') as infile:
    test_file = json.load(infile)

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename not in data_files:
            print filename
            filename_without_extension = filename[:-5]
            if filename_without_extension in train_file:
                entities = train_file[filename_without_extension]
            elif filename_without_extension in dev_file:
                entities = dev_file[filename_without_extension]
            elif filename_without_extension in test_file:
                entities = test_file[filename_without_extension]
            with open(path + filename, 'r') as infile:
                f = json.load(infile)
            new_file = []
            for i, sentence in enumerate(f):
                new_sentence = []
                for j, item in enumerate(sentence):
                    item_dict = dict()
                    item_dict['token'] = item[0]
                    item_dict['pos'] = item[1]

                    # Populate hypernym information
                    token = item[0].lower()
                    synsets = wn.synsets(token)
                    hypernym = '*'
                    if synsets:
                        hyper_synsets = synsets[0].hypernyms()
                        if hyper_synsets:
                            hypernym = hyper_synsets[0].lemma_names()[0]
                    item_dict['hypernym'] = hypernym

                    # Populate entity type information:
                    try:
                        entity_type = entities[str(i)][str(j)]
                    except KeyError:
                        entity_type = '*'
                    item_dict['entity_type'] = entity_type
                    new_sentence.append(item_dict)
                new_file.append(new_sentence)
            with open(path[:-6] + '/attributes/' + filename, 'w') as outfile:
                json.dump(new_file, outfile)
