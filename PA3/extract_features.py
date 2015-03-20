import sys
import os
import json
import re
from nltk.tree import Tree

def read_from_file(filename):
    """
    Read in data from file and do preprocessing on it for later use.
    """
    with open(os.getcwd() + '/data/' + filename + '.original', 'r') as infile:
        lines = infile.readlines()

    lines = [line.split() for line in lines]
    return lines


def clean_string(s):
    """Helper function that does some basic preprocessing on a string"""
    s = s.replace("`", "")
    s = s.replace("(", "")
    s = s.replace(")", "")
    if s.startswith("-"):
        s = s[1:]
    s = s.lower()
    return s


def get_label(line):
    """
    For a line in the training file, return the relation type between
    the mentions in that line ('yes' or 'no').
    """
    return line[0]

def entity_type_pair(line):
    """
    Returns a string containing entity types of both mentions, a la
    Jiang et al. 
    """
    return line[5] + "-" + line[11]


def interceding_in(line, sents):
    """
    Returns true if the two mentions appear in the same sentence, the second
    mention is of type GPE, and the word "in" appears between the two mentions.
    """
    if int(line[2]) == int(line[8]) and line[11] == 'GPE':
        sent = sents[int(line[2])]
        interceding_span = sent[int(line[4]):int(line[9])]
        tokens = [clean_string(item[0]) for item in interceding_span]
        if 'in' in tokens:
            return "interceding_in=True"
    pass


def extract_features(lines):
    """
    Given lines of data, extracts features.
    """
    features = []
    curr_file = None
    j_path = os.getcwd() + '/data/jsons/'
    for line in lines:
    	if line[1] != curr_file:
            curr_file = line[1]
            print(curr_file)
            with open(j_path + curr_file + '.json', 'r') as infile:
                sents = json.load(infile)
        f_list = [get_label(line),
                 entity_type_pair(line),
                 interceding_in(line, sents)]
        features.append([f for f in f_list if f is not None])
    return features


def write_to_file(filename, features, label=None, train=False):
    """
    Write resulting features to a feature file.  If this isn't training data,
    produce two files, one with labels and one without.  The label parameter
    indicates whether we want to re-write the file as a binary classification
    of that specific label.
    """
    with open(os.getcwd() + '/data/' + filename + '.labeled', 'w') as outfile:
        for feature in features:
            if label and (feature[0] != label):
                feature[0] = 'no_rel'
            outfile.write(' '.join(feature))
            outfile.write('\n')
    if not train:
        with open(os.getcwd() + '/data/' + filename + '.nolabel', 'w') as outfile:
            for feature in features:
                outfile.write(' '.join(feature[1:]))
                outfile.write('\n')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Specify an input filename (with no extension) a feature label"
              "to do binary classification on, and either 'train' or 'test'"
              "depending on datatype")
    else:
        file_name = sys.argv[1]
        label_name = sys.argv[2]
        if sys.argv[3] == 'train':
            train = True
        else:
            train = False
        lines = read_from_file(file_name)
        features = extract_features(lines)
        write_to_file(file_name, features, label_name, train)
