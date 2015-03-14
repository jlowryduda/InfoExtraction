import sys
import os
import json
import re
from nltk.tree import Tree

def read_from_file(filename):
    """
    Read in data from file and do preprocessing on it for later use.
    """
    with open(os.getcwd() + '/data/' + filename, 'r') as infile:
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
    For a line in the training file, return the value for coreference between
    the mentions in that line ('yes' or 'no').
    """
    return line[0]


def extract_features(lines):
    """
    Given lines of data, extracts features.
    """
    features = []
    curr_file = None
    for line in lines:
    	if line[0] != curr_file:
            curr_file = line[1]
            print(curr_file)
            with open(j_path + curr_file + '.raw.json', 'r') as infile:
                sents = json.load(infile)
        f_list = [get_label(line)]
        features.append([f for f in f_list if f is not None])
    return features


def write_to_file(filename, features, train=False):
    """
    Write resulting features to a feature file.  If this isn't training data,
    produce two files, one with labels and one without.
    """
    lines = [' '.join(f) for f in features]
    with open(filename + '.labeled', 'w') as outfile:
        for line in lines:
            outfile.write(line)
            outfile.write('\n')
    if not train:
        with open(filename + '.nolabel', 'w') as outfile:
            for line in lines:
                outfile.write(line[1:])
                outfile.write('\n')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Specify an input filename, an output filename, and")
        print("either 'train' or 'test' depending on datatype")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        if sys.argv[3] == 'train':
            train = True
        else:
            train = False
        lines = read_from_file(input_file)
        features = extract_features(lines)
        write_to_file(output_file, features, train)
