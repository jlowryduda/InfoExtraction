import sys
import os
import json
import re
from nltk.tree import Tree

### HELPER METHODS ###

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
    the mentions in that line ('no_rel' or the relation type).
    """
    return line[0]


def read_from_file(filename):
    """
    Read in data from file and do preprocessing on it for later use.
    """
    with open(os.getcwd() + '/data/' + filename, 'r') as infile:
        lines = infile.readlines()

    lines = [line.split() for line in lines]
    return lines


def geo_identity(line, geo_dict):
    """
    If the mentions are both a GPE, check the type of GPE by consulting GeoLite2
    database. This maps ex. "Warsaw" to "city" and "North Dakota" to "state".
    """
    if line[4] == "GPE" and line[9] == "GPE":
        cond_1 = False
        cond_2 = False
        try:
            cond_1 = clean_string(line[10]) in geo_dict[clean_string(line[5])]
        except:
            pass
        try:
            cond_2 = clean_string(line[5]) in geo_dict[clean_string(line[10])]
        except:
            pass
        if cond_1 or cond_2:
            return "geo_identity=True"
    pass



####
# Flat features
####


def entity_type_pair(line):
    """
    Returns a string containing entity types of both mentions, a la
    Jiang et al.
    """
    feature = line[5] + "-" + line[11]
    return "entity_type_pair=" + feature

def interceding_in(line, attributes): #GET THE ATTRIBUTES
    """
    Returns true if the two mentions appear in the same sentence, the second
    mention is of type GPE, and the word "in" appears between the two mentions.
    """
    if int(line[2]) == int(line[8]) and line[11] == 'GPE':
        sent = attributes[int(line[2])]
        interceding_span = sent[int(line[4]):int(line[9])]
        tokens = [clean_string(item['token']) for item in interceding_span]
        if 'in' in tokens:
            return "interceding_in=True"
    pass

def get_wm1(line):
    words = [clean_string(t) for t in line[7].split("_")]
    output = []
    for word in words:
        output.append("wm1_" + word + "=True")
    return " ".join(output)

def get_wm2(line):
    words = [clean_string(t) for t in line[13].split("_")]
    output = []
    for word in words:
        output.append("wm2_" + word + "=True")
    return " ".join(output)

def wb_null(line):
    """
    If there is no word between mentions.
    """
    if (line[2] == line[8]) and (line[4] == line[9]):
        return "wb_null=True"
    pass

def word_between(line, attributes):
    """
    If there is one word between the mentions, return that word.
    """
    sentence_no = int(line[2])
    index_1_end = int(line[4])
    index_2_start = int(line[9])
    if (line[2] == line[8]) and (index_2_start-index_1_end == 1):
        word_between = attributes[sentence_no][index_1_end]['token']
        return "word_between=" + word_between
    pass


def extract_features(lines):
    """
    Given lines of data, extracts features.
    """
    features = []
    a_path = os.getcwd() + '/data/attributes/'
    p_path = os.getcwd() + '/data/parsed/'
    curr_file = None
    for line in lines:
    	if line[1] != curr_file:
            curr_file = line[1]
            print(curr_file)
            with open(a_path + curr_file + '.json', 'r') as infile:
                attributes = json.load(infile)
            with open(p_path + curr_file + '.parsed', 'r') as infile:
                parses = infile.read()
                parses = parses.split('\n\n')
                parses = [sent for sent in parses if len(sent) > 0]
                constituents = [s for i, s in enumerate(parses) if i % 2 == 0]
                dependencies = [s.split('\n') for i, s in enumerate(parses)
                                if i % 2 == 1]
                constituents = [Tree.fromstring(c) for c in constituents]
        f_list = [get_label(line),
                  entity_type_pair(line),
                  interceding_in(line, attributes),
                  get_wm1(line),
                  get_wm2(line),
                  wb_null(line),
                  word_between(line, attributes)
                  ]
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
        lines = extract_features(lines)
        write_to_file(output_file, lines, train)
