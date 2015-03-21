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
    the mentions in that line ('no_rel' or the relation type).
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


def minimum_complete_tree(line, constituents):
    """
    Given a line with the indices of two mentions and a list of constituent
    parses, produce the minimum complete tree spanning those two mentions.
    """
    if int(line[2]) == int(line[8]):
        tree = constituents[int(line[2])]
        subtree_indices = tree.treeposition_spanning_leaves(int(line[3]),
                                                            int(line[10]))
        subtree = tree[subtree_indices]
        return tree_to_string(subtree)
    else:
        return '()'
    

def tree_to_string(subtree):
    """
    Given a subtree, convert it to a string with all intervening whitespace
    stripped out to be used as a feature.

    Eventually, we'll use the "attribute" parameter to specify which attribute
    in the tree we want to highlight instead of the leaves, but that's for
    another day.
    """
    subtree_string = subtree.pprint()
    subtree_lines = subtree_string.splitlines()
    tree = [t.strip() for t in subtree_lines]
    tree = ' '.join(tree)
    # The SVM-light-TK documentation suggests that no spaces are allowed
    # between sets of parentheses, so:
    string = tree.replace(') (', ')(')
    return string

    
    if int(line[2]) == int(line[8]):
        tree = constituents[int(line[2])]
        subtree = tree.treeposition_spanning_leaves(int(line[3]), int(line[10]))
        subtree_string = tree[subtree].pprint()
        subtree_lines = subtree_string.splitlines()
        tree = [t.strip() for t in subtree_lines]
        tree = ' '.join(tree)
        # The SVM-light-TK documentation suggests that no spaces are allowed
        # between sets of parentheses, so:
        tree = tree.replace(') (', ')(')
        return tree
    else:
        return '()'
    

def extract_features(lines):
    """
    Given lines of data, extracts features.
    """
    features = []
    curr_file = None
    j_path = os.getcwd() + '/data/jsons/'
    p_path = os.getcwd() + '/data/parsed/'
    for line in lines:
        if line[1] != curr_file:
            curr_file = line[1]
            print(curr_file)
            with open(j_path + curr_file + '.json', 'r') as infile:
                sents = json.load(infile)
            with open(p_path + curr_file + '.parsed', 'r') as infile:
                # Constituency parses and dependency parses are separated by
                # two new line characters.
                parses = infile.read()
                parses = parses.split('\n\n')
                parses = [sent for sent in parses if len(sent) > 0]
                constituents = [s for i, s in enumerate(parses) if i % 2 == 0]
                dependencies = [s.split('\n') for i, s in enumerate(parses)
                                if i % 2 == 1]
                constituents = [Tree.fromstring(c) for c in constituents]
        f_list = [get_label(line),
                  '\t|BT|',
                  minimum_complete_tree(line, constituents),
                  '|ET|']
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
                feature[0] = '-1'
            else:
                feature[0] = '1'
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
        file_type = sys.argv[3]
        if file_type == 'train':
            train = True
        else:
            train = False
        lines = read_from_file(file_name)
        features = extract_features(lines)
        write_to_file(file_name, features, label_name, train)

"""
if __name__ == "__main__":
    file_name = 'dev' #sys.argv[1]
    label_name = 'PHYS.Located' #sys.argv[2]
    file_type = 'train'
    if file_type == 'train':
        train = True
    else:
        train = False
    lines = read_from_file(file_name)
    features = extract_features(lines)
    write_to_file(file_name, features, label_name, train)
"""
