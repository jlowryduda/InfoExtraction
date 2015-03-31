import sys
import os
import json
import re
import itertools
import operator
import pickle
from nltk.tree import Tree, ParentedTree
from feature_dict import FeatureDict


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


### TREE METHODS ###


def minimum_complete_tree(line, constituents):
    """
    Given a line with the indices of two mentions and a list of constituent
    parses, produce the minimum complete tree spanning those two mentions.
    """
    if int(line[2]) == int(line[8]):
        tree = constituents[int(line[2])]
        tree = ParentedTree.fromstring(tree)
        tree = populate_entity_type(line, tree)
        subtree_indices = tree.treeposition_spanning_leaves(int(line[3]),
                                                            int(line[10]))
        subtree = tree[subtree_indices]
        return subtree
    else:
        return ParentedTree('S', [''])


def path_enclosed_tree(line, constituents, attributes, attrib):
    """
    Given a line with the indices of two mentions and a list of constituent
    parses, produce the path enclosed tree spanning those two mentions.
    Additionally take a data structure of attributes by token and a string
    representing the specific attribute we're interested in, and produce the
    tree with that attribute as the leaves.
    """
    if int(line[2]) == int(line[8]):
        # Get full tree:
        tree = constituents[int(line[2])]
        tree = ParentedTree.fromstring(tree)
        if attrib != 'token':
            tree = populate_by_attribute(line, tree, attributes, attrib)
        # Get pointers to the specific mentions in the tree:
        pointer_mention_1 = tree.leaf_treeposition(int(line[3]))
        pointer_mention_2 = tree.leaf_treeposition(int(line[10]) - 1)

        # Remove all right siblings from mention 2, moving up to root
        # (We do this first so as not to screw up indices for left siblings)
        curr = tree[pointer_mention_2[:-1]]
        while curr.parent():
            while curr.right_sibling():
                del tree[curr.right_sibling().treeposition()]
            curr = curr.parent()

        # Remove all left siblings from mention 1, moving up to root:
        curr = tree[pointer_mention_1[:-1]]
        while curr.parent():
            while curr.left_sibling():
                del tree[curr.left_sibling().treeposition()]
            curr = curr.parent()

        # Return trimmed tree:
        return tree
    else:
        return ParentTree('S', [''])


def populate_by_attribute(line, tree, attributes, attrib):
    """
    Takes a line from an input data file such as train, dev, or test;
    a tree representing the sentence indicated in that line;
    a data structure which can be indexed into by sentence and token to retrieve
    various attributes of that token such as hypernym and entity type;
    and a string representing which attribute we want to get.
    Goes through the tree for each index between the two mentions, replacing
    the leaf with the value of whatever attributes was supplied as a parameter.
    """
    index_span = range(int(line[3]), int(line[10]))
    attribs = [attributes[int(line[2])][index][attrib] for index in index_span]
    leaf_indices = [tree.leaf_treeposition(index) for index in index_span]
    for i, index in enumerate(leaf_indices):
        tree[index] = attribs[i]
    return tree


def tree_to_string(subtree):
    """
    Given a subtree, convert it to a string with all intervening whitespace
    stripped out to be used as a feature.
    """
    subtree_string = subtree.pprint()
    subtree_lines = subtree_string.splitlines()
    tree = [t.strip() for t in subtree_lines]
    tree = ' '.join(tree)
    # The SVM-light-TK documentation suggests that no spaces are allowed
    # between sets of parentheses, so:
    string = tree.replace(') (', ')(')
    return string


def get_bow_tree(line, constituents, attributes):
    """
    Return a bag-of-words tree representation, ex.
    (BOW (What *)(does *)(S.O.S. *)(stand *)(for *)(? *))
    """
    tree = path_enclosed_tree(line, constituents, attributes, 'token')
    leaves = set(tree.leaves())
    output = "(BOW "
    for leaf in leaves:
        output += "(" + leaf + " *)"
    output += ")"
    return output


### FLAT FEATURE METHODS ###


def entity_type_pair(line, flat_features_dict):
    """
    Returns a string containing entity types of both mentions, a la
    Jiang et al.
    """
    feature = line[5] + "-" + line[11]
    if (feature) in flat_features_dict:
        feature_id = flat_features_dict[(feature)]
    else:
        feature_id = flat_features_dict.add((feature))
    return [(feature_id, "1")]


def interceding_in(line, attributes, flat_features_dict):
    """
    Returns true if the two mentions appear in the same sentence, the second
    mention is of type GPE, and the word "in" appears between the two mentions.
    """
    if int(line[2]) == int(line[8]) and line[11] == 'GPE':
        sent = attributes[int(line[2])]
        interceding_span = sent[int(line[4]):int(line[9])]
        tokens = [clean_string(item['token']) for item in interceding_span]
        if ("interceding_in") in flat_features_dict:
            feature_id = flat_features_dict[("interceding_in")]
        else:
            feature_id = flat_features_dict.add(("interceding_in"))
        if 'in' in tokens:
            return [(feature_id, "1")]
    return []


def get_wm1(line, flat_features_dict):
    words = [clean_string(t) for t in line[7].split("_")]
    output = []
    for word in words:
        if (word, "wm1") in flat_features_dict:
            feature_id = flat_features_dict[(word, "wm1")]
        else:
            feature_id = flat_features_dict.add((word, "wm1"))
        output.append((feature_id, "1"))
    return output


def get_wm2(line, flat_features_dict):
    words = [clean_string(t) for t in line[13].split("_")]
    output = []
    for word in words:
        if (word, "wm2") in flat_features_dict:
            feature_id = flat_features_dict[(word, "wm2")]
        else:
            feature_id = flat_features_dict.add((word, "wm2"))
        output.append((feature_id, "1"))
    return output


def wb_null(line, flat_features_dict):
    """
    If there is no word between mentions.
    """
    if (line[2] == line[8]) and (line[4] == line[9]):
        if ("wb_null") in flat_features_dict:
            feature_id = flat_features_dict['wb_null']
        else:
            feature_id = flat_features_dict.add(("wb_null"))
        return [(feature_id, "1")]
    return []


def word_between(line, attributes, flat_features_dict):
    """
    If there is one word between the mentions, return that word.
    """
    sentence_no = int(line[2])
    index_1_end = int(line[4])
    index_2_start = int(line[9])
    if (line[2] == line[8]) and (index_2_start-index_1_end == 1):
        word_between =\
        line[7], attributes[sentence_no][index_1_end]['token'], line[13]
        if (word_between, "wb") in flat_features_dict:
            feature_id = flat_features_dict[(word_between, "wb")]
        else:
            feature_id = flat_features_dict.add((word_between, "wb"))
        return [(feature_id, "1")]
    return []

def gather_flat_features(flat_f):
    flat_f = list(itertools.chain(*flat_f))
    flat_f = list(set(flat_f))
    flat_f = sorted(flat_f, key=operator.itemgetter(0))
    flat_f = [str(feature_id) + ":" + value for feature_id, value in flat_f]
    flat_f = " ".join(flat_f)
    flat_f = "|BV| " + flat_f + " |EV|"
    return flat_f


### FILE READ/WRITE METHODS ###


def read_from_file(filename):
    """
    Read in data from file and do preprocessing on it for later use.
    """
    with open(os.getcwd() + '/data/' + filename + '.original', 'r') as infile:
        lines = infile.readlines()

    lines = [line.split() for line in lines]
    return lines

def get_labels(lines):
    label_dict = dict()
    for line in lines:
        if line[0] in label_dict:
            label_dict[line[0]] += 1
        else:
            label_dict[line[0]] = 1
    del label_dict['no_rel']
    return label_dict


def write_to_file(filename, features):
    """
    Write resulting features to a feature file.  If this isn't training data,
    produce two files, one with labels and one without.  The label parameter
    indicates whether we want to re-write the file as a binary classification
    of that specific label.
    """
    path = os.getcwd() + '/data/results_by_label/' + filename + '.'
    for label in get_labels(lines):
        with open(path + label + '.labeled', 'w') as outfile:
            for feature in features:
                if feature[0] != label:
                    l = '-1 '
                else:
                    l = '1 '
                outfile.write(l + ' '.join(feature[1:]))
                outfile.write('\n')
    if file_name != 'test':
        with open(path + '.nolabel', 'w') as outfile:
            for feature in features:
                outfile.write(' '.join(feature[1:]))
                outfile.write('\n')


### FEATURE EXTRACTION METHOD ###


def extract_features(lines, filename, feature_dict=None):
    """
    Given lines of data, extracts features.
    """
    features = []
    a_path = os.getcwd() + '/data/attributes/'
    p_path = os.getcwd() + '/data/parsed/'
    # When we're training, feature_dict will be None so we'll build one from
    # scratch. If test, we'll fetch the preexisting one from the parameter.
    if not feature_dict:
        flat_features_dict = FeatureDict()
    else:
        flat_features_dict = feature_dict
    curr_file = None
    for line in lines:
        if line[1] != curr_file:
            curr_file = line[1]
            print(curr_file)
            with open(a_path + curr_file + '.json', 'r') as infile:
                attributes = json.load(infile)
            with open(p_path + curr_file + '.parsed', 'r') as infile:
                # Constituency parses and dependency parses are separated by
                # two new line characters.
                parses = infile.read()
                parses = parses.split('\n\n')
                parses = [sent for sent in parses if len(sent) > 0]
                constituents = [s for i, s in enumerate(parses) if i % 2 == 0]
                dependencies = [s.split('\n') for i, s in enumerate(parses)
                                if i % 2 == 1]

        flat_f = [get_wm1(line, flat_features_dict),
                  get_wm2(line, flat_features_dict),
                  wb_null(line, flat_features_dict),
                  interceding_in(line, attributes, flat_features_dict),
                  word_between(line, attributes, flat_features_dict), 
                 entity_type_pair(line, flat_features_dict)]

        f_list = [get_label(line),
                  '|BT|',
                  #tree_to_string(minimum_complete_tree(line, constituents)),
                  get_bow_tree(line, constituents, attributes),
                  '|BT|',
                  tree_to_string(path_enclosed_tree(line, constituents,
                                                    attributes, 'token')),
                  '|BT|',
                  tree_to_string(path_enclosed_tree(line, constituents,
                                                    attributes, 'entity_type')),
                  '|BT|',
                  tree_to_string(path_enclosed_tree(line, constituents,
                                                    attributes, 'hypernym')),
                  '|ET|',
                  gather_flat_features(flat_f)]
        features.append([f for f in f_list if f is not None])
        
    # When feature_dict was None, we were training so now we save the
    # feature_dict we created for when we test.
    if not feature_dict:
        with open('feature_dict.obj', 'w') as outfile:
            pickle.dump(flat_features_dict, outfile)
    return features




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Specify an input filename (with no extension) a feature label"
              "to do binary classification on, and either 'train' or 'test'"
              "depending on datatype")
    else:
        file_name = sys.argv[1]
        if file_name == 'train':
            feature_dict = None
        else:
            with open('feature_dict.obj', 'r') as infile:
                feature_dict = pickle.load(infile)
        lines = read_from_file(file_name)
        features = extract_features(lines, file_name, feature_dict)
        write_to_file(file_name, features)
