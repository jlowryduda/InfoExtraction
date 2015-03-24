import sys
import os
import json
import re
from nltk.tree import Tree
from AttributeTree import AttributeTree as ATree

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
        tree = ATree.fromstring(tree)
        tree = populate_entity_type(line, tree)
        subtree_indices = tree.treeposition_spanning_leaves(int(line[3]),
                                                            int(line[10]))
        subtree = tree[subtree_indices]
        return tree_to_string(subtree)
    else:
        return '()'


def path_enclosed_tree(line, constituents):
    """
    Given a line with the indices of two mentions and a list of constituent
    parses, produce the path enclosed tree spanning those two mentions.
    """
    if int(line[2]) == int(line[8]):
        # Get full tree:
        tree = constituents[int(line[2])]
        tree = ATree.fromstring(tree)
        tree = populate_entity_type(line, tree)
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
        return ATree('S', [''])


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


def populate_entity_type(line, tree):
    """
    Given a subtree containing the two mentions in the line, populate the nodes
    in the subtree representing those mentions with the entity_type attribute
    that matches the mention.
    """
    entity_1 = line[5]
    entity_2 = line[11]
    span_1 = range(int(line[3]), int(line[4]))
    span_2 = range(int(line[9]), int(line[10]))
    leaves_1 = [tree.leaf_treeposition(index) for index in span_1]
    leaves_2 = [tree.leaf_treeposition(index) for index in span_2]
    for index in leaves_1:
        # Leave off last index to move up from the leaf to its parent node
        tree[index[:-1]].entity_type = entity_1
    for index in leaves_2:
        tree[index[:-1]].entity_type = entity_2
    return tree


def get_entity_tree(line, constituents):
    tree = path_enclosed_tree(line, constituents)
    indices = [tree.leaf_treeposition(i) for i in range(len(tree.leaves()))]
    for i in indices:
        entity_type = tree[i[:-1]].entity_type
        if entity_type:
            tree[i] = entity_type
        else:
            tree[i] = '*'
    return tree


def get_bow_tree(line, constituents):
    """
    Return a bag-of-words tree representation, ex.
    (BOW (What *)(does *)(S.O.S. *)(stand *)(for *)(? *))
    """
    tree = path_enclosed_tree(line, constituents)
    leaves = set(tree.leaves())
    output = "(BOW "
    for leaf in leaves:
        output += "(" + leaf + " *)"
    output += ")"
    return output


def extract_features(lines):
    """
    Given lines of data, extracts features.
    """
    features = []
    j_path = os.getcwd() + '/data/jsons/'
    p_path = os.getcwd() + '/data/parsed/'
    curr_file = None
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
                get_bow_tree(line, constituents)
        f_list = [get_label(line),
                  '|BT|',
                  #tree_to_string(minimum_complete_tree(line, constituents)),
                  get_bow_tree(line, constituents),
                  '|BT|',
                  tree_to_string(path_enclosed_tree(line, constituents)),
                  '|BT',
                  tree_to_string(get_entity_tree(line, constituents)),
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
