import sys
import os
import json
import re
from nltk.tree import ParentedTree

##################
# HELPER METHODS #
##################

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

def get_paths(tree, start, end):
    """
    Given a tree, and the indices of the starting and ending leaves,
    calculates the path up from the start leaf to the lowest common ancestor
    and the path down from the lowest common ancestor to the end leaf.
    """

    # Find tree positions of start and end leaves:
    start_pos = tree.leaf_treeposition(start)
    end_pos = tree.leaf_treeposition(end-1)

    # Find tree position of subtree rooted at lowest common ancestor:
    for i in range(min(len(start_pos), len(end_pos))):
        if start_pos[i] != end_pos[i]:
            break
    subtree_pos = start_pos[:i]

    # Refer to tree positions of start and end leaves in terms of subtree:
    revised_start = start_pos[len(subtree_pos):]
    revised_end = end_pos[len(subtree_pos):]

    # Calculate paths:
    path_up = [tree[subtree_pos][revised_start[:i]].label()
               for i in range(len(revised_start))]
    path_up.reverse()

    path_down = [tree[subtree_pos][revised_end[:i]].label()
                 for i in range(len(revised_end))]

    # Resulting path_up ends with, path_down starts with, dominating NP
    return path_up, path_down


def get_head(tree):
    """
    Determine the head of an NP constituent using Collins' (Collins, 1999)
    head-finding algorithm.
    """
    if tree[-1].label() == "POS":
        # If the last word is tagged POS, return the previous word
        return " ".join(tree[-2].leaves())
    else:
        for e in reversed(tree):
            # Else search from right to left to find a child that is an NN, NNP,
            # NNPS, NX, POS, or JJR
            if e.label() in ["NN", "NNP", "NNPS", "NX", "POS", "JJR"]:
                return " ".join(e.leaves())
        for e in tree:
            # Else search from left to right for the first child which is an NP
            if e.label() == "NP":
                return " ".join(e.leaves())
        for e in reversed(tree):
            # Else search from right to left to find a child that is a $, ADJP,
            # PRP
            if e.label() in ["$", "ADJP", "PRP"]:
                return " ".join(e.leaves())
        for e in reversed(tree):
            # Else search from right to left to find a child that is a CD
            if e.label() == "CD":
                return " ".join(e.leaves())
        for e in reversed(tree):
            # Else search from right to left to find a child that is a JJ, JJS,
            # RB or QP
            if e.label() in ["JJ", "JJS", "RB", "QP"]:
                return " ".join(e.leaves())
        # Else return the last word
        return " ".join(tree[-1].leaves())

#################
# Flat features #
#################


def entity_type_pair(line):
    """
    Returns a string containing entity types of both mentions, a la
    Jiang et al.
    """
    feature = line[5] + "-" + line[11]
    return "entity_type_pair=" + feature

def mention_1_possessive(line, attributes):
    sent = attributes[int(line[2])]
    if sent[int(line[4])-1]['pos'] == 'PRP$':
        return "mention_1_possessive=1"
    pass

def interceding_prep(line, attributes):
    """
    If a preposition appears between the two mentions, return that preposition.
    """
    sent = attributes[int(line[2])]
    interceding_span = sent[int(line[4]):int(line[9])]
    tokens = [clean_string(item['token']) for item in interceding_span
              if item['pos'] == 'IN']
    if tokens:
        return "interceding_prep=" + tokens[-1]
    pass

def interceding_conj(line, attributes):
    """
    If a conjunction appears between the two mentions, return that conjunction.
    """
    sent = attributes[int(line[2])]
    interceding_span = sent[int(line[4]):int(line[9])]
    tokens = [clean_string(item['token']) for item in interceding_span
              if item['pos'] == 'CC']
    if tokens:
        return "interceding_conj=" + tokens[0]
    pass

def token_distance(line):
    """
    Returns the distance between the last token of the first mention
    and the first token of the second mention
    """
    return "token_distance=" + str(int(line[9]) - int(line[4]))

def governing_constituents(line, constituents):
    """
    Returns the constituent labels of constituents governing each mention, in
    a pair such as 'NP-PP'.  If the mention is a single token, we use the
    grandparent of its part-of-speech constituent.  If it's multiple tokens,
    we use the parent of its subtree.
    """
    tree = constituents[int(line[2])]
    
    # Mention 1:
    start_1 = int(line[3])
    end_1 = int(line[4])
    if end_1 - start_1 == 1:
        label_1 = tree[tree.leaf_treeposition(start_1)[:-3]].label()
    else:
        subtree = tree[tree.treeposition_spanning_leaves(start_1, end_1)[:-1]]
        label_1 = subtree.label()
        
    # Mention 2:
    start_2 = int(line[9])
    end_2 = int(line[10])
    if end_2 - start_2 == 1:
        label_2 = tree[tree.leaf_treeposition(start_2)[:-3]].label()
    else:
        subtree = tree[tree.treeposition_spanning_leaves(start_2, end_2)[:-1]]
        label_2 = subtree.label()
    return "governing_constituents=" + label_1 + "-" + label_2

def tree_distance(line, constituents):
    tree = constituents[int(line[2])]
    path_up, path_down = get_paths(tree, int(line[3]), int(line[10]) - 1)
    return "tree_distance=" + str(len(path_up + path_down))
    
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

def head_mention(line, constituents, mention_number):
    """
    Returns the head of the NP immediately governing the mention stipulated
    """
    tree = constituents[int(line[2])]
    if mention_number == 1:
        start = int(line[3])
        end = int(line[4])
    elif mention_number == 2:
        start = int(line[9])
        end = int(line[10])
    subtree = tree[tree.treeposition_spanning_leaves(start, end)]
    if isinstance(subtree, ParentedTree):
        head = get_head(subtree)
    else:
        head = subtree
    if mention_number == 1:
        return "head_mention_1=" + head
    elif mention_number == 2:
        return "head_mention_2=" + head



####################
# Extract Features #
####################

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
                constituents = [ParentedTree.fromstring(c) for c in constituents]
        f_list = [get_label(line),
                  entity_type_pair(line),
                  mention_1_possessive(line, attributes),
                  interceding_prep(line, attributes),
                  interceding_conj(line, attributes),
                  get_wm1(line),
                  get_wm2(line),
                  wb_null(line),
                  word_between(line, attributes),
                  token_distance(line),
                  governing_constituents(line, constituents),
                  tree_distance(line, constituents)
                  #head_mention(line, constituents, 1),
                  #head_mention(line, constituents, 2)
                  ]
        features.append([f for f in f_list if f is not None])
    return features

def write_to_file(filename, features, train=False):
    """
    Write resulting features to a feature file.  If this isn't training data,
    produce two files, one with labels and one without.
    """
    with open(filename + '.labeled', 'w') as outfile:
        for f in features:
            outfile.write(' '.join(f))
            outfile.write('\n')
    if not train:
        with open(filename + '.nolabel', 'w') as outfile:
            for f in features:
                outfile.write(' '.join(f[1:]))
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
