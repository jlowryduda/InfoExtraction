import sys
import os
import json
import re
import nltk
from nltk.tree import ParentedTree
from nltk.corpus import conll2000

##################
# HELPER METHODS #
##################

# Taken directly from NLTK book:
class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data)

    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

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
    if tree[-1][1] == "POS":
        # If the last word is tagged POS, return the previous word
        return tree[-2][0]
    else:
        for token, tag in reversed(tree):
            # Else search from right to left to find a child that is an NN, NNP,
            # NNPS, NX, POS, or JJR
            if tag in ["NN", "NNP", "NNPS", "NX", "POS", "JJR"]:
                return token
        for token, tag in tree:
            # Else search from left to right for the first child which is an NP
            if tag == "NP":
                return token
        for token, tag in reversed(tree):
            # Else search from right to left to find a child that is a $, ADJP,
            # PRP
            if tag in ["$", "ADJP", "PRP"]:
                return token
        for token, tag in reversed(tree):
            # Else search from right to left to find a child that is a CD
            if tag == "CD":
                return token
        for token, tag in reversed(tree):
            # Else search from right to left to find a child that is a JJ, JJS,
            # RB or QP
            if tag in ["JJ", "JJS", "RB", "QP"]:
                return token
        # Else return the last word
        return tree[-1][0]


#######################
# WORD-LEVEL FEATURES #
#######################

def get_wm1(line):
    """
    Return the words in the first mention
    """
    words = [clean_string(t) for t in line[7].split("_")]
    output = []
    for word in words:
        output.append("wm1_" + word + "=True")
    return " ".join(output)

def get_wm2(line):
    """
    Return the words in the second mention
    """
    words = [clean_string(t) for t in line[13].split("_")]
    output = []
    for word in words:
        output.append("wm2_" + word + "=True")
    return " ".join(output)

def wb_null(line):
    """
    Return true if there is no word between mentions.
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

def get_wbf(line, attributes):
    """
    Return the first word in between when at least two words in between
    """
    sentence_no = int(line[2])
    index_1_end = int(line[4])
    index_2_start = int(line[9])
    if (line[2] == line[8]) and (index_2_start-index_1_end >= 2):
        first_word_between = attributes[sentence_no][index_1_end]['token']
        return "first_word_between=" + first_word_between
    pass

def get_wbl(line, attributes):
    """
    Return the last word in between when at least two words in between
    """
    sentence_no = int(line[2])
    index_1_end = int(line[4])
    index_2_start = int(line[9])
    if (line[2] == line[8]) and (index_2_start-index_1_end >= 2):
        last_word_between = attributes[sentence_no][index_2_start - 1]['token']
        return "last_word_between=" + last_word_between
    pass

def first_word_before_m1(line, attributes):
    """
    Return the first word before mention 1
    """
    sentence_no = int(line[2])
    index_1_start = int(line[3])
    if index_1_start > 0:
        first_before_m1 = attributes[sentence_no][index_1_start-1]['token']
        return "first_word_before_m1=" + first_before_m1
    pass

def second_word_before_m1(line, attributes):
    """
    Return the second word before mention 1
    """
    sentence_no = int(line[2])
    index_1_start = int(line[3])
    if index_1_start > 1:
        first_before_m1 = attributes[sentence_no][index_1_start-2]['token']
        return "first_word_before_m1=" + first_before_m1
    pass

def first_word_after_m2(line, attributes):
    """
    Return the first word ater mention 2
    """
    sentence_no = int(line[2])
    index_2_end = int(line[10])
    try:
        first_word_after_m2 = attributes[sentence_no][index_2_end]['token']
        return "first_word_after_m1" + first_word_after_m2
    except:
        pass

def second_word_after_m2(line, attributes):
    """
    Return the second word after mention 2
    """
    sent_num = int(line[2])
    index_2_end = int(line[10])
    try:
        second_word_after_m2 = attributes[sent_num][index_2_end + 1]['token']
        return "second_word_after_m1" + second_word_after_m2
    except:
        pass

#################
# FLAT FEATURES #
#################

def entity_type_pair(line):
    """
    Returns a string containing entity types of both mentions, a la
    Jiang et al.
    """
    feature = line[5] + "-" + line[11]
    return "entity_type_pair=" + feature

def pos1(line, attributes):
    sent = attributes[int(line[2])]
    pos_span = range(int(line[3]), int(line[4]))
    poses = list(set([sent[i]['pos'] for i in pos_span]))
    output = []
    for pos in poses:
        output.append("pos1_" + pos + "=True")
    return " ".join(output)

def pos2(line, attributes):
    sent = attributes[int(line[2])]
    pos_span = range(int(line[9]), int(line[10]))
    poses = list(set([sent[i]['pos'] for i in pos_span]))
    output = []
    for pos in poses:
        output.append("pos2_" + pos + "=True")
    return " ".join(output)

def mention_1_possessive(line, attributes):
    """
    Returns true if the first mention is a possessive pronoun.
    """
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
    and the first token of the second mention.
    """
    return "token_distance=" + str(int(line[9]) - int(line[4]))



#########################
# PHRASE-LEVEL FEATURES #
#########################

def no_interceding_chunk(line, attributes, chunker):
    """
    Returns true if there is no complete chunk between the two mentions.
    """
    sent = [(item['token'], item['pos']) for item in attributes[int(line[2])]]
    chunks = chunker.parse(sent)
    span = range(int(line[4]), int(line[9]))
    for index in span:
        if chunks[chunks.leaf_treeposition(index)[:-1]].label() == 'NP':
            return
    return "no_interceding_chunk=1"

def number_interceding_chunks(line, attributes, chunker):
    """
    Returns the number of complete chunks between the two mentions.
    """
    sent = [(item['token'], item['pos']) for item in attributes[int(line[2])]]
    chunks = chunker.parse(sent)
    tree_index_mention_1 = chunks.leaf_treeposition(int(line[4]))[0]
    tree_index_mention_2 = chunks.leaf_treeposition(int(line[9]))[0]
    i = 0
    for item in chunks[tree_index_mention_1:tree_index_mention_2]:
        if isinstance(item, Tree):
            i += 1
    return "number_interceding_chunks=" + str(i)

def path_of_phrase_labels(line, constituents, attributes):
    """
    Returns the syntactic path between the two mentions, removing duplicate
    phrase labels.
    """
    tree = constituents[int(line[2])]
    path_up, path_down = get_paths(tree, int(line[3]), int(line[10]))
    path = path_up + path_down
    # remove repetitions, ex. NP NP NP ---> NP
    output_path = []
    for index, element in enumerate(path):
        if index == 0 or element != output_path[-1]:
            output_path.append(element)
    output_path = "_".join(output_path)
    return "path=" + output_path

def head_mention(line, attributes, mention_number, chunker):
    """
    Returns the head of the NP chunk immediately governing the mention.
    """
    sent = [(item['token'], item['pos']) for item in attributes[int(line[2])]]
    chunks = chunker.parse(sent)

    if mention_number == 1:
        start = int(line[3])
        end = int(line[4]) - 1
    elif mention_number == 2:
        start = int(line[9])
        end = int(line[10]) - 1

    if chunks[chunks.leaf_treeposition(start)[:-1]].label() == 'S':
        # Not a chunk
        head = chunks.leaves()[start][0]
    elif chunks[chunks.leaf_treeposition(start)[:-1]].label() == 'NP':
        # A chunk
        head = get_head(chunks[chunks.leaf_treeposition(start)[:-1]])
    if mention_number == 1:
        return "head_mention_1=" + head
    elif mention_number == 2:
        return "head_mention_2=" + head

#######################
# TREE-LEVEL FEATURES #
#######################

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


######################
# Semantic Resources #
######################

def geo_identity(line, geo_dict):
    """
    If the mentions are both a GPE, check type of GPE by consulting GeoLite2
    database. This maps ex. "Warsaw" to "city" and "North Dakota" to "state".
    """
    if line[5] == "GPE" and line[11] == "GPE":
        cond_1 = False
        cond_2 = False
        try:
            cond_1 = clean_string(line[13]) in geo_dict[clean_string(line[7])]
        except:
            pass
        try:
            cond_2 = clean_string(line[7]) in geo_dict[clean_string(line[13])]
        except:
            pass
        if cond_1 or cond_2:
            return "geo_identity=True"
    pass

def entity_1_type_country(line, attributes, geo_dict):
    """
    Returns true if mention 1 is a country and mention 2 is a person.
    """
    sent_number = int(line[2])
    mention_1_start = int(line[3])
    mention_1 = clean_string(" ".join(line[7].split("_")))
    mention_1_type = attributes[sent_number][mention_1_start]['entity_type']
    try:
        cond = "country" in geo_dict[mention_1] and mention_2_type == "PER"
    except:
        cond = False
    if cond:
        return "entity_1_country=True"
    pass

def entity_2_type_country(line, attributes, geo_dict):
    """
    Returns true if mention 1 is a person and mention 2 is a country.
    """
    sent_number = int(line[2])
    mention_2_start = int(line[9])
    mention_2 = clean_string(" ".join(line[13].split("_")))
    mention_2_type = attributes[sent_number][mention_2_start]['entity_type']
    try:
        cond = "country" in geo_dict[mention_2] and mention_1_type == "PER"
    except:
        cond = False
    if cond:
        return "entity_2_type_country=True"
    pass

def possessive_plus_family(line, attributes):
    """
    Returns true if the first mention is a possessive pronoun and the second
    mention is contained within a list of family relation terms.
    """
    sent = attributes[int(line[2])]
    family = ['father', 'mother', 'parents', 'wife', 'husband', 'kids', 'son',
              'sons', 'children', 'grandchildren', 'daughter', 'brother',
              'sister', 'niece', 'nephew', 'cousin', 'aunt', 'uncle']
    if sent[int(line[4])-1]['pos'] == 'PRP$':
        if set(line[13].split("_")).intersection(family):
            return "possessive_plus_family=True"
    pass


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

    # Train the chunker:
    train_chunks = conll2000.chunked_sents('train.txt', chunk_types=['NP'])
    chunker = UnigramChunker(train_chunks)

    curr_file = None
    for line in lines:
        if line[1] != curr_file:
            curr_file = line[1]
            print(curr_file)
            with open(a_path + curr_file + '.json', 'r') as infile:
                attributes = json.load(infile)
            with open('geo_knowledge.json', 'r') as infile:
                geo_dict = json.load(infile)
            with open(p_path + curr_file + '.parsed', 'r') as infile:
                parses = infile.read()
                parses = parses.split('\n\n')
                parses = [sent for sent in parses if len(sent) > 0]
                constituents = [s for i, s in enumerate(parses) if i % 2 == 0]
                dependencies = [s.split('\n') for i, s in enumerate(parses)
                                if i % 2 == 1]
                constituents = [ParentedTree.fromstring(c) 
                                for c in constituents]
        f_list = [get_label(line),
                  entity_type_pair(line),
                  mention_1_possessive(line, attributes),
                  interceding_prep(line, attributes),
                  interceding_conj(line, attributes),
                  get_wm1(line),
                  get_wm2(line),
                  wb_null(line),
                  word_between(line, attributes),
                  no_interceding_chunk(line, attributes, chunker),
                  token_distance(line),
                  governing_constituents(line, constituents),
                  tree_distance(line, constituents),
                  pos1(line, attributes),
                  pos2(line, attributes),
                  possessive_plus_family(line, attributes),
                  entity_1_type_country(line, attributes, geo_dict),
                  entity_2_type_country(line, attributes, geo_dict),
                  entity_type_country(line, attributes, geo_dict),
                  path_of_phrase_labels(line, constituents, attributes),
                  head_mention(line, attributes, 1, chunker),
                  head_mention(line, attributes, 2, chunker),
                  number_interceding_chunks(line, attributes, chunker),
                  get_wbf(line, attributes),
                  get_wbl(line, attributes),
                  #first_word_before_m1(line, attributes),
                  #second_word_before_m1(line, attributes),
                  #first_word_after_m2(line, attributes),
                  #second_word_after_m2(line, attributes),
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
