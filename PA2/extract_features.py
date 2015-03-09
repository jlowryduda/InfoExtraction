import sys
import os
import json
import re

def read_from_file(filename):
    with open(os.getcwd() + '/data/' + filename, 'r') as infile:
        lines = infile.readlines()

    lines = [line.split() for line in lines]
    return lines


def pos_match(line, sents):
    """
    Given a line of train/dev/test data, and the document data structure
    for the appropriate file, it indexes into it at the indices given in the
    line to get the POS-tags for each entity in the coreference pair. Returns
    true if intersection of two sets of POS-tags is non-empty. Otherwise,
    returns false if there are no POS-tags in common.
    """
    span1 = sents[int(line[1])][int(line[2]):int(line[3])]
    span2 = sents[int(line[6])][int(line[7]):int(line[8])]
    tags1 = set([item[1] for item in span1])
    tags2 = set([item[1] for item in span2])
    if len(set.intersection(tags1, tags2)) > 0:
        return True
    return False


def is_singular_prp(word, tag):
    """Check if the word is a singular personal pronoun"""

    # list of singular PRP and PRP$
    singular_prp =\
        set(['it', 'its', 'he', 'his', 'him', 'her', 'hers',
             'i', 'my', 'mine', 'me'])
    return (tag == ('PRP' or 'PRP$') and (word.lower() in singular_prp))


def is_plural_prp(word, tag):
    """Check if word is a plural personal pronoun"""

    # list of plural PRP and PRP$
    plural_prp =\
        set(['they', 'theirs', 'their', 'them', 'we', 'our', 'ours', 'us'])

    return (tag == ('PRP' or 'PRP$') and (word.lower() in plural_prp))


def clean_string(s):
    s = s.replace("`", "")
    s = s.replace("(", "")
    s = s.replace(")", "")
    s = s.lower()
    return s


def exact_match(line, sents):
    if clean_string(line[5]) == clean_string(line[10]) and pos_match(line, sents):
        return "exact_match=True"
    else:
        pass


def number_match(line, sents):
    """Check if both mentions have the same number. Assume cardinality of 1
    for both sets of tags"""
    span1 = sents[int(line[1])][int(line[2]):int(line[3])]
    span2 = sents[int(line[6])][int(line[7]):int(line[8])]
    tag1 = iter(set([item[1] for item in span1])).next()
    tag2 = iter(set([item[1] for item in span2])).next()

    singular_tags = ('NN', 'NNP')
    plural_tags = ('NNS', 'NNPS')
    # both singular
    both_nn = ((tag1 == 'NN') and (tag2 == 'NN'))
    both_nnp = ((tag1 == 'NNP') and (tag2 == 'NNP'))
    nn_and_nnp = ((tag1 == 'NN') and (tag2 == 'NNP'))
    nnp_and_nn = ((tag1 == 'NNP') and (tag2 == 'NN'))
    sg_prp_match = ((tag1 == ('NN' or 'NNP') and
                     is_singular_prp(span2[0][0], tag2)) or
                    is_singular_prp(span1[0][0], tag1) and
                    tag2 == ('NN' or 'NNP'))

    both_singular = (both_nn or both_nnp or nn_and_nnp or
                     nnp_and_nn or sg_prp_match)

    # both plural
    both_nns = ((tag1 == 'NNS') and (tag2 == 'NNS'))
    both_nnps = ((tag1 == 'NNPS') and (tag2 == 'NNPS'))
    nns_and_nnps = ((tag1 == 'NNS') and (tag2 == 'NNPS'))
    nnps_and_nns = ((tag1 == 'NNPS') and (tag2 == 'NNS'))

    pl_prp_match = ((tag1 == ('NNS' or 'NNPS') and
                     is_plural_prp(span2[0][0], tag2)) or
                    is_plural_prp(span1[0][0], tag1) and
                    tag2 == ('NNS' or 'NNPS'))

    both_plural = (both_nns or both_nnps or nns_and_nnps or
                   nnps_and_nns or pl_prp_match)

    if (both_plural or both_singular):
        return True
    return False


def gender_match(line, sents, gender_dict):
    """
    Given a line containing a potential coreference pair, and a sentence data
    structure, determine whether or not the pair has the same gender.
    """
    # Assume that the first name part in any given span of tokens will be
    # most indicative of gender (because it's most likely the first name):
    names = [sents[int(line[1])][int(line[2])][0],
             sents[int(line[6])][int(line[7])][0]]
    ents = [line[4], line[9]]
    genders = ['N', 'N'] # Default to neutral gender

    # If the entity is a person, check to see if the person's gender is in the
    # gender dictionary as either male or female, and if so, reassign
    for i in range(len(names)):
        if ents[i] == 'PER':
            if names[i] in gender_dict:
                if gender_dict[names[i]] == 'male':
                    genders[i] = 'M'
                elif gender_dict[names[i]] == 'female':
                    genders[i] = 'F'

    if genders[0] == genders[1]:
        return True
    return False


def agreement(line, sents, gender_dict):
    if gender_match(line, sents, gender_dict) and number_match(line, sents):
        return "agreement=True"
    pass


def get_distance(line):
    """ Returns the distance between sentences """
    distance = abs(int(line[1]) - int(line[6]))
    if distance > 4:
        distance = 4
    return "distance=" + str(distance)


def same_sentence(line):
    """ Returns True if mentions are in the same sentence, else False """
    if int(line[1]) == int(line[6]):
        return "same_sentence=True"
    pass


def is_contained(line):
    if (line[10] in line[5]) or (line[5] in line[10]):
        return "is_contained=True"
    pass


def get_label(line):
    return line[-1]


def entity_types_match(line):
    if line[4] == line[9]:
        return "types_match=True"
    pass


def antecedent_pronoun(line):
    pronouns = ['himself', 'herself', 'he', 'him', 'you', 'hers', 'her']
    if line[5] in pronouns:
        return  "antecedent_is_pronoun=True"
    pass


def anaphor_pronoun(line):
    pronouns = ['himself', 'herself', 'he', 'him', 'you', 'hers', 'her']
    if line[10] in pronouns:
        return "anaphor_is_pronoun=True"
    pass


def both_proper_names(line, sents):
    span1 = sents[int(line[1])][int(line[2]):int(line[3])]
    span2 = sents[int(line[6])][int(line[7]):int(line[8])]
    tags1 = any([item[1] for item in span1 if item[1].startswith('NNP')])
    tags2 = any([item[1] for item in span2 if item[1].startswith('NNP')])
    if tags1 and tags2:
        return "both_proper_names=True"
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


def is_appositive(line, dependencies):
    if line[1] == line[6]:
        start_1 = int(line[2]) + 1
        end_1 = int(line[3]) + 1
        start_2 = int(line[7]) + 1
        end_2 = int(line[8]) + 1

        # index_range represents a list of all the indices covered by spans:
        index_range = range(start_1, end_1) + range(start_2, end_2)
        dependency = dependencies[int(line[1])]
        relations = [item for item in dependency if item.startswith('appos(')]
        # Extract indices from dependencies using a regular expression:
        pattern = '-(\d+)[,)]'
        for rel in relations:
            indices = [int(index) for index in re.findall(pattern, rel)]
            if (indices[0] in index_range and indices[1] in index_range):
                return "is_appositive=True"
    pass


def is_copula(line, dependencies):
    if line[1] == line[6]:
        start_1 = int(line[2]) + 1
        end_1 = int(line[3]) + 1
        start_2 = int(line[7]) + 1
        end_2 = int(line[8]) + 1

        index_range = range(start_1, end_1) + range(start_2, end_2)
        dependency = dependencies[int(line[1])]

        pattern = '-(\d+)[,)]'
        for i, line in enumerate(dependency):
            if line.startswith('nsubj'):
                indices = [int(index) for index in re.findall(pattern, line)]
                if (indices[0] in index_range and indices[1] in index_range):
                    for j, rel in enumerate(dependency[i+1:]):
                        if rel.startswith('nsubj('):
                            break
                        if rel.startswith('cop('):
                            cop_indices = [int(index) for
                                           index in re.findall(pattern, rel)]
                            if (cop_indices[0] in index_range or
                                cop_indices[1] in index_range):
                                return "is_copula=True"
    pass


def tree_distance(line, constituents):
    if line[1] == line[6]:
        tree = constituents[int(line[1])]
        indices = [int(line[2]), int(line[3]), int(line[7]), int(line[8])]
        start = min(indices)
        end = min(max(indices), len(tree.leaves()))
        path_up, path_down = get_paths(tree, start, end)
        distance = len(path_up + path_down) - 1
        return "tree_distance=" + str(distance)
    pass


def anaphor_definite(line, constituents):
    mention_1_start = int(line[2])
    mention_2_start = int(line[7])
    if mention_1_start > mention_2_start:
        tree = constituents[int(line[1])]
        start = int(line[2])
        end = min(int(line[3]), len(tree.leaves()))
    else:
        tree = constituents[int(line[6])]
        start = int(line[7])
        end = min(int(line[8]), len(tree.leaves()))
    first_token = ""
    try:
        subtree = tree[tree.treeposition_spanning_leaves(start, end)]
        if isinstance(subtree, Tree):
            first_token = subtree[0].leaves()[0].lower()
        else:
            first_token = subtree.lower()
        if first_token == "the":
            return "anaphor_definite=True"
        pass
    except:
        pass


def anaphor_demonstrative(line, constituents):
    mention_1_start = int(line[2])
    mention_2_start = int(line[7])
    if mention_1_start > mention_2_start:
        tree = constituents[int(line[1])]
        start = int(line[2])
        end = min(int(line[3]), len(tree.leaves()))
    else:
        tree = constituents[int(line[6])]
        start = int(line[7])
        end = min(int(line[8]), len(tree.leaves()))
    first_token = ""
    try:
        subtree = tree[tree.treeposition_spanning_leaves(start, end)]
        if isinstance(subtree, Tree):
            first_token = subtree[0].leaves()[0].lower()
        else:
            first_token = subtree.lower()
        if first_token in ("this", "that", "these", "those"):
            return "anaphor_definite=True"
        else:
            pass
    except:
        pass


def adjacent_subjects(line, dependencies, sents):
    """
    Given a line and a set of dependencies, checks if the mentions are the
    subjects of adjacent sentences.
    """
    mention_1_sentence = int(line[1])
    mention_2_sentence = int(line[6])

    # Adjacent sentence?
    if ((abs(mention_1_sentence - mention_2_sentence) == 1) and not
        both_proper_names(line, sents)):
        dependency_1 = dependencies[int(line[1])]
        dependency_2 = dependencies[int(line[6])]

        start_1 = int(line[2]) + 1
        end_1 = int(line[3]) + 1
        start_2 = int(line[7]) + 1
        end_2 = int(line[8]) + 1

        range_1 = range(start_1, end_1)
        range_2 = range(start_2, end_2)

        pattern = '-(\d+)[,)]'
        indices_1 = [int(i) for i in re.findall(pattern, dependency_1[0])]
        indices_2 = [int(i) for i in re.findall(pattern, dependency_2[0])]

        if (indices_1[0] in range_1 or indices_1[1] in range_1):
            if (indices_2[0] in range_2 or indices_2[1] in range_2):
                return "adjacent_subjects=True"
    pass


def jaccard_coefficient(line, sents):
    span1 = sents[int(line[1])][int(line[2]):int(line[3])]
    span2 = sents[int(line[6])][int(line[7]):int(line[8])]
    tokens1 = set([item[0] for item in span1])
    tokens2 = set([item[0] for item in span2])
    intersection = set.intersection(tokens1, tokens2)
    union = set.union(tokens1, tokens2)
    jc = len(intersection) / float(len(union))
    if jc > 0.25:
        return "jaccard_coefficient=" + str(jc)
    else:
        return "jaccard_coefficient=<0.25"


def get_head(tree):
    if tree[-1].label() == "POS":
        return tree[-2].leaves()
    else:
        for e in reversed(tree):
            if e.label() in ["NN", "NNP", "NNPS", "NX", "POS", "JJR"]:
                return " ".join(e.leaves())
        for e in tree:
            if e.label() == "NP":
                return " ".join(e.leaves())
        for e in reversed(tree):
            if e.label() in ["$", "ADJP", "PRN"]:
                return " ".join(e.leaves())
        for e in reversed(tree):
            if e.label() == "CD":
                return " ".join(e.leaves())
        for e in reversed(tree):
            if e.label() in ["JJ", "JJS", "RB", "QP"]:
                return " ".join(e.leaves())
        return " ".join(tree[-1].leaves())


def head_match(line, constituents):
    tree_1 = constituents[int(line[1])]
    tree_2 = constituents[int(line[6])]
    start_1 = int(line[2])
    end_1 = int(line[3])
    start_2 = int(line[7])
    end_2 = int(line[8])
    head_1 = ""
    head_2 = ""
    subtree_1 = tree_1[tree_1.treeposition_spanning_leaves(start_1, end_1)]
    subtree_2 = tree_2[tree_2.treeposition_spanning_leaves(start_2, end_2)]
    if isinstance(subtree_1, Tree):
        head_1 = get_head(subtree_1)
    else:
        head_1 = subtree_1
    if isinstance(subtree_2, Tree):
        head_2 = get_head(subtree_2)
    else:
        head_2 = subtree_2
    if head_1 == head_2:
        return "head_match=True"
    pass


def wh_clause(line):
    if line[1] == line[6]:
        if line[4] == line[9]:
            indices = sorted([int(line[2]), int(line[3]), int(line[7]), int(line[8])])
            start = indices[0]
            end = indices[-1]
            if (end - start) <= 4:
                if clean_string(line[10]) in ["who", "which", "whose", "that"]:
                    return "wh_clause=True"

    pass


def extract_features(lines):
    features = []
    j_path = os.getcwd() + '/data/jsons/'
    p_path = os.getcwd() + '/data/parsed/'
    curr_file = None
    with open('names_genders.json', 'r') as infile:
        gender_dict = json.load(infile)
    for line in lines:
    	if line[0] != curr_file:
            curr_file = line[0]
            print(curr_file)
            with open(j_path + curr_file + '.raw.json', 'r') as infile:
                sents = json.load(infile)
            with open(p_path + curr_file + '.raw.pos.parsed', 'r') as infile:
                parses = infile.read()
                parses = parses.split('\n\n')
                parses = [sent for sent in parses if len(sent) > 0]
                # Skip over dependency trees for now
                constituents = [s for i, s in enumerate(parses) if i % 3 == 1]
                dependencies = [s.split('\n') for i, s in enumerate(parses)
                                if i % 3 == 2]
                constituents = [Tree.fromstring(c) for c in constituents]
        f_list = [get_label(line),
                  #get_distance(line),
                  exact_match(line, sents),
                  #is_contained(line),
                  entity_types_match(line),
                  #pos_match(line, sents),
                  #antecedent_pronoun(line),
                  #anaphor_pronoun(line),
                  #same_sentence(line),
                  #both_proper_names(line, sents),
                  #anaphor_definite(line, constituents),
                  #anaphor_demonstrative(line, constituents),
                  #tree_distance(line, constituents),
                  agreement(line, sents, gender_dict),
                  jaccard_coefficient(line, sents),
                  adjacent_subjects(line, dependencies, sents),
                  head_match(line, constituents),
                  wh_clause(line),
                  #is_copula(line, dependencies),
                  is_appositive(line, dependencies)]
        f_list = [f for f in f_list if f is not None]
        features.append(f_list)
    return features


def write_to_file(filename, features, train=False):
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
    from nltk.tree import Tree
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
