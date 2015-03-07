import sys
import os
import json
from nltk.tree import Tree

def read_from_file(filename):
    with open(filename, 'r') as infile:
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
        return "pos_match=1"
    else:
        return "pos_match=0"


def gender_match(line, sents):
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
    with open('names_genders.json', 'r') as infile:
        gender_dict = json.load(infile)

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
        return "gender_match=1"
    else:
        return "gender_match=0"


def get_distance(line):
    """ Returns the distance between sentences """
    return "distance=" + str(abs(int(line[1]) - int(line[6])))


def is_contained(line):
    return "is_contained=" + str(line[10] in line[5])


def get_label(line):
    return line[-1]


def entity_types_match(line):
    return "types_match=" + str(line[4] == line[9])


def antecedent_pronoun(line):
    return "antecedent_is_pronoun=" + str(line[5] in ['himself', 'herself', 'he', 'him', 'you', 'hers', 'her'])


def anaphor_pronoun(line):
    return "anaphor_is_pronoun=" + str(line[10] in ['himself', 'herself', 'he', 'him', 'you', 'hers', 'her'])


def both_proper_names(line, sents):
    span1 = sents[int(line[1])][int(line[2]):int(line[3])]
    span2 = sents[int(line[6])][int(line[7]):int(line[8])]
    tags1 = set([item[1] for item in span1])
    tags2 = set([item[1] for item in span2])
    return "both_proper_names=" + str(any(item.startswith("NNP") for item in tags1) and any(item.startswith("NNP") for item in tags2))

def is_appositive(line, constituents):
    if line[1] == line[6]:
        tree = constituents[int(line[1])]
        start = int(line[2])
        end = int(line[7])
        subtree = tree[tree.treeposition_spanning_leaves(start, end)]

        if len(subtree) >= 4:
            if (subtree.label() == 'NP' and     # Rooted at NP
                subtree[0].label() == 'NP' and  # First subtree is an NP
                subtree[1].label() == ',' and   # Second subtree is a comma
                subtree[3].label() == ','):     # Fourth subtree is a comma
                # We may very well be dealing with an appositive,
                # but more exploration of subtree[2] is needed.
                # print(subtree)
                return "is_appositive=1"
    else:
        return "is_appositive=0"

def extract_features(lines, train=False):
    features = []
    json_path = os.getcwd() + '/jsons/'
    parse_path = os.getcwd() + '/parsed/'
    current_file = None
    for line in lines:
    	if line[0] != current_file:
            current_file = line[0]
            with open(json_path + current_file + '.raw.json', 'r') as infile:
                sents = json.load(infile)
            with open(parse_path + current_file + '.raw.pos.parsed', 'r') as infile:
                sents = infile.read()
                sents = sents.split('\n\n')
                sents = [sent for sent in sents if len(sent) > 0]
                # Skip over dependency trees for now
                constituents = [s for i, s in enumerate(sents) if i % 2 == 0]
                constituents = [Tree.fromstring(c) for c in constituents]
        f_list = [get_distance(line),
                  is_contained(line),
                  entity_types_match(line),
                  gender_match(line, sents),
                  pos_match(line, sents),
                  antecedent_pronoun(line),
                  anaphor_pronoun(line),
                  both_proper_names(line, sents),
                  is_appositive(line, constituents)]
        if train:
            f_list.insert(0, get_label(line))
        features.append(f_list)
    return features


def write_to_file(filename, features):
    lines = [' '.join(f) for f in features]
    with open(filename, 'w') as outfile:
        for line in lines:
            outfile.write(line)
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
        lines = extract_features(lines, train)
        write_to_file(output_file, lines)
