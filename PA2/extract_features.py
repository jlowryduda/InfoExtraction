import sys
import os

def read_from_file(filename):
    with open(filename, 'r') as infile:
        lines = infile.readlines()

    lines = [line.split() for line in lines]
    return lines

def read_tagged_file(filename):
    """
    Reads in an original pos-tagged file and converts it to a list of
    sentences, each of which contains a list of (token, tag) pairs.  The
    resulting data structure can be indexed into with the indices from the
    train/dev/test data.
    """
    path = os.getcwd() + "/data/postagged-files/"
    with open(path + filename + '.raw.pos', 'r') as infile:
        raw_string = infile.read()
    sents = raw_string.split('\n')
    # Get rid of empty sentences:
    sents = [sent.split() for sent in sents if len(sent) > 0]
    for i, sent in enumerate(sents):
        for j, token in enumerate(sent):
            sent[j] = tuple(token.rsplit('_', 1))
        sents[i] = sent
    return sents

def pos_match(line):
    """
    Given a line of train/dev/test data, creates the document data structure
    for the appropriate file and indexes into it at the indices given in the
    line to get the POS-tags for each entity in the coreference pair. Returns
    true if intersection of two sets of POS-tags is non-empty. Otherwise,
    returns false if there are no POS-tags in common.
    """
    sents = read_tagged_file(line[0])
    span1 = sents[int(line[1])][int(line[2]):int(line[3])]
    span2 = sents[int(line[6])][int(line[7]):int(line[8])]
    tags1 = set([item[1] for item in span1])
    tags2 = set([item[1] for item in span2])
    if len(set.intersection(tags1, tags2)) > 0:
        return "pos_match=1"
    else:
        return "pos_match=0"

def get_distance(line):
    """ Returns the distance between sentences """
    return "distance=" + str(abs(int(line[1]) - int(line[6])))

def is_contained(line):
    return "is_contained=" + str(line[5] in line[10])

def get_label(line):
    return line[-1]

def entity_types_match(line):
    return "types_match=" + str(line[4] == line[9])

def extract_features(lines, train=False):
    features = []
    for line in lines:
        f_list = [get_distance(line),
                  is_contained(line),
                  entity_types_match(line),
                  pos_match(line)]
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
