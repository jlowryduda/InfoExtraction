import sys

def read_from_file(filename):
    with open(filename, 'r') as infile:
        lines = infile.readlines()

    lines = [line.split() for line in lines]
    return lines

def get_distance(line):
    """ Returns the distance between sentences """
    return "distance=" + str(abs(int(line[1]) - int(line[6])))

def get_label(line):
    return line[-1]

def extract_features(lines, train=False):
    features = []
    for line in lines:
        f_list = [get_distance(line)]
        if train:
            f_list.insert(0, get_label(line))
        features.append(f_list)
    return features

def write_to_file(filename, features):
    lines = [' '.join(f) for f in features]
    with open(filename, 'w') as outfile:
        outfile.writelines(lines)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Specify an input filename, an output filename, and"
        print "either 'train' or 'test' depending on datatype"
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

