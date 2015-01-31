import nltk
import sys
import glob

def is_capitalized(token):
    return '1' if token[0].isupper() else '0'

def contains_digits(token):
    return '1' if any([ch.isdigit() for ch in token]) else '0'

def contains_dollar_sign(token):
    return '1' if any([ch == "$" for ch in token]) else '0'

def get_length(token):
    return str(len(token))

def get_word_shape(token):
    shape = ''
    for char in token:
        if char.isdigit():
            shape += '#'
        elif char.isupper():
            shape += 'A'
        elif char.islower():
            shape += 'a'
        # Else, it's a non-alphanumeric symbol
        else:
            shape += '.'
    return shape

def read_from_file(filepath):
    """
    Takes a filepath, reads in the lines, and does some basic processing
    on them to separate them into their discrete elements.
    """
    with open(filepath, 'r') as infile:
        lines = infile.readlines()
    # Split lines into their discrete elements:
    lines = [line.split() for line in lines]
    # This yields some empty lists -- remove them:
    lines = [line for line in lines if len(line) > 0]
    return lines

def extract_features(lines):
    """
    Goes line by line and compiles a new list of extracted features, which
    it then appends to the original feature list.
    """
    for i, line in enumerate(lines):
        token = line[1]
        additional_features = [is_capitalized(token),
                               contains_digits(token),
                               contains_dollar_sign(token),
                               get_length(token),
                               get_word_shape(token)]
        lines[i].extend(additional_features)
    return lines

def write_to_file(filepath, lines):
    for i, line in enumerate(lines):
        lines[i] = '\t'.join(line) + '\n'
    with open(filepath, 'w') as outfile:
        for line in lines:
            outfile.write(line)


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print "Specify a path to an input directory"
    if len(sys.argv) ==  2:
        input_directory = sys.argv[1]
    filename = 'train.gold'
    new_filename = 'new_train.gold'
    lines = read_from_file(input_directory + filename)
    new_lines = extract_features(lines)
    write_to_file(input_directory + new_filename, new_lines)
