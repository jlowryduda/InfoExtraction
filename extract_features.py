from __future__ import division
import sys
import csv

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

def get_cluster_dict():
    cluster_dict = {}
    with open("brown_c100.txt", "rb") as brown_cluster:
        for line in brown_cluster:
            token = line.split()[1]
            cluster_no = line.split()[0]
            cluster_dict[token] = cluster_no
    return cluster_dict

def get_brown_cluster(cluster_dict, token):
    """
    Returns an 8-bit representation of Brown cluster a given token belongs to.
    If not found, returns "00000000"
    """
    if token.lower() in cluster_dict:
        return cluster_dict[token.lower()][:8]
    else:
        return "00000000"

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

def get_loc_gazetteer():
    locations = []
    with open("GeoLite2-City-Locations-en.csv") as csv_file:
        loc_gazetter = csv.reader(csv_file)
        for row in loc_gazetter:
            country = row[5].lower()
            state = row[7].lower()
            city = row[10].lower()
            locations.extend([country, state, city])
    return set(locations)

def is_in_loc_gazeteer(locations, token):
    return "1" if token.lower() in locations else "0"

def extract_features(lines):
    """
    Goes line by line and compiles a new list of extracted features, which
    it then appends to the original feature list.
    """
    cluster_dict = get_cluster_dict() # read in Brown cluster
    loc_gazetteer = get_loc_gazetteer() # read in GeoLite2 data
    for i, line in enumerate(lines):
        token = line[1]
        bio_tag = lines[i].pop()
        additional_features = [is_capitalized(token),
                               contains_digits(token),
                               contains_dollar_sign(token),
                               get_length(token),
                               get_word_shape(token),
                               get_brown_cluster(cluster_dict, token),
                               is_in_loc_gazeteer(loc_gazetteer, token)]
        lines[i].extend(additional_features + [bio_tag])
    return lines

def write_to_file(filepath, lines):
    for i, line in enumerate(lines):
        lines[i] = '\t'.join(line) + '\n'
    with open(filepath, 'w') as outfile:
        for line in lines:
            outfile.write(line)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 3:
        print "Specify an input training file and an output filename"
    else:
        filename = sys.argv[1]
        new_filename = sys.argv[2]
        lines = read_from_file(filename)
        new_lines = extract_features(lines)
        write_to_file(new_filename, new_lines)

