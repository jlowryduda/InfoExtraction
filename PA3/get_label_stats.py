import os

def read_from_file(filename):
    """
    Read in data from file and do preprocessing on it for later use.
    """
    with open(os.getcwd() + '/data/' + filename + '.original', 'r') as infile:
        lines = infile.readlines()

    lines = [line.split() for line in lines]
    return lines

def read_lines(lines):
    label_dict = dict()
    for line in lines:
        if line[0] in label_dict:
            label_dict[line[0]] += 1
        else:
            label_dict[line[0]] = 1
    del label_dict['no_rel']
    return label_dict

def create_label_dicts():
    file_list = ['train', 'dev', 'test']
    for file_name in file_list:
        lines = read_from_file(file_name)
        label_dict = read_lines(lines)
        print file_name
        print '\n'
        for k in sorted(label_dict, key=label_dict.get, reverse=True):
            print k, label_dict[k]
        print '\n\n'

create_label_dicts()
