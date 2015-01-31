import nltk
import sys
import glob

def is_capitalized(token):
    return 1 if token[0].isupper() else 0

def contains_digits(token):
    return 1 if any([ch.isdigit() for ch in token]) else 0

def contains_dollar_sign(token):
    return 1 if any([ch == "$" for ch in token]) else 0

def get_length(token):
    return len(token)

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 2:
        print "Specify a path to an input directory"
    if len(sys.argv) ==  2:
        input_directory = sys.argv[1]
        for train_file in glob.glob(input_directory + "train.gold"):
            with open(train_file) as train:
                for line in train:
                    line = line.split()
                    if line:
                        token = line[1]
                        if contains_dollar_sign(token):
                            print token
