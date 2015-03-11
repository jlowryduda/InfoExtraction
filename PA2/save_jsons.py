import os
import sys
import json

def save_jsons():
	path = '/home/j/xuen/teaching/cosi137/spring-2015/projects/project2/data/postagged-files/'
	destination = os.getcwd() + '/jsons/'
	for root, dirs, files in os.walk(path):
		for f in files:
			if f.endswith('.pos'):              
				with open(path + f, 'r') as infile:
					raw_string = infile.read()
				sents = raw_string.split('\n')
    			# Get rid of empty sentences:
    			sents = [sent.split() for sent in sents if len(sent) > 0]
    			for i, sent in enumerate(sents):
    				for j, token in enumerate(sent):
    					sent[j] = tuple(token.rsplit('_', 1))
    				sents[i] = sent
    			with open(destination + f[:-4] + '.json', 'w') as outfile:
			    	json.dump(sents, outfile)

if __name__ == '__main__':
	save_jsons()