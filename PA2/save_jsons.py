import os
import sys
import json

def save_jsons():
	data_path = '/home/j/xuen/teaching/cosi137/spring-2015/projects/project2/data/postagged-files/'
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

def json_to_sgm():
	for root, dirs, files in os.walk(json_path):
		for f in files:
			with open(path + f, 'r') as infile:
				sents = json.load(infile)
			with open(sgm_path + f[:-5] + '.sgm', 'w') as outfile:
				for sent in sents:
					formatted = '<s> '+' '.join([w for (w, t) in sent])+' </s>'
					outfile.write(formatted)
					outfile.write('\n')


def sgm_to_trees():
	i = 0
	for root, dirs, files in os.walk(sgm_path):
		for f in files:
			i += 1
			print(i)
			os.system('/home/j/clp/chinese/bin/chariak-parse.sh ' + sgm_path + 
				f + ' > ' + parsed_path + f[:-4] + '.parse')

if __name__ == '__main__':
	json_path = os.getcwd() + '/jsons/'
	sgm_path = os.getcwd() + '/sgms/'
	parsed_path = os.getcwd() + '/parsed_files/'
	"""
	print('Converting pos-tagged files to jsons')
	save_jsons()
	print('Converting jsons to sgm format')
	json_to_sgm()
	"""
	print('Parsing sgm files')
	sgm_to_trees()