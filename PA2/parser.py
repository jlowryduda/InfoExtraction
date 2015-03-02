#filepath = '/Users/slibenson/Desktop/InfoExtraction/PA2/data/postagged-files/'
#filename = 'APW20001001.2021.0521.head.coref.raw.pos'

def read_file(filepath, filename):
	"""
	Reads in an original pos-tagged file and converts it to a list of sentences,
	each of which contains a list of (token, tag) pairs.  The resulting data 
	structure can be indexed into with the indices from the train/dev/test data.
	"""

	with open(filepath + filename, 'r') as infile:
		raw_string = infile.read()
	sents = raw_string.split('\n')
	# Get rid of empty sentences:
	sents = [sent.split() for sent in sents if len(sent) > 0]
	for i, sent in enumerate(sents):
		for j, token in enumerate(sent):
			sent[j] = tuple(token.rsplit('_', 1))
		sents[i] = sent
	return sents