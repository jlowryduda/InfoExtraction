filename = 'train.features.labeled'

with open(filename, 'r') as infile:
    lines = infile.readlines()

lines = [line.split() for line in lines]
labels = ['yes', 'no']
features = []
for line in lines:
    for feature in line:
        if feature not in features:
            features.append(feature)

features.sort()
features = [f for f in features if (not f.startswith('distance=') and
                                    f != 'yes' and f != 'no')]

def compute_precision(results):
    tp, tn, fp, fn = results
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 1.0

def compute_recall(results):
    tp, tn, fp, fn = results
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 1.0

def compute_fmeasure(results):
    tp, tn, fp, fn = results
    try:
        return 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        return 1.0
    

for feature in features:
    tp, tn, fp, fn = 0., 0., 0., 0.
    for line in lines:
        if line[0] == 'yes' and feature in line:
            tp += 1
        elif line[0] == 'no' and feature in line:
            fp += 1
        elif line[0] == 'yes' and feature not in line:
            fn += 1
        else:
            tn += 1
        results = [tp, tn, fp, fn]
    print feature, results
    print feature + ' Precision: ', compute_precision(results)
    print feature + ' Recall: ', compute_recall(results)
    print feature + ' F-measure: ', compute_fmeasure(results)
    print '\n'
        
