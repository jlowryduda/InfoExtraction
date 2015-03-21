Steps to run an experiment (we'll write a shell script for this eventually):

1. Navigate to your InfoExtraction/PA3 directory.

2. Extract features for training data on a sample relation type:
	< python extract_features.py train PHYS.Located train >

3. Extract features for dev data on that same sample relation type:
	< python extract_features.py dev PHYS.Located train >

4. Navigate to your SVM-Light-TK directory.

5. Train the classifier:
	< svm_learn -t 5 -C + -S 1 -d 5 path/to/file/train.labeled model.data >

6. Classify new data:
	< svm_classify path/to/file/dev.labeled model.data >
	