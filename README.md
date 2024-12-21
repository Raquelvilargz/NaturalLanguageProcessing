# Natural Language Processing
Python assignments for the IIT CS585 course

## Assignment 1: BPE Tokenization
Manual implementation of the BPE tokenization algorithm.
It includes two Python files: Assignment1.py and analysis.py. The first one is the tokenizer itself, and the second one is used to compare the performance of the tokenizer when using different length training texts and number of iterations.

To run the BPE tokenizer:

```
python Assignment1.py K TRAIN_FILE TEST_FILE
```

## Assignment 2: Sentence probabilities
This assignment is divided in two sub-assignments, both using the Brown corpus for training and Python's NLTK package.

**Assignment 2A** (Assignment2A.py) calculates the probability of a sentence entered by the user.
**Assignment 2B** (Assignment2B.py) asks the user for an input and returns the top 3 most likely words recursively until the user requests to stop.


## Assignment 3: Naive Bayes vs Logistic Regression text classifier
In this assignment the Kaggle Fake vs True news dataset (https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) is used to build two classifiers that allow to distinguish between True and Fake news. The first one is a manual implementation of a Naive Bayes classifier, and the second one is a Logistic Regression classifier. For the latter, BoW vectors had to be created manually without using any Python library. To run this code:

```
python Assignment3.py TRAIN_SIZE ALGO_TYPE
```

Where TRAIN_SIZE is a number between 50 and 80 that defines the size of the training set, and ALGO_TYPE is either 0 (Naive Bayes) or 1 (Logistic Regression).

