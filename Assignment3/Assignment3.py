import sys
import kagglehub
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# To avoid getting warnings from pandas
pd.options.mode.chained_assignment = None

if len(sys.argv) != 3: # If arguments are not 2, use default values
    train_size = 0.8
    algo = 0
else: # Else, get their values
    train_size = int(sys.argv[1])/100
    algo = int(sys.argv[2])

# If train size is not between 50-80, use default value
if train_size not in range(50,81):
    train_size = 0.8

# If algo is not 0 or 1, use default value
if algo not in [0,1]:
    algo = 0

print("Vila Rodriguez, Raquel, A20598805 solution:")
print(f"Training set size: {train_size}")
algo_names = ["Naive Bayes", "Logistic Regression"]
print(f"Classifier type: {algo_names[algo]}")

# Download fake and real news dataset from kaggle and create dataframes
path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
fake_file = rf"{path}\fake.csv"
df_fake = pd.read_csv(fake_file)
true_file = rf"{path}\true.csv"
df_true = pd.read_csv(true_file)

# Divide between train and test and add label to compare true value and prediction
df_true_train = df_true.iloc[:int(len(df_true)*train_size)]
df_true_test = df_true.iloc[int(len(df_true)*train_size):]
df_true_test.loc[:, "label"] = 0
df_fake_train = df_fake.iloc[:int(len(df_fake)*train_size)]
df_fake_test = df_fake.iloc[int(len(df_fake)*train_size):]
df_fake_test.loc[:, "label"] = 1

# Create stopwords set to remove from texts
stopwords_set = set(stopwords.words('english'))

if algo==0:
    # Initialize a set (avoid repetitions) for the vocabulary
    vocab = set()
if algo==1:
    # For logistic regression, get all words as we will need to count their frequency
    words_all = []

def preprocess(text):

    text = text.lower()     # convert text to lower-case
    text = re.sub("\\W"," ",text)   # remove special characters

    # Split text in words
    words = text.split()

    # Remove stop words and very short words 
    words = [word for word in words if word not in stopwords_set and len(word) > 2]
    
    # Add words to vocabulary
    if algo==0:
        vocab.update(words)
    if algo==1:
        words_all.extend(words)


    return words

print("Processing text and getting vocabulary...")
df_true_train.loc[:,'filtered words'] = df_true_train['text'].apply(lambda x: preprocess(x))
df_fake_train.loc[:,'filtered words'] = df_fake_train['text'].apply(lambda x: preprocess(x))
df_true_test.loc[:,'filtered words'] = df_true_test['text'].apply(lambda x: preprocess(x))
df_fake_test.loc[:,'filtered words'] = df_fake_test['text'].apply(lambda x: preprocess(x))

# Combine the DataFrames
combined_df = pd.concat([df_fake_test, df_true_test])
# Shuffle the combined DataFrame
df_test = combined_df.sample(frac=1)

def train_NB(df_true_train, df_fake_train):

    # Initialize dictionaries for the word counts in true and fake news (parameters of NB)
    dict_words_true = {}
    dict_words_fake = {}

    for idx, row in df_true_train.iterrows():

        # Using the filtered and processed words, create a dictionary with the count for each word
        words = row["filtered words"]
        for w in words:
            if w not in dict_words_true:
                dict_words_true[w] = 1
            else:
                dict_words_true[w] += 1
    
    for idx, row in df_fake_train.iterrows():

        # Using the filtered and processed words, create a dictionary with the count for each word
        words = row["filtered words"]
        for w in words:
            if w not in dict_words_fake:
                dict_words_fake[w] = 1
            else:
                dict_words_fake[w] += 1

    # Calculate sum(count(xi,y) for all V)
    n_true = sum(dict_words_true.values())
    n_fake = sum(dict_words_fake.values())

    # Add also the counts = 0 for the words not present in each of the datasets
    for word in vocab:
        if word not in dict_words_true.keys():
            dict_words_true[word] = 0
        if word not in dict_words_fake.keys():
            dict_words_fake[word] = 0

    # Transform from count to log probabilities log(xi/y) with add-1 smoothing
    probs_true = [np.log((x + 1)/(n_true+len(vocab))) for x in dict_words_true.values()]
    dict_words_true = dict(zip(dict_words_true.keys(), probs_true))
    probs_fake = [np.log((x + 1)/(n_fake+len(vocab))) for x in dict_words_fake.values()]
    dict_words_fake = dict(zip(dict_words_fake.keys(), probs_fake))

    # Calculate log(p(label=true)) and log(p(label=fake))
    prob_true = np.log(len(df_true_train)/(len(df_true_train)+len(df_fake_train)))
    prob_fake = np.log(len(df_fake_train)/(len(df_true_train)+len(df_fake_train))) 

    return dict_words_true, dict_words_fake, prob_true, prob_fake

def test_NB(dict_words_true, dict_words_fake, prob_true, prob_fake, words):           

    # Using the filtered and processed words, create a Counter with the count for each word
    word_counts = Counter(words)

    # Intialize the products log(p(xi/y))
    log_prob_true = prob_true
    log_prob_fake = prob_fake

    # For each word in the document that appears k times, do p(xi/y)^k --> k*log(p(xi/y))
    for word in word_counts.keys():
        # Ignore OOV words
        if word in dict_words_true:
            log_prob_true += word_counts[word] * dict_words_true[word]
        if word in dict_words_fake:
            log_prob_fake += word_counts[word] * dict_words_fake[word]

    # Compare probabilities and assign label
    predicted_label = 0 if log_prob_true > log_prob_fake else 1

    #Normalize probabilities so they sum 1 before exponentiating
    log_probs = np.array([log_prob_true, log_prob_fake])
    max_logP = np.max(log_probs)
    exp_shifted = np.exp(log_probs - max_logP)
    normalized_probs = exp_shifted / np.sum(exp_shifted)

    return predicted_label, log_prob_true, log_prob_fake, normalized_probs[1]

def create_bow_vector(words, vocabulary):
    # Create a vector with all 1s (add-1 smoothing)
    vector = [1] * len(vocabulary)
    # For each word in the text, if the word is in the vocabulary
    # get the position of the word in the BoW vector (value of word in 
    # vocab dictionary) and add 1
    for word in words:
        if word in vocabulary:
            idx = vocabulary[word]
            vector[vocabulary[word]] += 1
    return vector

def compute_and_print_metrics(predictions, actual_labels):
    
    # Get True Positive, False Positive, False Negtive, True Positive values
    tn, fp, fn, tp = confusion_matrix(actual_labels, predictions).ravel()

    # Get metrics
    sensitivity = recall_score(actual_labels, predictions) 
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
    precision = precision_score(actual_labels, predictions)  
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0  
    accuracy = accuracy_score(actual_labels, predictions)  
    f_score = f1_score(actual_labels, predictions)  

    # Print results
    print(f"Number of true positives: {tp}")
    print(f"Number of true negatives: {tn}")
    print(f"Number of false positives: {fp}")
    print(f"Number of false negatives: {fn}")
    print(f"Sensitivity (recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Negative predictive value: {negative_predictive_value:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F-score: {f_score:.4f}")


if algo==0:
    print("Training Naive Bayes classifier...")
    # Use train_NB with the true and fake news dataframes and obtain the parameters
    dict_words_true, dict_words_fake, prob_true, prob_fake = train_NB(df_true_train, df_fake_train)
    predictions = []
    print("Testing Naive Bayes classifier...")
    # Compute the prediction for each row with test_NB and the obtained parameters
    for idx, row in df_test.iterrows():
        words = row["filtered words"] 
        predicted_label, log_prob_true, log_prob_fake, score = test_NB(dict_words_true, dict_words_fake, prob_true, prob_fake, words)
        predictions.append(predicted_label)

    # Get the true values and obtain and print the metrics
    actual_labels = df_test["label"].values

    compute_and_print_metrics(predictions, actual_labels)

if algo==1:

    # For Logistic Regression, as we will have to create BoW vectors, reduce the vocabulary size
    # to avoid getting a MemoryError. This is done by removing words that do not appear more than a number of times
    word_counts = Counter(words_all)
    min_frequency = 50
    # Filter the vocab to keep only words whose count is > min_frequency, creating a dictionary with the vocab and count
    vocab = {word: count for word, count in word_counts.items() if count >= min_frequency}
    # Change the dictionary so that the keys will be the words and the values will be their index in the BoW vector 
    vocab = {word: idx for idx, (word, count) in enumerate(vocab.items())}

    # Create BoW vectors for both true and false datasets
    print("Creating BoW vectors")
    df_true_train.loc[:, "bow vector"] = df_true_train['filtered words'].apply(lambda x: create_bow_vector(x, vocab))
    df_true_train.loc[:, 'label'] = 0
    df_fake_train.loc[:, "bow vector"] = df_fake_train['filtered words'].apply(lambda x: create_bow_vector(x, vocab))
    df_fake_train.loc[:, 'label'] = 1

    combined_df_train = pd.concat([df_true_train, df_fake_train])
    df_train = combined_df_train.sample(frac=1)

    df_test.loc[:,"bow vector"] = df_test['filtered words'].apply(lambda x: create_bow_vector(x, vocab))

    # Create matrix with BoW vectors
    X_train = np.vstack(df_train["bow vector"])
    y_train = df_train["label"].values

    # Create matrix with BoW vectors
    X_test = np.vstack(df_test["bow vector"])
    y_test = df_test["label"].values

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(random_state=0, max_iter=15000).fit(X_train, y_train)
    print("Testing Logistic Regression classifier...")
    predictions = clf.predict(X_test)

    compute_and_print_metrics(predictions, y_test)

label_name = ["True News", "Fake News"]

# Keep asking for more sentences until yes is true (user answers "N")
while True:
    sentence = input("Enter your sentence/document: \n")
    print(f"Sentence/document S: {sentence}")

    # Apply same pre-processing as with dataframes' texts
    sentence = sentence.lower()  
    sentence = re.sub("\\W"," ",sentence)   
    words = sentence.split()
    words = [word for word in words if word not in stopwords_set and len(word) > 2]

    # For Naive Bayes, obtain predicted class and probabilities with train_NB and previous parameters and print
    if algo == 0:
        predicted_label, log_prob_true, log_prob_fake, score = test_NB(dict_words_true, dict_words_fake, prob_true, prob_fake, words)
        label_name = ["True News", "Fake News"]

        print(f"was classified as {label_name[predicted_label]}")
        print(f"P(True News/S) = {1-score}")
        print(f"P(Fake News/S) = {score}")

    # For Logistic regression, create BoW vector for sentence and predict probability of each label using
    # pre-built method. Assign label and print.
    if algo == 1:
        bow_s = create_bow_vector(words, vocab)
        predicted_label = clf.predict(np.array(bow_s).reshape(1,-1))

        print(f"was classified as {label_name[predicted_label[0]]}")
    
    
    # Ask user and repeat question until they answer Y or N
    answer = input("Do you want to enter another sentence [Y/N]? ")
    while answer:
        if answer == "Y":
            answer = None
        elif answer == "N":
            sys.exit()
        else:
            answer = input("Answer Y or N ")


