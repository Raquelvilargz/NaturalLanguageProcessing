import sys, os.path, string, time, re, json

num = [50, 100, 150, 200]
train_file = ["500_train.txt", "1000_train.txt", "1500_train.txt", "2000_train.txt"]
test_file = ["500_test.txt", "500_test.txt", "500_test.txt", "500_test.txt"]

def process_and_split_text(t):
    text = t.translate(str.maketrans('', '', string.punctuation))
    processed_text = re.sub("[^a-zA-Z\s]", "", text)
    split_text = processed_text.split()
    split_text = [word + "_" for word in split_text]
    return split_text

def split_by_vocab(w, V):
    V_sorted = sorted(V, key=len, reverse=True)
    pattern = "|".join(re.escape(v) for v in V_sorted)
    splits = re.findall(pattern, w)
    return splits

training = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
testing = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
text_lengs = [500, 1000, 1500, 2000]
a = 0

for k in num:
    for n in range(len(train_file)):

        #print("Training file name: " + train_file[n])
        #print("Test file name: " + test_file[n])
        print("Number of merges: " + str(k) + ", text length: " + str(text_lengs[n]))

        with open(train_file[n], 'r') as file:
            train_text = file.read()
        
        times_train = 0
        times_test = 0

        for rep in range(0,5):
            vocab = list(string.ascii_lowercase) + list(string.ascii_uppercase) +  [" ", "_"]
            training_start_time = time.time()
            split_train_text = process_and_split_text(train_text)
            
            for i in range(1,k+1):
                pairs = {}
                # Count occurences of each bigram and saves in dictionary pairs
                for w in split_train_text:
                    c = split_by_vocab(w,vocab)
                    for j in range(len(c)-1):
                        if c[j]+c[j+1] in pairs:
                            pairs[c[j]+c[j+1]] += 1
                        else:
                            pairs[c[j]+c[j+1]] = 1        
                # Get most frequent pair
                most_freq = max(pairs, key=pairs.get)
                #print(most_freq)
                vocab.append(most_freq)

            #print(vocab[55:65])
            training_end_time = time.time()
            training_time = training_end_time-training_start_time

            #print("Training time: " + str(training_time) + " seconds")

            times_train += training_time

            with open(test_file[n], 'r') as file:
                test_text = file.read()

            tokenization_start_time = time.time()
            split_test_text = process_and_split_text(test_text)
            result = list()

            for w in split_test_text:
                tokenized_w = split_by_vocab(w, vocab)
                result.extend(tokenized_w)

            tokenization_end_time = time.time()
            tokenization_time = tokenization_end_time-tokenization_start_time

            times_test += tokenization_time

            #print("Tokenization time: " + str(tokenization_time) + " seconds")
        
        training[a][n] = times_train/5
        #print(training)
        testing[a][n] = times_test/5
        #print(testing)
        print(" ".join(result[0:20]))

    a += 1


print(training)
print(testing)

with open('training_times.json', 'w') as file:
    json.dump(training, file)

with open('tokenizing_times.json', 'w') as file:
    json.dump(testing, file)

