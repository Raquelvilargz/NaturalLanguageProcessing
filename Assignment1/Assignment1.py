import sys, os.path, string, time, re

if len(sys.argv) != 4:
    raise ValueError("Input 3 arguments")

num = sys.argv[1]
train_file = sys.argv[2]
test_file = sys.argv[3]

if not os.path.isfile(train_file):
    raise OSError("Train file not found")

if not os.path.isfile(test_file):
    raise OSError("Test file not found")

try:
    k = int(num)
    if k <= 0:
        k = 5
except ValueError:
    k=5

print("Vila Rodriguez, Raquel, A20598805 solution:")
print("Number of merges: " + str(k))
print("Training file name: " + train_file)
print("Test file name: " + test_file)
print()

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

with open(train_file, 'r') as file:
    train_text = file.read()

training_start_time = time.time()
split_train_text = process_and_split_text(train_text)
vocab = list(string.ascii_lowercase) + list(string.ascii_uppercase) +  ["_"]

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
    vocab.append(most_freq)

training_end_time = time.time()

print("Training time: " + str(training_end_time-training_start_time) + " seconds")

with open(test_file, 'r') as file:
    test_text = file.read()

tokenization_start_time = time.time()
split_test_text = process_and_split_text(test_text)
result = list()

for w in split_test_text:
    tokenized_w = split_by_vocab(w, vocab)
    result.extend(tokenized_w)

tokenization_end_time = time.time()

print("Tokenization time: " + str(tokenization_end_time-tokenization_start_time) + " seconds")
print()

result_text = " ".join(result)
result_vocab = "\n".join(vocab)

if len(result) > 20:
    result_first_twenty = result[0:20]
    result_text_first_twenty = " ".join(result_first_twenty)
    print("Tokenization result: " + result_text_first_twenty)
    print("Tokenized text is longer than 20 tokens")
else:
    print("Tokenization result: " + result_text)

with open("CS585_P01_A20598805_VOCAB.txt", "w") as file:
    file.write(result_vocab)

with open("CS585_P01_A20598805_RESULT.txt", "w") as file:
    file.write(result_text)



