import nltk
from nltk import ConditionalFreqDist

nltk.download('brown')

#First we have to get the probability of each bigram in the Brown Corpus.
b_words = nltk.corpus.brown.words()
lwb_words = [word.lower() for word in b_words]

bigrams = nltk.ngrams(lwb_words, 2)

# We calculate first the count of each bigram (with ConditionalFreqDist such that we get a collection of frequency counts for each first word of the bigram)
bigrams_condfreq = ConditionalFreqDist(bigrams)

# Then we obtain a dictionary of dictionaries where the first key will be the first word of the bigram and the second key the second word, and the value will
# be count(wn+1,wn)/count(wn)

bigrams_prob = {}
for w1 in bigrams_condfreq:
    bigrams_prob[w1] = {}
    count_w1 = bigrams_condfreq[w1].N()
    for w2 in bigrams_condfreq[w1]:
        bigrams_prob[w1][w2] = bigrams_condfreq[w1][w2]/count_w1

#Next we ask the user to enter a sentence and lowercase it
sentence = input("Enter a sentence: ")
lw_sentence = sentence.lower()

#And then we will calculate the probability of this sentence by multiplying the bigram probabilities (and assuming P(w/<s>) = P(<s>/w) = 0.25)
bigrams_s = nltk.ngrams(lw_sentence.split(), 2)
prob_s = 0.25*0.25

print("Bigrams probability: ")
print(" <s>, " + lw_sentence.split()[0] + ": 0.25")

#Print probability of each bigram and calculate total probability of the sentence
for b in bigrams_s:
    try:
        prob_s = prob_s * bigrams_prob[b[0]][b[1]]
        print(b[0] + " , " + b[1] + " : " + str(bigrams_prob[b[0]][b[1]]))
    except KeyError:
        print(b[0] + " , " + b[1] + " : " + "0")
        prob_s = 0
        continue

print(" <s>, " + lw_sentence.split()[-1] + ": 0.25")

print("Probability of the sentence is: " + str(prob_s))


