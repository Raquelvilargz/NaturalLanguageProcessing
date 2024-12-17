import nltk
from nltk import ConditionalFreqDist
from nltk.corpus import stopwords

#nltk.download('brown')
#nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

#First we have to get the probability of each bigram in the Brown Corpus.
b_words = nltk.corpus.brown.words()

filtered_words = [word for word in b_words if word.lower() not in stop_words]
lwb_words = [word.lower() for word in filtered_words]

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
c = True
w = input("Enter a word: ")

#Repeat until user says stop
while c:
    print(f"{w} ...")
    lw_w = w.lower()

    #If the word is not in the corpus we can either QUIT or ask again
    if w not in bigrams_prob.keys():
        print("Word not in corpus")
        answer = input("Quit? y/n: ")
        if answer == "y":
            print("Exiting program")
            exit()
        w = input("Enter new word: ")  
        continue

    #Find possible next words after the chosen word
    next_words = bigrams_prob[w]

    #Sort by probability to get top 3 most possible words
    sorted_next_words = sorted(next_words.items(), key=lambda x: x[1], reverse=True)

    #Print top 3 most likely words
    print("Which word should follow: " )
    for i, (next_word, prob) in enumerate(sorted_next_words[:3], start=1):
        print(f"{i}) {next_word} P({w} {next_word})= {prob}")
    print("4) QUIT")

    number = input("")

    #If the number is not in the range assume it is 1
    while number not in ["1", "2", "3", "4"]:
        print("Number not in range, using 1")
        number = "1"
    
    if number == "4":
        c = False
        continue
    
    #Get next word from chosen number and repeat again
    w = sorted_next_words[int(number)-1][0]






