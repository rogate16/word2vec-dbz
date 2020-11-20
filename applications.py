import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import warnings
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec

model = Word2Vec.load("data_input/model")

state = True

print("What do you want to know?")
print("1. Similarity Score")
print("2. Most Similar Words")
print("3. One-Odd Out")
print("Enter q to quit")

while state:
    try:
        choice = input("Your Choice : ")
        if choice=="q":
            print("Closing program...")
            state=False
        elif int(choice)==1:
            a = input("First word : ")
            b = input("Second word : ")
            print("Similarities :",model.wv.similarity(a.lower(), b.lower()))
            print()
        elif int(choice)==2:
            a = input("Enter a word : ")
            print("List of most similar words")
            print(model.wv.most_similar(positive=[a.lower()]))
            print()
        elif int(choice)==3:
            a = input("First word : ")
            b = input("Second word : ")
            c = input("Third word : ")
            print(model.wv.doesnt_match((a.lower(), b.lower(), c.lower())),"doesn't belong to the list")
            print()
        else:
            print("Sorry, you might be entering the wrong number")
    except KeyError:
        print("Oops! Words not found in vocabulary. Please try another word.")
        print()
    except Exception as e:
        print("Unknown error occurred. Please try again.")
        print()