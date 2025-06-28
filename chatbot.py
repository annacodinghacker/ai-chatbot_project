import random
import json
import nltk
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Prepare training data
corpus = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        corpus.append(pattern)
        tags.append(intent['tag'])

# Vectorize text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus).toarray()

# Train model
clf = MultinomialNB()
clf.fit(X, tags)

# Predict function
def predict_class(user_input):
    user_input = vectorizer.transform([user_input]).toarray()
    return clf.predict(user_input)[0]

# Get chatbot response
def get_response(intent_tag):
    for intent in data["intents"]:
        if intent["tag"] == intent_tag:
            return random.choice(intent["responses"])
    return "Sorry, I don't understand."

# Chat loop
if __name__ == "__main__":
    print("Chatbot is ready! (type 'quit' to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        intent = predict_class(user_input)
        print("Bot:", get_response(intent))
