import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from CountVectorizer import CountVectorizer
from MultinomialNB import MultinomialNaiveBayes

dataset=pd.read_csv('train.csv',encoding='latin1')

dataset=dataset.dropna()

X=dataset['selected_text']
y=dataset['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


nb_classifier=MultinomialNaiveBayes()
nb_classifier.fit(X_train_vectorized,y_train)

y_pred=nb_classifier.predict(X_test_vectorized)
accuracy=accuracy_score(y_test,y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


def predict_sentiment(sentence):
    sentence_vectorized = vectorizer.transform([sentence])
    prediction = nb_classifier.predict(sentence_vectorized) #M
    return prediction[0]