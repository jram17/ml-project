import numpy as np
import pandas as pd
import math
from tfidf import TF_IDF

stopwords=['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

df = pd.read_csv("train.csv", encoding='ISO-8859-1')
df = df[["text", "sentiment"]]

df1 = pd.read_csv("train1.csv", encoding='ISO-8859-1')
df1 = df1[["text", "sentiment"]]

df_combined = pd.concat([df, df1], ignore_index=True)
df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)

df2 = pd.read_csv("test.csv", encoding='ISO-8859-1')
df2 = df2[["text", "sentiment"]]

df_combined['text'] = df_combined['text'].fillna("").astype(str)
df2 = df2.dropna(subset=['sentiment'])
df2['text'] = df2['text'].fillna("").astype(str)

positive = df_combined[df_combined['sentiment'] == 'positive']['text'].to_numpy()
neutral = df_combined[df_combined['sentiment'] == 'neutral']['text'].to_numpy()
negative = df_combined[df_combined['sentiment'] == 'negative']['text'].to_numpy()

sentiment = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}

X = df_combined['text']
Y = df_combined['sentiment']

train_x = X
train_y = Y

test_x = df2["text"]
test_y = df2["sentiment"]

features = []

def get_features(list):
    for sentence in list:
        for word in sentence.split():
            if word not in features and word not in stopwords:
                features.append(word)

positive_sentences = TF_IDF(positive, 10000)
neutral_sentences = TF_IDF(neutral, 10000)
negative_sentences = TF_IDF(negative, 10000)

get_features(positive_sentences)
get_features(neutral_sentences)
get_features(negative_sentences)

word_counts = {
    'positive': {word: 1 for word in features},
    'neutral': {word: 1 for word in features},
    'negative': {word: 1 for word in features}
}

tot_counts = [1,1,1]
total_count = len(train_x) + 3

def train_complement_naive_bias():
    for i in range(0,len(train_x)):
        if train_y[i] == 'positive':
            tot_counts[2] += 1
            sentiment = 'positive'
        elif train_y[i] == 'neutral':
            tot_counts[1] += 1
            sentiment = 'neutral'
        elif train_y[i] == 'negative':
            tot_counts[0] += 1
            sentiment = 'negative'

        for word in train_x.iloc[i].split():
            if word in features:
                word_counts[sentiment][word] += 1

train_complement_naive_bias()

probs = [1,1,1]
for i in range(0,3):
    probs[i] = tot_counts[i]/total_count

def complement_prob(word, c):
    sum = 1
    total = 1 + (1*len(train_x))
    for i in range(1,3):
        sum += word_counts[sentiment[(c+i)%3]][word]
        total += tot_counts[(c+i)%3]
    return sum/total

def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = ''.join([char for char in sentence if char.isalpha() or char.isspace()])
    cleaned_words = [word for word in sentence.split() if word not in stopwords]
    return ' '.join(cleaned_words)

def pred_complement_naive_bias(sent):
    total_prob = 0
    out = 0
    sent = str(sent)
    sentence = clean_sentence(sent)
    for i in range(0,3):
        vec = []
        sum = 1
        for word in sentence.split():
            if word in features:
                vec.append(word)
                prob = complement_prob(word,i)
                sum += math.log((1-prob)/prob)
        sum += probs[i]
        if sum > total_prob:
            total_prob = sum
            out = i
    return out

true_positives = [0, 0, 0]
false_positives = [0, 0, 0]
false_negatives = [0, 0, 0]
true_negatives = [0, 0, 0]

for i, sentence in enumerate(test_x):
    predicted_class = pred_complement_naive_bias(sentence)
    actual_class = {v: k for k, v in sentiment.items()}[test_y.iloc[i]]

    for c in range(3):
        if c == predicted_class and c == actual_class:
            true_positives[c] += 1
        elif c == predicted_class and c != actual_class:
            false_positives[c] += 1
        elif c != predicted_class and c == actual_class:
            false_negatives[c] += 1
        else:
            true_negatives[c] += 1

precision = [0, 0, 0]
recall = [0, 0, 0]
f1_score = [0, 0, 0]
accuracy = sum(true_positives) / len(test_y)

for c in range(3):
    if true_positives[c] + false_positives[c] > 0:
        precision[c] = true_positives[c] / (true_positives[c] + false_positives[c])
    if true_positives[c] + false_negatives[c] > 0:
        recall[c] = true_positives[c] / (true_positives[c] + false_negatives[c])
    if precision[c] + recall[c] > 0:
        f1_score[c] = 2 * (precision[c] * recall[c]) / (precision[c] + recall[c])

for c in range(3):
    print(f"Class {sentiment[c]}:")
    print(f"  Precision: {precision[c]:.4f}")
    print(f"  Recall: {recall[c]:.4f}")
    print(f"  F1-Score: {f1_score[c]:.4f}")
print(f"Overall Accuracy: {accuracy:.4f}")

while True:
    inp = input("\nsentence: ")
    print(sentiment[pred_complement_naive_bias(inp)])