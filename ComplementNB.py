import numpy as np
import pandas as pd
import math
from tfidf import TF_IDF

class ComplementNB:
    def __init__(self, stopwords, features_limit=100000):
        self.stopwords = stopwords
        self.features_limit = features_limit
        self.sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self.features = []
        self.word_counts = {
            'positive': {},
            'neutral': {},
            'negative': {}
        }
        self.tot_counts = [1, 1, 1]
        self.probs = [1, 1, 1] 

    def get_features(self, list_of_sentences):
        for sentence in list_of_sentences:
            sentence = str(sentence) 
            for word in sentence.split():
                if word not in self.features and word not in self.stopwords:
                    self.features.append(word)

    def train_complement_naive_bayes(self, train_x, train_y, fragments=10):
        fragment_size = len(train_x) // fragments
        accumulated_counts = {
            'positive': {word: 1 for word in self.features},
            'neutral': {word: 1 for word in self.features},
            'negative': {word: 1 for word in self.features}
        }
        accumulated_tot_counts = [1, 1, 1]

        for f in range(fragments):
            start = f * fragment_size
            end = (f + 1) * fragment_size if f < fragments - 1 else len(train_x)
            
            fragment_x = train_x[start:end]
            fragment_y = train_y[start:end]

            fragment_word_counts = {
                'positive': {word: 1 for word in self.features},
                'neutral': {word: 1 for word in self.features},
                'negative': {word: 1 for word in self.features}
            }
            fragment_tot_counts = [1, 1, 1]

            for i in range(len(fragment_x)):
                sentiment = fragment_y.iloc[i]
                if sentiment == 'positive':
                    fragment_tot_counts[2] += 1
                elif sentiment == 'neutral':
                    fragment_tot_counts[1] += 1
                elif sentiment == 'negative':
                    fragment_tot_counts[0] += 1

                for word in fragment_x.iloc[i].split():
                    if word in self.features:
                        fragment_word_counts[sentiment][word] += 1

            for sentiment_label in ['positive', 'neutral', 'negative']:
                for word in self.features:
                    accumulated_counts[sentiment_label][word] += fragment_word_counts[sentiment_label][word]
            for c in range(3):
                accumulated_tot_counts[c] += fragment_tot_counts[c]

        for sentiment_label in ['positive', 'neutral', 'negative']:
            for word in self.features:
                self.word_counts[sentiment_label][word] = accumulated_counts[sentiment_label][word] / fragments
        for c in range(3):
            self.tot_counts[c] = accumulated_tot_counts[c] / fragments

        total_count = len(train_x) + 3 
        for i in range(3):
            self.probs[i] = self.tot_counts[i] / total_count

    def complement_prob(self, word, c):
        total = 1 + len(self.tot_counts) * len(self.features)
        word_sum = 1 

        for i in range(1, 3):
            word_sum += self.word_counts[self.sentiment[(c + i) % 3]][word]
            total += self.tot_counts[(c + i) % 3]

        return word_sum / total

    def clean_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = ''.join([char for char in sentence if char.isalpha() or char.isspace()])
        cleaned_words = [word for word in sentence.split() if word not in self.stopwords]
        return ' '.join(cleaned_words)

    def predict(self, sentence):
        total_prob = 0
        predicted_class = 0
        sentence = str(sentence)
        cleaned_sentence = self.clean_sentence(sentence)

        for i in range(3):
            prob_sum = 1
            for word in cleaned_sentence.split():
                if word in self.features:
                    prob_sum += math.log((1 - self.complement_prob(word, i)) / self.complement_prob(word, i))
            prob_sum += self.probs[i]
            if prob_sum > total_prob:
                total_prob = prob_sum
                predicted_class = i

        return predicted_class

    def evaluate(self, test_x, test_y):
        true_positives = [0, 0, 0]
        false_positives = [0, 0, 0]
        false_negatives = [0, 0, 0]
        true_negatives = [0, 0, 0]

        for i, sentence in enumerate(test_x):
            predicted_class = self.predict(sentence)
            actual_class = {v: k for k, v in self.sentiment.items()}[test_y.iloc[i]]

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
            print(f"Class {self.sentiment[c]}:")
            print(f"  Precision: {precision[c]:.4f}")
            print(f"  Recall: {recall[c]:.4f}")
            print(f"  F1-Score: {f1_score[c]:.4f}")
        print(f"Overall Accuracy: {accuracy:.4f}")

    def run(self, train_data, test_data):
        positive = TF_IDF(train_data[train_data['sentiment'] == 'positive']['text'], self.features_limit)
        neutral = TF_IDF(train_data[train_data['sentiment'] == 'neutral']['text'], self.features_limit)
        negative = TF_IDF(train_data[train_data['sentiment'] == 'negative']['text'], self.features_limit)
        self.get_features(positive)
        self.get_features(neutral)
        self.get_features(negative)
        self.train_complement_naive_bayes(train_data['text'], train_data['sentiment'])
        
        self.evaluate(test_data['text'], test_data['sentiment'])

def main():
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    # df1 = pd.read_csv("datasets/dataset1.csv", encoding='ISO-8859-1')
    # df1 = df1[["text", "sentiment"]]

    df2 = pd.read_csv("datasets/dataset2.csv", encoding='ISO-8859-1')
    df2 = df2[["text", "sentiment"]]

    df3 = pd.read_csv("datasets/dataset3.csv", encoding='ISO-8859-1')
    df3 = df3[["text", "sentiment"]]

    df4 = pd.read_csv("datasets/dataset4.csv", encoding='ISO-8859-1')
    df4 = df4[["text", "sentiment"]]

    df5 = pd.read_csv("datasets/dataset5.csv", encoding='ISO-8859-1')
    df5 = df5[["text", "sentiment"]]

    df_train_combined = pd.concat([df2, df3, df4, df5], ignore_index=True)
    df_train_combined = df_train_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    df_test = pd.read_csv("test.csv", encoding='ISO-8859-1')[["text", "sentiment"]]

    df_train_combined['text'] = df_train_combined['text'].fillna("").astype(str)
    df_train_combined['sentiment'] = df_train_combined['sentiment'].str.lower()
    df_train_combined = df_train_combined.dropna(subset=['sentiment'])
    df_test = df_test.dropna(subset=['sentiment'])
    df_test['text'] = df_test['text'].fillna("").astype(str)
    
    model = ComplementNB(stopwords)
    model.run(df_train_combined, df_test)

    while True:
        inp = input("\nsentence: ")
        print(model.sentiment[model.predict(inp)])

if __name__ == "__main__":
    main()