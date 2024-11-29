import numpy as np
import pandas as pd
import math
import pickle 
from tfidf import TF_IDF

class ComplementNB:
    def __init__(self, stopwords, features_limit=10000):
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

    def train_complement_naive_bayes(self, train_x, train_y):
        self.word_counts = {key: {word: 1 for word in self.features} for key in self.word_counts}

        for i in range(len(train_x)):
            sentence = str(train_x.iloc[i])
            if not sentence or sentence.strip() == "":
                continue
        
        for i in range(len(train_x)):
            sentiment = train_y.iloc[i]
            if sentiment == 'positive':
                self.tot_counts[2] += 1
            elif sentiment == 'neutral':
                self.tot_counts[1] += 1
            elif sentiment == 'negative':
                self.tot_counts[0] += 1

            for word in train_x.iloc[i].split():
                if word in self.features:
                    self.word_counts[sentiment][word] += 1

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

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def run(self, train_data, test_data):
        self.get_features(train_data['text'].to_numpy())
        self.train_complement_naive_bayes(train_data['text'], train_data['sentiment'])
        self.evaluate(test_data['text'], test_data['sentiment'])

def main():
    stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    df_train1 = pd.read_csv("train.csv", encoding='ISO-8859-1')[["text", "sentiment"]]

    df_train_combined = pd.concat([df_train1], ignore_index=True)
    df_train_combined = df_train_combined.sample(frac=1, random_state=42).reset_index(drop=True)

    df_test = pd.read_csv("test.csv", encoding='ISO-8859-1')[["text", "sentiment"]]

    df_train_combined['text'] = df_train_combined['text'].fillna("").astype(str)
    df_test = df_test.dropna(subset=['sentiment'])
    df_test['text'] = df_test['text'].fillna("").astype(str)
    
    model = ComplementNB(stopwords)
    model.run(df_train_combined, df_test)

    # Save the trained model
    model.save_model("complement_nb_model.pkl")
    
    # model = ComplementNB.load_model("complement_nb_model.pkl")

    while True:
        inp = input("\nsentence: ")
        print(model.sentiment[model.predict(inp)])

if __name__ == "__main__":
    main()
