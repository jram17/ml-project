import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_and_tokenize(sentence):

    tokens = word_tokenize(sentence.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

def TF_IDF(sentences, num_sentences):
    words = set()
    processed_sentences = []

    for sentence in sentences:
        filtered_words = clean_and_tokenize(sentence)
        words.update(filtered_words)
        processed_sentences.append(filtered_words)

    words = sorted(words)
    word_index = {word: i for i, word in enumerate(words)}

    rows, cols, data = [], [], []
    for i, sentence in enumerate(processed_sentences):
        total_words = len(sentence)
        for word in sentence:
            rows.append(i)
            cols.append(word_index[word])
            data.append(1 / (total_words + 1)) 

    tf_matrix = csr_matrix((data, (rows, cols)), shape=(len(sentences), len(words)))

    word_doc_count = np.bincount(tf_matrix.indices, minlength=tf_matrix.shape[1])
    idf = np.log(1 + len(sentences) / (1 + word_doc_count))

    tfidf_matrix = tf_matrix.multiply(idf)

    similarity_matrix = cosine_similarity(tfidf_matrix)

    scores = np.ones(len(sentences)) / len(sentences)
    damping_factor = 0.85
    for _ in range(100):
        new_scores = (1 - damping_factor) / len(sentences) + damping_factor * similarity_matrix.T.dot(scores)
        if np.allclose(scores, new_scores, atol=1e-4):  
            break
        scores = new_scores

    ranked_sentences = sorted(zip(scores, sentences), key=lambda x: x[0], reverse=True)
    return [sentence for _, sentence in ranked_sentences[:num_sentences]]
