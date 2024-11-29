from stopwords import stopwords
class CountVectorizer:
    def __init__(self):
        self.vocabulary = {}
        self.vocab_list = []

    def fit(self, data):
        uniqueWords = set()
        for sentence in data:
            for word in sentence.split():
                if len(word) > 2 and word not in stopwords: 
                    uniqueWords.add(word.lower())

        self.vocab_list = sorted(uniqueWords)
        self.vocabulary = {word: index for index, word in enumerate(self.vocab_list)}
        return self

    def transform(self, data):

        if not self.vocabulary:
            raise ValueError("The vocabulary is empty. Call fit() before transform().")

        matrix = [[0] * len(self.vocabulary) for _ in range(len(data))]

        for i, sentence in enumerate(data):
            words = sentence.split()
            for word in words:
                if len(word) > 2 and word not in stopwords:
                    col_index = self.vocabulary.get(word.lower())
                    if col_index is not None:
                        matrix[i][col_index] += 1
        return matrix

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)



