class CountVectorizer:
    def __init__(self):
        self.vocabulary = {}
        self.vocab_list = []

    def fit(self, data):
        """
        Builds the vocabulary for the input text data.
        """
        unique_words = set()
        for sentence in data:
            for word in sentence.split():
                if len(word) > 2:  # Only consider words of length > 2
                    unique_words.add(word.lower())

        self.vocab_list = sorted(unique_words)
        self.vocabulary = {word: index for index, word in enumerate(self.vocab_list)}
        return self

    def transform(self, data):
        """
        Transforms input text data into a document-term matrix.
        """
        if not self.vocabulary:
            raise ValueError("The vocabulary is empty. Call fit() before transform().")

        # Create a matrix of zeros
        matrix = [[0] * len(self.vocabulary) for _ in range(len(data))]

        for i, sentence in enumerate(data):
            words = sentence.split()
            for word in words:
                if len(word) > 2:
                    col_index = self.vocabulary.get(word.lower())
                    if col_index is not None:
                        matrix[i][col_index] += 1

        return matrix

    def fit_transform(self, data):
        """
        Combines fit and transform to create the vocabulary and generate the matrix.
        """
        self.fit(data)
        return self.transform(data)