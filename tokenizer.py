import string
import re

class Tokenizer:
    def __init__(self, text, stopwords):
        self.text = text
        self.stopwords = stopwords

    def word_tokenizer(self):

        clean_text = ''.join([char if char not in string.punctuation else ' ' for char in self.text])
        tokens = clean_text.split()
        filtered_tokens = [word for word in tokens if word.lower() not in self.stopwords]
        return filtered_tokens

    def sentence_tokenizer(self):

        sentences = []
        sentence = ""
        abbreviations = {"mr", "mrs", "dr", "etc", "e.g", "i.e", "vs", "st", "jr"}  
        end_punctuations = {".", "!", "?"}

        i = 0
        while i < len(self.text):
            char = self.text[i]
            sentence += char
            if char in end_punctuations:
                next_char = self.text[i + 1] if i + 1 < len(self.text) else ""
                prev_char = self.text[i - 1] if i > 0 else ""

                if (i + 1 < len(self.text) and next_char == ".") or \
                   (prev_char.isalpha() and self.text[i + 1:i + 3].lower() in abbreviations):
                    pass
                else:
                    sentences.append(sentence.strip())
                    sentence = ""  
            i += 1

        if sentence.strip():
            sentences.append(sentence.strip())

        return sentences
