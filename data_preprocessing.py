import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem.snowball import SnowballStemmer


class Tokenizer:
    '''
    Tokenizing of documents
    The class is responsible for data preprocessing to prepare data to further analysis and application of the
    algorithms
    '''
    def __init__(self, documents):
        '''
        Initializing variables
        :param documents: list of documents (descriptions of a product)
        '''
        self.documents = documents
        self.texts = []

    def _omit_punctuation(self):
        '''
        Omitting the punctuation like dots, commas, brackets to avoid their role in data analysis
        :return:
        '''
        punctuation = '.,/()[]{}-+=*&^%$#@!~:;"\''
        for sentence in self.documents:
            sentence_change = sentence
            for symbol in sentence:
                if symbol in punctuation:
                    sentence_change = sentence_change.replace(symbol, " ")

            self.texts.append(sentence_change)

    def _omit_stopwords(self):
        '''
        Omitting stop words like the, a, this, etc, to make sure they have no effect on the document's main topic
         and hence do not influence the process of similar products search
        :return:
        '''
        stop_words = set(stopwords.words('english'))

        self.texts = [
            [word for word in document.lower().split() if word not in stop_words]
            for document in self.texts
        ]

    def _lemmatize(self):
        '''
        Using NLP algorithm of lemmatization (stemming) we make the module understand, that the words like
        run, running, ran are the same word
        With stemming we reduce the word to the root form (stem) even if the stem is not a word itself
        The suffixes and prefixes are removed used with word
        :return:
        '''
        englishStemmer = SnowballStemmer("english")
        self.texts = [
            [englishStemmer.stem(word) for word in document]
            for document in self.texts
        ]

    def _omit_frequency(self):
        '''
        We assume that a word which repeats once in a whole document set, it carries no valuable information and for
        that reason we avoid words that occur once
        :return:
        '''
        if len(self.documents) > 1:
            frequency = defaultdict(int)
            for text in self.texts:
                for token in text:
                    frequency[token] += 1

            self.texts = [
                [token for token in text if frequency[token] > 1]
                for text in self.texts
            ]

    def add_document(self, document):
        '''
        Just a method to add a document to a set of documents, but we have to remember to rerun the process of
        tokenization
        :param document:
        :return:
        '''
        self.documents.append(document)

    def tokenize(self):
        '''
        Tokenization process
        Here we just call the methods written above as the entire algorithm of data preprocessing
        :return: list of tokens for each document
        '''
        #call upper functions
        self._omit_punctuation()
        self._omit_stopwords()
        self._omit_frequency()
        self._lemmatize()

        return self.texts

