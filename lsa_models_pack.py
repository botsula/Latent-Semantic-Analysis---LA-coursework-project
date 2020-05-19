import math

from gensim import corpora
from gensim import models, similarities
from data_preprocessing import Tokenizer
import numpy as np
import matplotlib.pyplot as plt
import random
import logging
logger = logging.getLogger('my_module_name')
logger.propagate = False

class Corpus:
    '''
    Class for creating a corpus for future models
    '''

    def __init__(self, texts):
        '''
        Parameters initialization
        :param texts:
        '''
        self.texts = texts
        self.dictionary = corpora.Dictionary(self.texts)
        self.corpus = None

    def create_corpus(self):
        '''
        corpus that is generated, consists of arrays of tuples,
        which structure is the following -> (index, amount), where index is the index of a particular token from
        dictionary of tokens (dictionary.token2id), and amount is the amount of times it appears in a particular
        document
        :return:
        '''
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

    def get_corpus(self):
        '''
        :return: corpus
        '''
        return self.corpus

    def add_text(self, text):
        self.texts.append(text)
        self.dictionary = corpora.Dictionary(self.texts)


class TFIDFModel:
    '''
    A class representing a TFIDF model applied on a Corpus
    '''

    def __init__(self, corpus):
        '''
        Initializing parameters
        :param corpus:
        '''
        self.corpus = corpus
        self.tfidf = None
        self.corpus_tfidf = None
        self.tokens_to_weight = 0
        self.counter = 0

    def apply_tfidf(self):
        '''
        Applying tfidf where the general formula of TFiDF is used
        :return:
        '''
        self.tfidf = models.TfidfModel(self.corpus.corpus)

    def get_dfs(self):
        '''
        :return: dictionary {index : amount}, where index - index of the word from dictionary of tokens and
        amount - the number of times the token appears in the general set of documents
        '''
        return self.tfidf.dfs

    @staticmethod
    def _has_number(string):
        '''
        Method to check whether the string contains a numerical values
        :param string: str
        :return: boolean
        '''
        return any(char.isdigit() for char in string)

    def _additional_weight_lambda(self, doc):
        '''
        Calculation of term dependency part of lambda coefficient, which is aimed to decrease the value and importance
        of numerical tokens in the particular document
        :param doc: list of (token index, weight)
        :return: float
        '''
        self.counter += 1
        token_id = self.corpus.dictionary
        tokens_to_weighten = 0
        for t in doc:
            if self._has_number(token_id[t[0]]):
                tokens_to_weighten += 1

        if self.counter <= len(self.corpus.texts):
            self.tokens_to_weight += tokens_to_weighten

        lmbd = tokens_to_weighten / len(doc)
        return lmbd

    def tfidf_corpus(self):
        '''
        Modification of the old corpus into the weighted one, where each token has its own weight regarding the
        set of documents
        The output will consist of vectors (one for each document)
        In some of them not all tokens are included
        That is because there are some tokens that are present in each of the document, which makes them irrelevant
        If talking mathematically, TF * IDF, where IDF is equal to
        log(number of documents / number of documents the token is present in), which in case of a token being in each
        of the documents, is equal to log(n/n) = log(1) = 0
        :return:
        '''
        token_id = self.corpus.dictionary
        self.corpus_tfidf = self.tfidf[self.corpus.corpus]
        updated_corpus = []

        for doc_1 in range(len(self.corpus_tfidf)):
            self._additional_weight_lambda(self.corpus_tfidf[doc_1])

        if self.tokens_to_weight == 0:
            self.tokens_to_weight = 1

        lambda_add_weight = \
            math.log(len(self.corpus.dictionary.id2token) / self.tokens_to_weight)

        for doc in range(len(self.corpus_tfidf)):
            lmbd = self._additional_weight_lambda(self.corpus_tfidf[doc]) * lambda_add_weight

            updated_corpus.append([])
            for t in range(len(self.corpus_tfidf[doc])):
                if self._has_number(token_id[self.corpus_tfidf[doc][t][0]]):

                    lst_doc = list(self.corpus_tfidf[doc][t])
                    lst_doc[1] = lst_doc[1] + lmbd
                    lst_doc = tuple(lst_doc)

                else:
                    lst_doc = self.corpus_tfidf[doc][t]
                updated_corpus[doc].append(lst_doc)

        self.corpus_tfidf = updated_corpus


    def get_tfidf_corpus(self):
        '''
        :return: modified corpus
        '''
        return self.corpus_tfidf


class LSIModel:
    '''
    The class is directed to performing the LSI model to either split data on groups by text similarity or
    find the most similar document to the one outer query
    '''

    def __init__(self, tfidf_corpus, documents):
        '''
        Initializing parameters
        :param tfidf_corpus: the TFIDFModel object
        :param documents: set of documents (raw data)
        '''
        self.tfidf_corpus = tfidf_corpus
        self.documents = documents
        self.lsi_model = None
        self.lsi_corpus = None

    def apply_lsi(self, num_of_topics=None):
        '''
        Here we are creating an LSI vector space, where each document is represented as a numerical vector considering
        number of topics we want to split the data in and other documents
        :param num_of_topics: either default or 90% of min(no.documents, no.tokens) or specified number
        :return:
        '''
        if num_of_topics is None:
            num_of_topics = math.floor(0.9 * min(len(self.tfidf_corpus.corpus.texts),
                                                 len(self.tfidf_corpus.corpus.corpus)))

        self.lsi_model = models.LsiModel(self.tfidf_corpus.get_tfidf_corpus(),
                                         id2word=self.tfidf_corpus.corpus.dictionary,
                                         num_topics=num_of_topics)

        self.lsi_corpus = self.lsi_model[self.tfidf_corpus.corpus_tfidf]

    def grouping(self):
        '''
        This method just prints out the results of grouping which are just vector representations of each of the
        documents
        :return:
        '''
        for doc, as_text in zip(self.lsi_corpus, self.documents):
            print(doc, as_text)

    def similar_queries(self, query=None, top=None):
        '''
        This method is performing a similarity search across the whole vector space by preprocessing the input
        query, creating a vector of the query in the previously formed LSI vector space using the LSI model based
        on documents' corpus
        After query vectorization it uses MatrixSimilarity function to compute the cosine similarity for two topics.
        Then builds a scatter plot of result vectors.
        :param query:
        :return:
        '''

        print("## SIMILAR QUERIES ##")
        if query is None:
            query = input("\nInput your query: ")
        else:
            print("\nInput query: '", query, "'\n")

        self.tfidf_corpus.corpus.add_text(Tokenizer([query]).tokenize()[0])
        self.tfidf_corpus.corpus.create_corpus()

        self.tfidf_corpus.apply_tfidf()
        self.tfidf_corpus.tfidf_corpus()
        vector_query = self.tfidf_corpus.corpus_tfidf[-1]

        vector_lsi_query = self.lsi_model[vector_query]

        print("\nApplying cosine similarity: ")
        index = similarities.MatrixSimilarity(self.lsi_corpus)
        similar = index[vector_lsi_query]
        similar = sorted(enumerate(similar), key=lambda item: -item[1])

        if top:
            print("\n--- TOP ", top, " similar documents ---\n")
        else:
            print("\n--- ALL similar documents ---\n")

        for i, s in enumerate(similar):
            if top == None or i < top:
                print(i + 1, ' - ', s, self.documents[s[0]])

    def get_double_classification(self, save_plot=False):
        '''
        This method is performing the classification by two topics with application of LSI model.
        Can be used right after creating an object of class LSIModel, preferred to use with data .
        :return:
        '''

        print("\n## DOUBLE CLASSIFICATION ##\n----------------------")

        self.apply_lsi(num_of_topics=2)
        # helping containers
        second_topic_container = []
        x, y = [[], []], [[], []]

        maxx, minn = 0, 1000

        # show input data
        print("Input data:\n")
        for document in self.documents:
            print("- ", document)
        print("----------------------")

        # show two topics one by one
        print("Apply classifier:\n\n--- FIRST TOPIC: \n")
        for doc, as_text in zip(self.lsi_corpus, self.documents):

            # relation coefficient for two vectors
            if len(doc) > 1:
                two_vectors_relation = doc[1][1] / doc[0][1]

                if two_vectors_relation > maxx:
                    maxx = two_vectors_relation
                elif two_vectors_relation < minn:
                    minn = two_vectors_relation
            else:
                print("Current vector can`t be classified.")

        lambda_break = min(abs(minn), abs(maxx)) / max(abs(maxx), abs(minn))

        for doc, as_text in zip(self.lsi_corpus, self.documents):
            if len(doc) > 1:
                two_vectors_relation = doc[1][1] / doc[0][1]

                if -1 * two_vectors_relation < lambda_break:
                    print(doc, as_text)

                    x[0].append(doc[0][1])
                    y[0].append(doc[1][1])
                else:
                    second_topic_container.append([doc, as_text])
                    x[1].append(doc[0][1])
                    y[1].append(doc[1][1])

        print("\n--- SECOND TOPIC: \n")
        for doc, as_text in second_topic_container:
            print(doc, as_text)
        print("----------------------")

        # plot result vectors
        self._plot_classification(x, y, save_plot)

    def _plot_classification(self, xxx, yyy, save_plot):
        '''
        Creates scatter plot for visualising vectors of LSI model with 2 topics,
        helping function for get_double_classification(). Shows and saves png plots in current directory.
        :param xxx: x parameters of topic vectors
        :param yyy: y parameters of topic vectors
        :return:
        '''
        fig, ax = plt.subplots()

        for data in zip(xxx, yyy, ['red', 'green'], ["TOPIC 1", "TOPIC 2"]):
            x, y, color, label = data
            ax.scatter(x, y, c=color, alpha=0.5, label=label)

        ax.set_title("LSI Visualization")
        ax.legend()
        if save_plot:
            plt.savefig(str(hash(random.random()))[1:5] + "lsi_representation.png")
        plt.show()

