import os
import pickle
import nltk as nltk
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from data_manager import DataManager

correct_labels = ['Premise', 'Claim', 'MajorClaim']
correct_links = ['Support', 'Attacks']

divided_args = []
divided_links = []

class Classification:

    def __init__(self):
        pass

    def set_data(self, data):
        args = DataManager().filter_labels(data, correct_labels)
        links = DataManager().filter_links(data)
        global divided_args
        divided_args = DataManager().divide_sentences(args)
        global divided_links
        divided_links = DataManager().divide_sentences(links)

    @staticmethod
    def save_pickle(classifier, pickle_name):
        pickle_dir = 'pickle_files'
        filename = os.path.join(pickle_dir, pickle_name)
        with open(filename, 'wb') as f:
            pickle.dump(classifier, f)

    @staticmethod
    def getFeatures(words):
        content = DataManager().get_content(words)
        features = nltk.FreqDist(content).keys()
        return features

    def set_naivebayes_classifier(self, train_args, train_links):
        classifier = nltk.NaiveBayesClassifier.train(train_args)
        Classification().save_pickle(classifier, 'args_naivebayes.pickle')
        classifier = nltk.NaiveBayesClassifier.train(train_links)
        Classification().save_pickle(classifier, 'links_naivebayes.pickle')
        pass

    def set_sklearn_classifier(self, train_args, train_links):
        classifier = SklearnClassifier(MultinomialNB()).train(train_args)
        self.save_pickle(classifier, 'args_sklearn.pickle')
        classifier = SklearnClassifier(MultinomialNB()).train(train_links)
        self.save_pickle(classifier, 'links_sklearn.pickle')
        pass

    def set_logisticregression_classifier(self, train_args, train_links):
        classifier = SklearnClassifier(LogisticRegression()).train(train_args)
        self.save_pickle(classifier, 'args_logisticregression.pickle')
        classifier = SklearnClassifier(LogisticRegression()).train(train_links)
        self.save_pickle(classifier, 'links_logisticregression.pickle')
        pass

    def train_classifiers(self, arguments_training_set, links_training_set):
        self.set_naivebayes_classifier(arguments_training_set, links_training_set)
        self.set_sklearn_classifier(arguments_training_set, links_training_set)
        self.set_logisticregression_classifier(arguments_training_set, links_training_set)
        pass

    def load_classifier(self, filename):
        classifier_file = open('pickle_files/'+filename, "rb")
        classifier = pickle.load(classifier_file, encoding="latin1")
        classifier_file.close()
        return classifier








