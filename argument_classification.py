# todo: rename file
import nltk as nltk

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
    def getFeatures(words):
        content = DataManager().get_content(words)
        features = nltk.FreqDist(content).keys()
        return features


