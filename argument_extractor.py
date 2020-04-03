from pathlib import Path
import argument_classification
import nltk
from brat_data_collector import BratDataCollector
from bratreader.repomodel import RepoModel
from argument_classification import Classification

# will this path to brat repository be the field of UI?
brat_folder = Path('C:\\Users\\crysn\\Desktop\\Диплом\\prog\\essays\\original')
brat_reader = RepoModel(brat_folder)
collector = BratDataCollector(brat_reader)
data = collector.collect_data()

Classification().set_data(data)
arguments = argument_classification.divided_args
links = argument_classification.divided_links
# how visualise results?
arguments_features = Classification().getFeatures(arguments)
# print(arguments_features)
links_features = Classification().getFeatures(links)


def argument_extract_features(document):
    argument_features = {}
    for word in arguments_features:
        argument_features['hold(%s)' % word] = (word in set(document))
    return argument_features


def extract_features(document):
    features = {}
    if 'Support' and 'Attack' in document:
        src = links_features
    else:
        src = arguments_features
    for word in src:
        features['hold(%s)' % word] = (word in set(document))
    return features


argument_training_set = nltk.classify.apply_features(extract_features, arguments)
links_training_set = nltk.classify.apply_features(extract_features, links)



