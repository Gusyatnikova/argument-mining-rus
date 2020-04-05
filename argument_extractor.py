from pathlib import Path
import classification
import nltk
import pickle_files
import pickle
from brat_data_collector import BratDataCollector
from bratreader.repomodel import RepoModel
from classification import Classification

from deeppavlov import build_model, configs

# will this path to brat repository be the field of UI?
brat_folder = Path('C:\\Users\\crysn\\Desktop\\Диплом\\prog\\essays\\original')
brat_reader = RepoModel(brat_folder)
collector = BratDataCollector(brat_reader)
data = collector.collect_data()

Classification().set_data(data)
arguments = classification.divided_args
links = classification.divided_links
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

# todo: move this section to argument_classification
# todo: divide training classifiers, save it to dir
# todo: use saved classifiers for classify, will be faster

arguments_training_set = nltk.classify.apply_features(extract_features, arguments)
links_training_set = nltk.classify.apply_features(extract_features, links)

# uncomment when train data was edited
Classification().train_classifiers(arguments_training_set, links_training_set)

# --- Loading Classifiers from pickle file ----------------#

##### Naive Bayes

args_naivebayes_classifier = open('pickle_files/args_naivebayes.pickle', "rb")
args_naivesentiment_classifier = pickle.load(args_naivebayes_classifier, encoding="latin1")
args_naivebayes_classifier.close()

links_naivebayes_classifier = open('pickle_files/links_naivebayes.pickle', "rb")
links_naivesentiment_classifier = pickle.load(links_naivebayes_classifier)
links_naivebayes_classifier.close()

test_data = collector.get_test_data('essay81')

# deeppavlov usage rusentiment_elmo_twitter_cnn.json
#todo: save deeppavlov to pickle
deeppavlov_model = build_model(configs.classifiers.sentiment_twitter, download=True)
'''
try get some predictions
'''
args_predicted = []
links_predicted = []
total_equals = 0

for sentence in test_data:
    naivebayes_prediction = args_naivebayes_classifier.classify(extract_features(sentence.split()))
    print('Naive Bayes Prediction \n')
    print((sentence, naivebayes_prediction))
    print('************************************************')
    args_predicted.append((sentence, naivebayes_prediction))

    sklearn_prediction = args_naivebayes_classifier.classify(extract_features(sentence.split()))
    print('Sklearn Prediction \n')
    print((sentence, sklearn_prediction))
    print('************************************************')
    args_predicted.append((sentence, sklearn_prediction))

    logisticregression_prediction = args_logisticreg_classifier.classify(extract_features(sentence.split()))
    print('Logistic Regression Prediction \n')
    print((sentence, logisticregression_prediction))
    print('************************************************')
    args_predicted.append((sentence, logisticregression_prediction))

    if naivebayes_prediction == sklearn_prediction == logisticregression_prediction:
        total_equals = total_equals + 1
        print('\n Prediction is equal to: ', (sentence, sklearn_prediction),
              '\n sentiment: ', deeppavlov_model([sentence]))
print(' all_sentences: ', len(test_data), ' matches: ', total_equals, ' matches%: ', total_equals/len(test_data))
