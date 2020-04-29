from pathlib import Path
import nltk
from data_collector import BratDataCollector
from bratreader.repomodel import RepoModel
from classification import Classification
from deeppavlov import build_model, configs
import operator

# will this path to brat repository be the field of UI?
brat_folder = Path('D:\\Диплом\\prog\\essays\\original')
# brat_folder = Path('D:\\Диплом\\prog\\essays\\russian')

brat_reader = RepoModel(brat_folder)

collector = BratDataCollector(brat_reader)
data = collector.collect_data()

def get_sentiment_statistic(data, language, deeppavlov_model):
    from data_manager import DataManager
    correct_labels = ['Premise', 'Claim']
    ru_sentiment = ['positive', 'neutral', 'negative']
    en_sentiment = ['Positive', 'Neutral', 'Negative']
    cur_sentiment = []
    if language == 'ru':
        cur_sentiment = ru_sentiment
    if language == 'en':
        cur_sentiment = en_sentiment

    args = DataManager().filter_labels(data, correct_labels)
    all_premises = 0
    all_claims = 0
    premises_neg = 0
    premises_neutral = 0
    premises_pos = 0
    claims_neg = 0
    claims_neutral = 0
    claims_pos = 0
    for tuple in args:
        if tuple[0] == 'Claim':
            all_claims = all_claims + 1
            sentiment = deeppavlov_model([tuple[1]])[0]
            if sentiment == cur_sentiment[0]:
                claims_pos = claims_pos + 1
            if sentiment == cur_sentiment[1]:
                # claims_neutral = claims_neutral + 1
                claims_pos = claims_pos + 1
            if sentiment == cur_sentiment[2]:
                claims_neg = claims_neg + 1
        else:
            all_premises = all_premises + 1
            sentiment = deeppavlov_model([tuple[1]])[0]
            if sentiment == cur_sentiment[0]:
                premises_pos = premises_pos + 1
            if sentiment == cur_sentiment[1]:
               # premises_neutral = premises_neutral + 1
               premises_pos = premises_pos + 1
            if sentiment == cur_sentiment[2]:
                premises_neg = premises_neg + 1
    print('language: ', language, '\n',
          'claims-pos: ', claims_pos/all_claims*100, '%\n',
          'claims-neg: ', claims_neg/all_claims*100, '%\n',
          'claims-neutral: ', claims_neutral/all_claims*100, '%\n',
          'premises_pos: ', premises_pos/all_premises*100, '%\n',
          'premises_neutral: ', premises_neutral/all_premises*100, '%\n',
          'premises_neg: ', premises_neg/all_premises*100, '%\n',)
    a = 4

""" deeppavlov for english dataset """
en_deeppavlov_model = build_model(configs.classifiers.sentiment_twitter)
get_sentiment_statistic(data, 'en', en_deeppavlov_model)

""" deeppavlov for russian dataset """
# ru_deeppavlov_model = build_model(configs.classifiers.rusentiment_cnn, download=True)
# get_sentiment_statistic(data, 'ru', ru_deeppavlov_model)



classifier = Classification()
classifier.set_data(data)
arguments = classifier.get_divided_args()
'''
create frequency of word for claims and premices
'''
all_claims_words = []
all_premices_words = []

for item in arguments:
    if item[1] == 'Claim':
        all_premices_words.extend(item[0])
    else:
        all_claims_words.extend(item[0])

claims_freq_list = []
premices_freq_list = []
already_in_claims = []
already_in_premices = []

for word in all_premices_words:
    if word not in already_in_premices:
        already_in_premices.append(word)
        premices_freq_list.append((word, all_premices_words.count(word)))
for word in all_claims_words:
    if word not in already_in_claims:
        already_in_claims.append(word)
        claims_freq_list.append((word, all_claims_words.count(word)))

premices_freq_list.sort(key=operator.itemgetter(1))
premices_freq_list = premices_freq_list[::-1]
claims_freq_list.sort(key=operator.itemgetter(1))
claims_freq_list = claims_freq_list[::-1]

links = classifier.get_divided_links()
# how visualise results?
arguments_features = classifier.getFeatures(arguments)
links_features = classifier.getFeatures(links)


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

arguments_training_set = nltk.classify.apply_features(extract_features, arguments)
links_training_set = nltk.classify.apply_features(extract_features, links)

# --- uncomment when train data was edited
classifier.train_classifiers(arguments_training_set, links_training_set)
# ---

# --- loading classifiers from pickle package ----------------#

args_naivebayes_classifier = classifier.load_classifier('args_naivebayes.pickle')
links_naivebayes_classifier = classifier.load_classifier('links_naivebayes.pickle')

args_sklearn_classifier = classifier.load_classifier('args_sklearn.pickle')
links_sklearn_classifier = classifier.load_classifier('links_sklearn.pickle')

args_logisticregression_classifier = classifier.load_classifier('args_logisticregression.pickle')
links_logisticregression_classifier = classifier.load_classifier('links_logisticregression.pickle')


# --- Get prediction for test data





total_equals = 0
total_sentences = 0
for i in range(81, 91):
    test_data = collector.get_test_data('essay'+str(i), True)
    # test_data = collector.get_test_data('essay81', False)

    args_predicted = []
    links_predicted = []
    sen_matches = 0
    for sentence in set(test_data):
        total_sentences = total_sentences + 1
        naivebayes_prediction = args_naivebayes_classifier.classify(extract_features(sentence.split()))
       # print('Naive Bayes Prediction: \n')
        print((sentence, naivebayes_prediction))
        # print('\n________________************************************************________________\n')
        args_predicted.append((sentence, naivebayes_prediction))

        sklearn_prediction = args_sklearn_classifier.classify(extract_features(sentence.split()))
      #  print('Sklearn Prediction \n')
        print((sentence, sklearn_prediction))
      #   print('\n________________************************************************________________\n')
        args_predicted.append((sentence, sklearn_prediction))

        logisticregression_prediction = args_logisticregression_classifier.classify(extract_features(sentence.split()))
     #    print('Logistic Regression Prediction \n')
        print((sentence, logisticregression_prediction))
    #     print('\n________________************************************************________________\n')
        args_predicted.append((sentence, logisticregression_prediction))
        if naivebayes_prediction == sklearn_prediction == logisticregression_prediction:
            total_equals = total_equals + 1
    #         print('\n Prediction is equal to: ', (sentence, sklearn_prediction))
            with open('report_HandleTokenize.txt', 'a') as report_file:
                report_file.write('\n sen: ' + sentence + ' prediction: --' + sklearn_prediction + '--')
            # print('\n Prediction is equal to: ', (sentence, sklearn_prediction), '\n sentiment: ', deeppavlov_model([sentence]))
    print(' all_sentences: ', len(test_data), ' matches: ', total_equals, ' matches%: ', total_equals / len(test_data))
    print("\n\ntotal sentences: ", total_sentences, ' total_equals: ', total_equals, 'common matches: ', total_equals/total_sentences)
'''
predicted_claims = []
predicted_premises = []
for prediction_pair in args_predicted:
    if prediction_pair[1] == 'Claim':
        predicted_claims.append((prediction_pair[0], prediction_pair[1]))
    if prediction_pair[1] == 'Premise':
        predicted_claims.append((prediction_pair[0], prediction_pair[1]))
'''
# relation analyzer may be here

# accuracy_data = collector.get_accuracy_data()
# arg_test_set = accuracy_data[0]
# links_test_set = accuracy_data[1]

# from Metrics import Metrics

# Metrics().evaluation(arg_test_set, args_logisticregression_classifier)

'''
print("NB: ", )
print("Naive Bayes: ",ArgNaiveBayesScore,LinkNaiveBayesScore)
print("Sklearn : ",ArgSklearnBayesScore,LinkSklearnBayesScore)
print("Logistic reg: ",ArglogBayesScore,LinklogBayesScore)
'''


