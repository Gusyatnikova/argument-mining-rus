import nltk
import os
from classification import Classification

# doc_features = []
doc_features_glob = []

class BratDataCollector:
    def __init__(self, brat_reader):
        self.brat_reader = brat_reader

    def collect_data(self):
        data = []
        for i in range(1, 80):
            document = self.brat_reader.documents['essay' + str(i)]
            data.append(self.extract_doc_features(document, 'essay' + str(i)))
        return data

    def extract_doc_features(self, brat_doc, doc_key):
        doc_features = []
        for annotation in set(brat_doc.annotations):
            ann_features = {'sentence': annotation.repr}
            label_list = []
            for label in annotation.labels.items():
                for sub_label in label:
                    if isinstance(sub_label, list) and len(sub_label) != 0:
                        sub_label = sub_label[0]
                    if sub_label != [] and sub_label is not None:
                        label_list.append(sub_label)
            ann_features['labels'] = label_list
            link_list = []
            for link in annotation.links.items():
                sub_link = link[0]
                for sub_sub_link in link[1]:
                    real_link = {sub_link: sub_sub_link.repr}
                    link_list.append(real_link)
            ann_features['links'] = link_list
            # global doc_features
            doc_features.append(ann_features)
        global doc_features_glob
        doc_features_glob.append({doc_key: doc_features})
        return {doc_key: doc_features}

    def get_test_data(self, doc_key, isannotated):
        if isannotated is True:
            document = self.brat_reader.documents[doc_key]
            annotation_list = []
            for annotation in set(document.annotations):
                annotation_list.append({'annotation': annotation.repr,
                                        'labels': annotation.labels.items(),
                                        'links': annotation.links})
            data = self.extract_test_sentences(annotation_list)
            return data
        else:
            import argument_extractor
            file_dir = argument_extractor.brat_folder
            with open(os.path.join(file_dir, doc_key + '.txt'), 'r') as file:
                raw_text = file.read()
                file.close()
                raw_sentences = nltk.sent_tokenize(raw_text)
                # try delete stopwords in sentences
            return raw_sentences
        pass

    def extract_test_sentences(self, data):
        sentences = []
        for line in data:
            sentences.append(line['annotation'])
        return sentences

    def get_arguments(self):
        # todo: move here from get_data else block
        pass

    def get_accuracy_data(self):
        classification = Classification()
        data = []
        data = doc_features_glob
        for i in range(80, 90):
            document = self.brat_reader.documents['essay' + str(i)]
            self.extract_doc_features(document, 'essay' + str(i))
            # data.append(doc_features_glob[-1])
            classification.set_data(data)
        arguments = classification.divided_args
        links = classification.divided_links

        arguments_features = classification.getFeatures(arguments)
        links_features = classification.getFeatures(links)
# delete ?
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

        arguments_test_set = nltk.classify.apply_features(extract_features, arguments)
        links_test_set = []
        # links_test_set = nltk.classify.apply_features(extract_features, links)
        return [arguments_test_set, links_test_set]
