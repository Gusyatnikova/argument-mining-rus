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
            doc_features.append(ann_features)
        return {doc_key: doc_features}

    def get_test_data(self, doc_key):
        document = self.brat_reader.documents[doc_key]
        annotation_list = []
        for annotation in set(document.annotations):
            annotation_list.append({'annotation': annotation.repr,
                                    'labels': annotation.labels.items(),
                                    'links': annotation.links})
        data = self.extract_test_sentences(annotation_list)
        return data

    def extract_test_sentences(self, data):
        sentences = []
        for line in data:
            sentences.append(line['annotation'])
        return sentences



