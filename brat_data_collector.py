class BratDataCollector:
    def __init__(self, brat_reader):
        self.brat_reader = brat_reader

    def collect_data(self):
        data = []
        for i in range(1, len(self.brat_reader.documents)):
            document = self.brat_reader.documents['essay' + str(i)]
            data.append(self.extract_doc_features(document, 'essay' + str(i)))
        return data

    def extract_doc_features(self, brat_doc, doc_key):
        sentence_list = []
        label_list = []
        link_list = []
        doc_features = {}
        for annotation in set(brat_doc.annotations):
            print(brat_doc)
            # print("annotation :", annotation.repr)
            # print("labels :", annotation.labels.items())
            # print("links :", annotation.links)
            # print("********************************************************************************")
            sentence_list.append(annotation.repr)
            for label in annotation.labels.items():
                for sub_label in label:
                    if isinstance(sub_label, list) and len(sub_label) != 0:
                        sub_label = sub_label[0]
                    if sub_label != [] and sub_label is not None:
                        label_list.append(sub_label)
            for link in annotation.links.items():
                sub_link = link[0]
                for sub_sub_link in link[1]:
                    real_link = {sub_link: sub_sub_link.repr}
                    link_list.append(real_link)
        doc_features['key'] = doc_key
        doc_features['sentences'] = sentence_list
        doc_features['labels'] = label_list
        doc_features['links'] = link_list
        return doc_features

