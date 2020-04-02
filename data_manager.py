
class DataManager:

    def __init__(self):
        pass

    @staticmethod
    def filter_labels(data, correct_labels):
        filtered_labels = []
        for document in data:
            for sentence_list in document.items():
                for sentence in sentence_list[1]:
                    for label in sentence['labels']:
                        if label in correct_labels:
                            filtered_labels.append((label, sentence['sentence']))
        return filtered_labels

    @staticmethod
    def filter_links(data):
        filtered_links = []
        for document in data:
            for sentence_list in document.items():
                for sentence in sentence_list[1]:
                    for link in sentence['links']:
                        for sub_link in link.items():
                            filtered_links.append(sub_link)
        return filtered_links

    @staticmethod
    def divide_sentences(sentences):
        divided = []
        for(label, words) in sentences:
            divided_lowercase = [x.lower() for x in words.split() if len(x) >= 3]
            divided.append((label, divided_lowercase))
        return divided
