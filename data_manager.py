
class DataManager:

    def __init__(self):
        '''
    def args_filter(self, argument):
        if argument == 'Premise' or argument == 'Claim' or argument == 'MajorClaim':
            return True
        else:
            return False
'''

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
