from data_manager import DataManager

correct_labels = ['Premise', 'Claim', 'MajorClaim']
correct_links = ['Support', 'Attacks']


class Classification:

    def __init__(self):
        self.data = []
        self.divided_args = []
        self.divided_links = []
        pass

    def set_data(self, data):
        self.data = data
        self.divided_args = DataManager().filter_labels(data, correct_labels)
        self.divided_links = DataManager().filter_links(data)
        a = 4
        pass

    '''

    def prepare_train_data(self):
        arguments = []
        for doc in self.data:
            # each essay
            labels = filter(self.args_filter, doc['labels'])
            arguments = [labels, doc['sentences']]


        # arguments = self.data[]
        
        words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
        все в нижнем регистре, разделены на слова, убраны слова меньше трех, вид: метка - набор слов
        :return: [filteredArgWords, filteredLink words] 
     
        pass
        '''



