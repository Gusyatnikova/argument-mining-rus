class Classification:

    def __init__(self):
        self.data = []
        self.pre_train_data = []
        pass

    def set_data(self, data):
        self.data = data
        # self.prepare_train_data()
        # call reformat data to pre_train_data
        pass

    '''
    def args_filter(self, argument):
        if argument == 'Premise' or argument == 'Claim' or argument == 'MajorClaim':
            return True
        else:
            return False

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



