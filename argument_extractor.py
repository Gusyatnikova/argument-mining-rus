from pathlib import Path

from brat_data_collector import BratDataCollector
from bratreader.repomodel import RepoModel
from argument_classification import Classification

# will this path to brat repository be the field of UI?
brat_folder = Path('C:\\Users\\crysn\\Desktop\\Диплом\\prog\\essays\\original')
brat_reader = RepoModel(brat_folder)
collector = BratDataCollector(brat_reader)
data = collector.collect_data()
classificator = Classification().set_data(data)
a = 4
