from pathlib import Path
import argument_classification
from brat_data_collector import BratDataCollector
from bratreader.repomodel import RepoModel
from argument_classification import Classification

# will this path to brat repository be the field of UI?
brat_folder = Path('C:\\Users\\crysn\\Desktop\\Диплом\\prog\\essays\\original')
brat_reader = RepoModel(brat_folder)
collector = BratDataCollector(brat_reader)
data = collector.collect_data()

Classification().set_data(data)
arguments = argument_classification.divided_args
links = argument_classification.divided_links
# how visualise results?
arguments_features = Classification().getFeatures(arguments)
links_features = Classification().getFeatures(links)
a = 4
