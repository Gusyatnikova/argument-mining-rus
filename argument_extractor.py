from pathlib import Path
from brat_data_collector import bratDataCollector
from bratreader.repomodel import RepoModel

# will this path to brat repository be the field of UI?
brat_repo = Path('C:\\Users\\crysn\\Desktop\\Диплом\\prog\\essays\\original')
brat_reader = RepoModel(brat_repo)
collector = bratDataCollector(brat_reader)
data = collector.collect_data()
