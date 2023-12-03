import pandas as pd
from ds import Dataset

pd.read_csv("data/tag.csv", encoding='utf-8').to_csv("data/tag.csv", encoding='utf-8', index=True)
