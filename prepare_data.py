import pandas as pd
import json
from preprocessing import *

train_claim = pd.read_csv("data/checkworthy/english_train.tsv", sep='\t')
print(train_claim)

train_stance = pd.read_json("data/stance/english_train.json", orient='records', lines=True)
print(train_stance)

dev_stance = pd.read_json("data/stance/english_dev.json", orient='records', lines=True)
print(train_stance)

cleanup_dataframe(dev_stance)
cleanup_dataframe(train_stance)

dev_stance_sub = dev_stance[["id", "rumor", "label"]]
train_stance_sub = train_stance[["id", "rumor", "label"]]

train_stance.to_json("data/stance/cleaned_train.json", orient='records', lines=True)
dev_stance.to_json("data/stance/cleaned_dev.json", orient='records', lines=True)

train_stance_sub.to_csv("data/stance/cleaned_train.tsv", sep='\t')
dev_stance_sub.to_csv("data/stance/cleaned_dev.tsv", sep='\t')