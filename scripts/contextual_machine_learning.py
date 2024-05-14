import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from qdrant_client import QdrantClient
import numpy as np
import warnings

warnings.filterwarnings("ignore")

client = QdrantClient(host="localhost", port=6333)
my_collection = "image_collection"

def extract_proba_errors(pred):
    probs = [proba_0 := pred[:, 0][0],
    proba_1 := pred[:, 1][0],
    proba_2 := pred[:, 2][0]]
    if probs.index(max(probs)) == 2:
        return True, 0
    elif probs.index(max(probs)) == 1:
        return False, probs[1]
    else:
        return False, 0.05

def return_top_250(client, collection, query_embedding, train_objs, train_labels):
    results = client.search(
        collection_name=collection,
        query_vector=query_embedding,
        limit=250
    )
    return [train_objs[i.id] for i in results], [train_labels[i.id] for i in results]

def train_top_250(top50objs, top50labels, model):
    X = pd.concat(top50objs, ignore_index=True)
    y = top50labels
    trained_model = model.fit(X,y)
    return trained_model

def extract_proba(pred):
    proba_0 = pred[:, 0][0]
    proba_1 = pred[:, 1][0]
    if proba_0 >= proba_1:
        return 1 - proba_0
    else:
        return proba_1

print("Pred")

train_csv = pd.read_csv("data/combined_pca.csv", index_col=0)
train_csv = train_csv.drop(columns=["original_glcm_MCC.1"])
labels = list(train_csv["LABEL"])
train_labels = train_csv["LABEL"]
train_csv = train_csv.drop(columns=["LABEL"])
train_pred_objs = [pd.DataFrame.from_dict({key: [train_csv.iloc[i][key]] for key in list(train_csv.iloc[i].keys())}) for i in range(len(labels))]

test_csv = pd.read_csv("data/extracted_test_pca.csv", index_col=0)
test_csv = test_csv.drop(columns=["original_glcm_MCC.1"])
test_pred_objs = [pd.DataFrame.from_dict({key: [test_csv.iloc[i][key]] for key in list(test_csv.iloc[i].keys())}) for i in range(100)]

embeddings1 = np.load("data/vectors_test.npy").tolist()

for i in range(len(test_pred_objs)):
    query = embeddings1[i]
    top250objs, top250labels = return_top_250(client, my_collection, query, train_pred_objs, labels)
    trained_model = train_top_250(top250objs, top250labels, HistGradientBoostingClassifier())
    pred = extract_proba(trained_model.predict_proba(test_pred_objs[i]))
    print(pred)