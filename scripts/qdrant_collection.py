from datasets import load_dataset
import torch
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(host="localhost", port=6333)

my_collection = "image_collection"

client.recreate_collection(
    collection_name=my_collection,
    vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE)
)

dataset = load_dataset("imagefolder", data_dir="breastcancer-ultrasound/", split="train")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
model = AutoModel.from_pretrained('facebook/dinov2-large').to(device)

def get_embeddings(batch):
    inputs = processor(images=batch['image'], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    batch['embeddings'] = outputs
    return batch

dataset = dataset.map(get_embeddings, batched=True, batch_size=16)

np.save("data/vectors_train", np.array(dataset['embeddings']), allow_pickle=False)

dataset1 = load_dataset("imagefolder", data_dir="breastcancer-ultrasound/", split="test")

dataset1 = dataset1.map(get_embeddings, batched=True, batch_size=16)

np.save("data/vectors_test", np.array(dataset1['embeddings']), allow_pickle=False)

payload = dataset.select_columns([
    "label"
]).to_pandas().fillna(0).to_dict(orient="records")

ids = list(range(dataset.num_rows))
embeddings = np.load("data/vectors_train.npy").tolist()

batch_size = 1000

for i in range(0, dataset.num_rows, batch_size):

    low_idx = min(i+batch_size, dataset.num_rows)

    batch_of_ids = ids[i: low_idx]
    batch_of_embs = embeddings[i: low_idx]
    batch_of_payloads = payload[i: low_idx]

    client.upsert(
        collection_name=my_collection,
        points=models.Batch(
            ids=batch_of_ids,
            vectors=batch_of_embs,
            payloads=batch_of_payloads
        )
    )
