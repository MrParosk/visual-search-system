import glob

import faiss
import torch
from transforms import load_transform_image


model = torch.load("/home/user/artifact/model.pt")
files = glob.glob("/home/user/data/caltech-101/**/*.jpg")


index = faiss.IndexFlatIP(64)

with torch.no_grad():
    for file in files[0:1000]:
        img_array = load_transform_image(file)

        embedding = model.forward_embedding(img_array)
        embedding = embedding.numpy()

        index.add(embedding)


faiss.write_index(index, "/home/user/artifact/index.fs")
