import glob
import json

import faiss
import torch
from torchvision import io


model = torch.jit.load("/home/user/artifact/model.pt")
transforms = torch.jit.load("/home/user/artifact/transforms.pt")
files = glob.glob("/home/user/data/caltech-101/**/*.jpg")


index = faiss.IndexFlatIP(64)

with torch.no_grad():
    for file in files[0:1000]:
        img_array = io.read_image(file)
        img_array = transforms(img_array)

        embedding = model.forward_embedding(img_array)
        embedding = embedding.numpy()

        index.add(embedding)

with open("/home/user/artifact/image_files.json", "w") as fp:
    json.dump(files, fp)

faiss.write_index(index, "/home/user/artifact/index.fs")
