import uuid
import os
import glob

from torchvision import io
import torch
import faiss
from fastapi import FastAPI, File, UploadFile


IMAGEDIR = "/home/user/"

model = torch.jit.load("/home/user/artifact/model.pt")
transforms = torch.jit.load("/home/user/artifact/transforms.pt")
index = faiss.read_index("/home/user/artifact/index.fs")


files = glob.glob("/home/user/data/caltech-101/**/*.jpg")


app = FastAPI()

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    with open(os.path.join(IMAGEDIR, file.filename), "wb") as f:
        f.write(contents)

    img_array = io.read_image(os.path.join(IMAGEDIR, file.filename))
    img_array = transforms(img_array)

    embedding = model.forward_embedding(img_array)
    embedding = embedding.detach().numpy()

    k = 5
    D, I = index.search(embedding, k)
    D, I = D[0, :].tolist(), I[0, :].tolist()

    closest_images = []
    for d, i in zip(D, I):
        closest_images.append(
            {
                "file": files[i],
                "distance": d
            }
        )

    return closest_images
