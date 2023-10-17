from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import uuid
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import faiss
import os
from transforms import load_transform_image

IMAGEDIR = "fastapi-images/"

model = torch.jit.load("../artifact/model.pt")
index = faiss.read_index("../artifact/index.fs")


app = FastAPI()
# pip install fastapi, python-multipart

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):

    file.filename = f"{uuid.uuid4()}.jpg"
    contents = await file.read()

    with open(os.path.join(IMAGEDIR, file.filename), "wb") as f:
        f.write(contents)

    img_array = load_transform_image(os.path.join(IMAGEDIR, file.filename))

    embedding = model.forward_embedding(img_array)
    embedding = embedding.detach().numpy()

    k = 5
    D, I = index.search(embedding, k)
    print(D, I)

    return {"filename": file.filename}
