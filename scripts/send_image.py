import io

import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


filename = "../data/caltech-101/airplanes/image_0081.jpg"


with open(filename, "rb") as fp:
    img = Image.open(fp, mode="r")

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()


files = {"file": (filename, img_byte_arr)}

response = requests.post(
    "http://0.0.0.0:8080/images",
    files=files,
)


f, axarr = plt.subplots(1, 6, figsize=(15, 15))
axarr[0].imshow(np.asarray(img))
axarr[0].set_title("Input image")

for idx, file_dict in enumerate(response.json()):
    file_path = file_dict["file"].replace("/home/user/", "../")

    with open(file_path, "rb") as fp:
        img = Image.open(fp, mode="r")
        img = np.asarray(img)
    
    axarr[idx + 1].imshow(img)
    axarr[idx + 1].set_title(f"Distance={format(file_dict['distance'], '.1f')}")


plt.show()
