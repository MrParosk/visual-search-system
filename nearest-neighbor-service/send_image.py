# client.py
import requests
import io
import io
from PIL import Image

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

print(response.json())
