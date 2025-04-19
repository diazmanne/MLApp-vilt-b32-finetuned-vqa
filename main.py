from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io

from model import model_vqa2

app = FastAPI()

#@app.get("/")
#def read_root():
#    return {"Hello": "World"}

@app.post("/ask")
def ask(text: str, image: UploadFile):
    content = image.file.read()
    img = Image.open(io.BytesIO(content))
    result = model_vqa2(text, img)
    return {"answer": result}