🧠🖼️ Visual Question Answering with ViLT
This project demonstrates how to implement Visual Question Answering (VQA) using the ViLT (Vision-and-Language Transformer) model. Given an image and a natural language question, the model predicts the most appropriate answer.

✨ Concepts
ViLT Model: A vision-and-language transformer that combines image and text processing into a single transformer-based architecture.

Visual Question Answering: A multimodal AI task where the model answers questions related to an input image.

Transformer Processing: Uses a unified attention mechanism to process and align visual and textual information.

✅ Requirements
Python 3.10+

Libraries:

transformers

Pillow

requests

fastapi

uvicorn

📦 Installation
First, clone the repo and install dependencies:

bash
Copy
Edit
git clone https://github.com/yourusername/MLApp-vilt-b32-finetuned-vqa.git
cd MLApp-vilt-b32-finetuned-vqa

pip install -r requirements.txt
Or install them manually:

bash
Copy
Edit
pip install transformers Pillow requests fastapi uvicorn
🚀 Run the FastAPI Server
Start the FastAPI server using:

bash
Copy
Edit
uvicorn app:app --reload
This will launch the server at: http://127.0.0.1:8000

📥 How to Use
Endpoint: POST /ask
Request:

text: The question (form field)

image: The image file (form field)

Example using curl:

bash
Copy
Edit
curl -X POST "http://127.0.0.1:8000/ask" \
-F "text=What is the man doing?" \
-F "image=@example.jpg"
Response:

json
Copy
Edit
{
  "answer": "surfing"
}
🧠 Model Details
The model used is:

python
Copy
Edit
from transformers import ViltProcessor, ViltForQuestionAnswering

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
The image and question are processed as:

python
Copy
Edit
def model_vqa2(text: str, image: Image):
    encoding = processor(image, text, return_tensors="pt")
    output = model(**encoding)
    idx = output.logits.argmax(-1).item()
    return model.config.id2label[idx]
🧪 Project Structure
bash
Copy
Edit
.
├── app.py                  # FastAPI application
├── model.py                # VQA model logic
├── requirements.txt        # Dependencies
├── README.md               # This file
🐳 Docker (Optional)
To build and run with Docker:

bash
Copy
Edit
docker build -t vqa-fastapi .
docker run -p 8000:8000 vqa-fastapi
📚 Resources
ViLT Paper

HuggingFace ViLT Model

FastAPI Docs

🧑‍💻 Author
Emmanuel Díaz Ortega
Bridging deep tech and human intelligence.
@diazmanne

