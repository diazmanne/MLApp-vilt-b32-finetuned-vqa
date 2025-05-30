from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def model_vqa2(text: str, image: Image):
    encoding = processor(image, text, return_tensors="pt")
    output = model(**encoding)
    logits = output.logits
    idx = logits.argmax(-1).item()

    return model.config.id2label[idx]
