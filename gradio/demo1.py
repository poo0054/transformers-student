import gradio as gr
import pytesseract
from transformers import pipeline

pytesseract.pytesseract.tesseract_cmd = 'D:/tools/Tesseract-OCR/tesseract.exe'

# pipe = pipeline("image-classification", model="google/vit-base-patch16-224")
pipe = pipeline(model="impira/layoutlm-document-qa")

gr.Interface.from_pipeline(pipe).launch(share=True)
