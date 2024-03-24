import pytesseract
from transformers import pipeline

pytesseract.pytesseract.tesseract_cmd = 'D:/tools/Tesseract-OCR/tesseract.exe'

vqa = pipeline(model="impira/layoutlm-document-qa")
# l = vqa(image="E:/project/python/AL0-SWNMDUDBPWYCC_page-0001.jpg",
#         question="商品编号是什么?", )
l = vqa(image="E:/project/python/AL0-SWNMDUDBPWYCC_page-0004.jpg",
        question="What is the QUANTITY ?")

print(l)
