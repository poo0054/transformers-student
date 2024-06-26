from transformers import pipeline

vision_classifier = pipeline(model="google/vit-base-patch16-224")
preds = vision_classifier(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)

print(preds)

preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]

print(preds)
