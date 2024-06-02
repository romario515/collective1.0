import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Загрузка токенизатора и модели
tokenizer = BertTokenizer.from_pretrained('./results')
model = BertForSequenceClassification.from_pretrained('./results')

# Загрузка словаря меток
with open('./results/label_to_id.json', 'r') as f:
    label_to_id = json.load(f)

# Обратный словарь для отображения меток
id_to_label = {v: k for k, v in label_to_id.items()}

# Функция для предсказания на входном тексте
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions


# Входные данные
input_text = """
2
Model 2|train|81.38|51.4927978515625|20.13525390625|1224.0775146484375|582999999999999999.4278564453125
Model 2|person|74.53|993.0375456754674646444955078125|94.64898681640625|286.814697265625|611.1302490234375

"""

# Обработка текста для модели
lines = input_text.strip().split('\n')
count = lines[0]
text_lines = '\n'.join(lines[1:])

# Предсказание
predictions = predict(text_lines)
predicted_label_id = torch.argmax(predictions, dim=1).item()

# Вывод предсказаний
predicted_label = id_to_label[predicted_label_id]
print(f"Predicted label: {predicted_label}")

# Вывод вероятностей для всех классов
for i, prob in enumerate(predictions[0]):
    label = id_to_label[i]
    print(f"{label}: {prob.item() * 100:.2f}%")