import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json

# Абсолютные пути
RESULTS_DIR = '/home/jupyter/work/resources/results'

# Загрузка токенизатора и модели
tokenizer = BertTokenizer.from_pretrained(f'{RESULTS_DIR}')
model = BertForSequenceClassification.from_pretrained(f'{RESULTS_DIR}')

# Загрузка словаря меток
with open(f'{RESULTS_DIR}/label_to_id.json', 'r') as f:
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
Model 3|rail|83.95|730.3089599609375|0.0|255.2872314453125|680.51123046875
Model 3|rail|27.05|517.669921875|11.56787109375|144.6182861328125|676.1633911132812

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
