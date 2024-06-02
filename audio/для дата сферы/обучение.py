import os
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import json
from sklearn.model_selection import train_test_split

# Функция для чтения данных из текстовых файлов
def read_data(data_dir):
    data = []
    label_to_id = {}
    label_id = 0
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            if label not in label_to_id:
                label_to_id[label] = label_id
                label_id += 1
            for file in os.listdir(label_dir):
                if file.endswith('.txt'):
                    file_path = os.path.join(label_dir, file)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                        count_str = lines[0].strip()
                        combined_lines = ''.join(lines[1:]).strip()
                        data.append((combined_lines, label_to_id[label]))
    return pd.DataFrame(data, columns=['text', 'label']), label_to_id

# Чтение данных
train_data, train_label_to_id = read_data('/home/jupyter/work/resources/dataset/train/')
valid_data, valid_label_to_id = read_data('/home/jupyter/work/resources/dataset/validate')

# Сохранение словаря меток
os.makedirs('/home/jupyter/work/resources/results', exist_ok=True)
with open('/home/jupyter/work/resources/results/label_to_id.json', 'w') as f:
    json.dump(train_label_to_id, f)

# Разделение данных на тренировочные и валидационные
train_texts, valid_texts, train_labels, valid_labels = train_test_split(
    train_data['text'], train_data['label'], test_size=0.1
)

# Токенизация данных
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, return_tensors="pt")
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True, return_tensors="pt")

# Создание PyTorch датасетов
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CustomDataset(train_encodings, list(train_labels))
valid_dataset = CustomDataset(valid_encodings, list(valid_labels))

# Загрузка модели BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(train_label_to_id))

# Настройка параметров обучения
# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir='/home/jupyter/work/resources/results',
    evaluation_strategy="epoch",  # Обновите evaluation_strategy на eval_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",  # Сохранение модели на каждой эпохе
)

# Создание тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
)

# Обучение модели
trainer.train()

# Сохранение модели
model.save_pretrained('/home/jupyter/work/resources/results')
tokenizer.save_pretrained('/home/jupyter/work/resources/results')
