import os
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import json

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
train_data, train_label_to_id = read_data('dataset/train')
valid_data, valid_label_to_id = read_data('dataset/validate')

# Сохранение словаря меток
os.makedirs('./results', exist_ok=True)
with open('./results/label_to_id.json', 'w') as f:
    json.dump(train_label_to_id, f)

# Создание датасетов
train_dataset = Dataset.from_pandas(train_data)
valid_dataset = Dataset.from_pandas(valid_data)

# Токенизация данных
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)

# Преобразование меток в тензоры
def format_dataset(dataset):
    dataset = dataset.map(lambda examples: {'labels': examples['label']}, batched=True)
    return dataset

tokenized_train = format_dataset(tokenized_train)
tokenized_valid = format_dataset(tokenized_valid)

# Загрузка модели BERT
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(train_label_to_id))

# Настройка параметров обучения
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",  # Обновите evaluation_strategy на eval_strategy
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
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
)

# Обучение модели
trainer.train()

# Сохранение модели
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')
