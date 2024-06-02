import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import torch

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
                        count_str = int(lines[0].strip())  # Считывание первого значения, указывающего количество строк
                        combined_text = ' '.join([line.strip() for line in lines[1:]])
                        data.append((combined_text, label_to_id[label]))
                        print(1)
    return pd.DataFrame(data, columns=['text', 'label']), label_to_id

# Чтение данных
train_data, train_label_to_id = read_data('dataset/train')
valid_data, valid_label_to_id = read_data('dataset/validate')

# Преобразование меток в числовые значения
label_encoder = LabelEncoder()
train_data['label'] = label_encoder.fit_transform(train_data['label'])
valid_data['label'] = label_encoder.transform(valid_data['label'])

# Преобразование данных в массивы numpy
X_train = train_data['text'].values
y_train = train_data['label'].values
X_valid = valid_data['text'].values
y_valid = valid_data['label'].values

# Преобразование текстовых данных в числовые векторы
vectorizer = TfidfVectorizer(max_features=1000)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_valid_vec = vectorizer.transform(X_valid).toarray()

# Сохранение векторизатора
import pickle
with open('./tabnet_model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Обучение модели TabNet
clf = TabNetClassifier()

clf.fit(
    X_train_vec, y_train,
    eval_set=[(X_train_vec, y_train), (X_valid_vec, y_valid)],
    eval_name=['train', 'valid'],
    #eval_metric=['accuracy'],
    max_epochs=100,
    patience=10,
    batch_size=8,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Сохранение модели
os.makedirs('./tabnet_model', exist_ok=True)
clf.save_model('./tabnet_model/tabnet_classifier')

# Оценка модели
preds = clf.predict(X_valid_vec)
print(f"Accuracy: {accuracy_score(y_valid, preds)}")
