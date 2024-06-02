import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Загрузка модели
clf = TabNetClassifier()
clf.load_model('./tabnet_model/tabnet_classifier.zip')

# Преобразование меток в числовые значения (должен соответствовать обучающим данным)
label_encoder = LabelEncoder()
label_encoder.fit(["Not detection", "alert"])  # Обновите это, чтобы соответствовать вашим меткам

# Загрузка векторизатора
with open('./tabnet_model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Функция для предсказания на входном тексте
def predict(input_text):
    lines = input_text.strip().split('\n')
    count = int(lines[0])
    combined_text = ' '.join([line.strip() for line in lines[1:]])

    # Преобразование текстовых данных в числовые векторы
    X_input_vec = vectorizer.transform([combined_text]).toarray()

    # Предсказание
    preds = clf.predict(X_input_vec)
    pred_labels = label_encoder.inverse_transform(preds)

    return pred_labels

# Входные данные
input_text = """
3
Model 2|train|81.73|32.9521484375|18.3690185546875|1244.64697265625|622.8226318359375
Model 3|rail - v1 2024-04-27 1-51pm|30.29|1135.7283935546875|241.85845947265625|124.723876953125|207.11376953125
Model 3|-|28.62|1098.089111328125|269.76422119140625|181.357421875|344.09869384765625
"""

# Получение предсказаний
predictions = predict(input_text)
print(predictions)
