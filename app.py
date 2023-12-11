from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import io
import base64

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename.endswith('.csv'):
            # Чтение данных из загруженного файла
            data = pd.read_csv(file)
            
            # Подготовка признаков и меток
            X = data["internet_activity"]
            y_gender = data["gender"]
            y_age = data["age"]

            # Разделение данных на обучающую и тестовую выборки
            X_train, X_test, y_gender_train, y_gender_test, y_age_train, y_age_test = train_test_split(X, y_gender, y_age, test_size=0.2, random_state=42)

            # Преобразование текстовых признаков в числовые с помощью мешка слов
            vectorizer = CountVectorizer()
            X_train_vectorized = vectorizer.fit_transform(X_train)
            X_test_vectorized = vectorizer.transform(X_test)

            # Обучение модели для определения пола с настройкой гиперпараметров
            gender_classifier = LogisticRegression()
            gender_param_grid = {'C': [0.1, 1, 10]}
            gender_grid_search = GridSearchCV(gender_classifier, gender_param_grid)
            gender_grid_search.fit(X_train_vectorized, y_gender_train)
            gender_best_model = gender_grid_search.best_estimator_

            # Предсказание пола для тестовых данных
            gender_predictions = gender_best_model.predict(X_test_vectorized)

            # Оценка точности предсказаний пола
            gender_accuracy = accuracy_score(y_gender_test, gender_predictions)

            # Визуализация результатов предсказания пола
            gender_labels = np.unique(y_gender_test)
            gender_cm = confusion_matrix(y_gender_test, gender_predictions, labels=gender_labels)
            gender_cm_df = pd.DataFrame(gender_cm, index=gender_labels, columns=gender_labels)

            # Генерация графика тепловой карты
            plt.figure(figsize=(8, 6))
            plt.title('Confusion Matrix - Gender Prediction')
            sns.heatmap(gender_cm_df, annot=True, cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')

            # Преобразование графика в изображение в формате base64
            img = io.BytesIO()
            plt.savefig(img, format='png', bbox_inches='tight')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Отправка результатов на страницу
            return render_template('results.html', accuracy=gender_accuracy, plot_url=plot_url)

    return "Invalid file format. Please upload a CSV file."

if __name__ == '__main__':
    app.run()
