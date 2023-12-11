from flask import Flask, render_template, request, send_file
import pandas as pd
import seaborn as sns
import base64
import io
import os

app = Flask(__name__)

# Генерация предсказания на основе данных пользователя
def generate_prediction(user_data, accuracy_setting):
    # Ваш код для генерации предсказания на основе данных пользователя
    # Пример: просто возвращаем строку "Мужчина, Возраст: 30"
    prediction = f"Мужчина, Возраст: {accuracy_setting}"
    return prediction

# Генерация графика тепловой карты на основе предсказания
def generate_heatmap_plot(prediction):
    # Ваш код для генерации графика тепловой карты
    # Пример: создаем фиктивный DataFrame и строим тепловую карту
    data = {'Мужчина': [0.8, 0.2], 'Женщина': [0.2, 0.8]}
    df = pd.DataFrame(data, index=['Молодой', 'Пожилой'])
    sns.heatmap(df, annot=True, cmap='YlGnBu')
    plot = sns.plt.gcf()
    return plot

# Расчет точности предсказаний
def calculate_accuracy(prediction):
    # Ваш код для расчета точности предсказаний
    # Пример: просто возвращаем значение 0.85 в качестве точности
    accuracy = 0.85
    return accuracy

# Проверка наличия модели
def check_model():
    return os.path.exists('models/model.pkl')

# Удаление модели
def delete_model():
    os.remove('models/model.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Получение загруженного файла
    file = request.files['file']
    # Загрузка и обработка данных из файла
    df = pd.read_csv(file)
    user_data = df.values.tolist()
    # Получение выбранного уровня точности
    accuracy_setting = request.form['accuracy']
    # Генерация предсказания на основе данных файла и выбранного уровня точности
    prediction = generate_prediction(user_data, accuracy_setting)
    # Генерация графика тепловой карты
    plot = generate_heatmap_plot(prediction)
    # Конвертация графика в изображение в формате base64
    buffer = io.BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.getvalue()).decode()
    # Расчет точности предсказаний
    accuracy = calculate_accuracy(prediction)
    # Отображение результатов
    return render_template('results.html', accuracy=accuracy, plot_url=plot_image)

@app.route('/download-model', methods=['GET'])
def download_model():
    return send_file('models/model.pkl', as_attachment=True, attachment_filename='model.pkl')

@app.route('/reset-model', methods=['GET'])
def reset_model():
    if check_model():
        delete_model()
        return "Модель успешно сброшена и удалена."
    else:
        return "Модель уже отсутствует."

if __name__ == '__main__':
    app.run()
