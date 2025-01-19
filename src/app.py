from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io, json
app = Flask(__name__)

with open("../configs/config.json", "r") as file:
    config = json.load(file)

# Завантаження моделі
# Завантаження моделі
try:
    model1 = tf.keras.models.load_model(config['training']['final_model_dir'] + "VGG19.keras")
    model2 = tf.keras.models.load_model(config['training']['final_model_dir'] + "MobileNetV2.keras")
    model3 = tf.keras.models.load_model(config['training']['final_model_dir'] + "model15.01MobileNetV2.h5")  # Додайте назву вашої третьої моделі
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    print(f"Error loading models: {str(e)}")
    
class_names = config['data']['class_names']
print(class_names)

model_names = {
    "model1": "VGG19",
    "model2": "MobileNetV2 second",
    "model3": "MobileNetV2 first"
}

# Функція для обробки зображення
def preprocess_image(image):
    # Зміна розміру зображення до 256x256
    image = image.resize((256, 256))
    # Конвертація в масив numpy
    image = np.array(image)
    # Нормалізація пікселів до [0, 1]
    image = image / 255.0
    # Додавання розмірності для моделі (batch size = 1)
    image = np.expand_dims(image, axis=0)
    return image

# Маршрут для головної сторінки
@app.route("/")
def home():
    return render_template("index.html")

# Маршрут для передбачення
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Логування для перевірки
        print("New image received for prediction.")

        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        processed_image = preprocess_image(image)
        
        # Передбачення для всіх моделей
        predictions1 = model1.predict(processed_image)
        predictions2 = model2.predict(processed_image)
        predictions3 = model3.predict(processed_image)

        # Функція для отримання топ-3 результатів
        def get_top_3(predictions):
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            top_3_probabilities = predictions[0][top_3_indices]
            return [{
                "class": int(i),
                "class_name": class_names[i],
                "probability": float(p)
            } for i, p in zip(top_3_indices, top_3_probabilities)]

        results1 = get_top_3(predictions1)
        results2 = get_top_3(predictions2)
        results3 = get_top_3(predictions3)

        # Об'єднання результатів
        results = {
            "model1": {
                "name": model_names["model1"],
                "predictions": results1
            },
            "model2": {
                "name": model_names["model2"],
                "predictions": results2
            },
            "model3": {
                "name": model_names["model3"],
                "predictions": results3
            }
        }

        # Логування результатів
        print("Predictions:", results)

        # Відключення кешування
        response = jsonify(results)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)