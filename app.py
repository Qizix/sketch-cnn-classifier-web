from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
app = Flask(__name__)

# Завантаження моделі
try:
    model = tf.keras.models.load_model("model/model15.01MobileNetV2.h5")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    
class_names = ['airplane','alarm clock','angel','ant','apple','arm','armchair','ashtray','axe','backpack','banana','barn','baseball bat','basket','bathtub','bear (animal)','bed','bee','beer-mug','bell','bench','bicycle','binoculars','blimp','book','bookshelf','boomerang','bottle opener','bowl','brain','bread','bridge','bulldozer','bus','bush','butterfly','cabinet','cactus','cake','calculator','camel','camera','candle','cannon','canoe','car (sedan)','carrot','castle','cat','cell phone','chair','chandelier','church','cigarette','cloud','comb','computer monitor','computer-mouse','couch','cow','crab','crane (machine)','crocodile','crown','cup','diamond','dog','dolphin','donut','door','door handle','dragon','duck','ear','elephant','envelope','eye','eyeglasses','face','fan','feather','fire hydrant','fish','flashlight','floor lamp','flower with stem','flying bird','flying saucer','foot','fork','frog','frying-pan','giraffe','grapes','grenade','guitar','hamburger','hammer','hand','harp','hat','head','head-phones','hedgehog','helicopter','helmet','horse','hot air balloon','hot-dog','hourglass','house','human-skeleton','ice-cream-cone','ipod','kangaroo','key','keyboard','knife','ladder','laptop','leaf','lightbulb','lighter','lion','lobster','loudspeaker','mailbox','megaphone','mermaid','microphone','microscope','monkey','moon','mosquito','motorbike','mouse (animal)','mouth','mug','mushroom','nose','octopus','owl','palm tree','panda','paper clip','parachute','parking meter','parrot','pear','pen','penguin','person sitting','person walking','piano','pickup truck','pig','pigeon','pineapple','pipe (for smoking)','pizza','potted plant','power outlet','present','pretzel','pumpkin','purse','rabbit','race car','radio','rainbow','revolver','rifle','rollerblades','rooster','sailboat','santa claus','satellite','satellite dish','saxophone','scissors','scorpion','screwdriver','sea turtle','seagull','shark','sheep','ship','shoe','shovel','skateboard','skull','skyscraper','snail','snake','snowboard','snowman','socks','space shuttle','speed-boat','spider','sponge bob','spoon','squirrel','standing bird','stapler','strawberry','streetlight','submarine','suitcase','sun','suv','swan','sword','syringe','t-shirt','table','tablelamp','teacup','teapot','teddy-bear','telephone','tennis-racket','tent','tiger','tire','toilet','tomato','tooth','toothbrush','tractor','traffic light','train','tree','trombone','trousers','truck','trumpet','tv','umbrella','van','vase','violin','walkie talkie','wheel','wheelbarrow','windmill','wine-bottle','wineglass','wrist-watch','zebra']

# Функція для обробки зображення
def preprocess_image(image):
    # Зміна розміру зображення до 128x128
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
        predictions = model.predict(processed_image)
        top_10_indices = np.argsort(predictions[0])[-10:][::-1]
        top_10_probabilities = predictions[0][top_10_indices]

        # Додавання назв класів до результатів
        results = [{
            "class": int(i),
            "class_name": class_names[i],  # Назва класу
            "probability": float(p)
        } for i, p in zip(top_10_indices, top_10_probabilities)]

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