// Отримання елементів DOM
const canvas = document.getElementById("sketchCanvas");
const ctx = canvas.getContext("2d");
let isDrawing = false;

// Список доступних класів (приклад)
const classNames = ["airplane","alarm clock","angel","ant","apple","arm","armchair","ashtray","axe","backpack","banana","barn","baseball bat","basket","bathtub","bear (animal)","bed","bee","beer-mug","bell","bench","bicycle","binoculars","blimp","book","bookshelf","boomerang","bottle opener","bowl","brain","bread","bridge","bulldozer","bus","bush","butterfly","cabinet","cactus","cake","calculator","camel","camera","candle","cannon","canoe","car (sedan)","carrot","castle","cat","cell phone","chair","chandelier","church","cigarette","cloud","comb","computer monitor","computer-mouse","couch","cow","crab","crane (machine)","crocodile","crown","cup","diamond","dog","dolphin","donut","door","door handle","dragon","duck","ear","elephant","envelope","eye","eyeglasses","face","fan","feather","fire hydrant","fish","flashlight","floor lamp","flower with stem","flying bird","flying saucer","foot","fork","frog","frying-pan","giraffe","grapes","grenade","guitar","hamburger","hammer","hand","harp","hat","head","head-phones","hedgehog","helicopter","helmet","horse","hot air balloon","hot-dog","hourglass","house","human-skeleton","ice-cream-cone","ipod","kangaroo","key","keyboard","knife","ladder","laptop","leaf","lightbulb","lighter","lion","lobster","loudspeaker","mailbox","megaphone","mermaid","microphone","microscope","monkey","moon","mosquito","motorbike","mouse (animal)","mouth","mug","mushroom","nose","octopus","owl","palm tree","panda","paper clip","parachute","parking meter","parrot","pear","pen","penguin","person sitting","person walking","piano","pickup truck","pig","pigeon","pineapple","pipe (for smoking)","pizza","potted plant","power outlet","present","pretzel","pumpkin","purse","rabbit","race car","radio","rainbow","revolver","rifle","rollerblades","rooster","sailboat","santa claus","satellite","satellite dish","saxophone","scissors","scorpion","screwdriver","sea turtle","seagull","shark","sheep","ship","shoe","shovel","skateboard","skull","skyscraper","snail","snake","snowboard","snowman","socks","space shuttle","speed-boat","spider","sponge bob","spoon","squirrel","standing bird","stapler","strawberry","streetlight","submarine","suitcase","sun","suv","swan","sword","syringe","t-shirt","table","tablelamp","teacup","teapot","teddy-bear","telephone","tennis-racket","tent","tiger","tire","toilet","tomato","tooth","toothbrush","tractor","traffic light","train","tree","trombone","trousers","truck","trumpet","tv","umbrella","van","vase","violin","walkie talkie","wheel","wheelbarrow","windmill","wine-bottle","wineglass","wrist-watch","zebra"];

// Заповнення списку доступних класів у вигляді рядка
const classList = document.getElementById("classList");
classList.textContent = classNames.join(", ");

// Змінна для обмеження частоти викликів
let lastPredictionTime = 0;
const predictionInterval = 200; // Інтервал між передбаченнями (200 мс)

// Функція для автоматичного передбачення
function autoPredict() {
    const now = Date.now();
    if (now - lastPredictionTime >= predictionInterval) {
        predictSketch(); // Викликаємо передбачення
        lastPredictionTime = now; // Оновлюємо час останнього передбачення
    }
}

// Подія початку малювання
canvas.addEventListener("mousedown", (event) => {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(event.offsetX, event.offsetY);
    autoPredict(); // Викликаємо передбачення при початку малювання
});

// Подія малювання
canvas.addEventListener("mousemove", (event) => {
    if (isDrawing) {
        ctx.lineTo(event.offsetX, event.offsetY);
        ctx.stroke();
        autoPredict(); // Викликаємо автоматичне передбачення під час малювання
    }
});

// Подія завершення малювання
canvas.addEventListener("mouseup", () => {
    isDrawing = false;
    ctx.closePath();
    autoPredict(); // Викликаємо автоматичне передбачення після завершення малювання
});

// Подія виходу курсора за межі Canvas
canvas.addEventListener("mouseout", () => {
    isDrawing = false;
    ctx.closePath();
});

// Функція для очищення Canvas
function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#FFF"; // Темний фон для Canvas
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

// Функція для передбачення
function predictSketch() {
    // Отримання зображення з Canvas
    canvas.toBlob((blob) => {
        // Створення FormData для відправки файлу
        const formData = new FormData();
        formData.append("image", blob, "sketch.png");

        // Відправка на сервер
        fetch("/predict", {
            method: "POST",
            body: formData,
        })
            .then((response) => response.json())
            .then((data) => {
                // Оновлення результатів для першої моделі
                updatePredictions(data.model1.predictions, "predictionsBag1");
                document.getElementById("model1Name").textContent = data.model1.name;
    
                // Оновлення результатів для другої моделі
                updatePredictions(data.model2.predictions, "predictionsBag2");
                document.getElementById("model2Name").textContent = data.model2.name;
            })
            .catch((error) => {
                console.error("Error:", error);
                alert("An error occurred. Please check the console for details.");
            });
    });
}
function updatePredictions(predictions, elementId) {
    const predictionsBag = document.getElementById(elementId);
    
    // Оновлення існуючих або створення нових бар-плотів
    predictions.forEach((item, index) => {
        let bar = predictionsBag.querySelector(`.bar:nth-child(${index + 1})`);
        
        if (!bar) {
            bar = document.createElement("div");
            bar.className = "bar";
            predictionsBag.appendChild(bar);

            const barLabel = document.createElement("div");
            barLabel.className = "bar-label";
            bar.appendChild(barLabel);

            const barProgress = document.createElement("div");
            barProgress.className = "bar-progress";
            const barProgressFill = document.createElement("div");
            barProgressFill.className = "bar-progress-fill";
            barProgress.appendChild(barProgressFill);
            bar.appendChild(barProgress);

            const barPercentage = document.createElement("div");
            barPercentage.className = "bar-percentage";
            bar.appendChild(barPercentage);
        }

        // Оновлення даних бар-плоту
        bar.querySelector('.bar-label').textContent = item.class_name;
        bar.querySelector('.bar-progress-fill').style.width = `${item.probability * 100}%`;
        bar.querySelector('.bar-percentage').textContent = `${(item.probability * 100).toFixed(2)}%`;
    });

    // Видалення зайвих бар-плотів, якщо такі є
    while (predictionsBag.children.length > predictions.length) {
        predictionsBag.removeChild(predictionsBag.lastChild);
    }
}

// Очищення Canvas при завантаженні сторінки
clearCanvas();