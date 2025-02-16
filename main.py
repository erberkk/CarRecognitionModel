import os
import numpy as np
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import layers, models, callbacks
import tempfile
import uuid

app = Flask(__name__)

MODEL_DIR = './saved_model'
MODEL_PATH = os.path.join(MODEL_DIR, 'model.keras')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.npy')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
THRESHOLD = 70.0

model = None
class_names = None

CLASS_TRANSLATIONS = {
    "chevrolet_silverado_2004": "Chevrolet Silverado 2004",
    "dodge_grand caravan_2005": "Dodge Grand Caravan 2005",
    "ford_explorer_2003": "Ford Explorer 2003",
    "ford_explorer_2004": "Ford Explorer 2004",
    "ford_mustang_2000": "Ford Mustang 2000",
    "honda_civic_2002": "Honda Civic 2002",
    "nissan_altima_2002": "Nissan Altima 2002",
    "nissan_altima_2003": "Nissan Altima 2003",
    "nissan_altima_2005": "Nissan Altima 2005",
    "non-car": "Non-Car",
}


def create_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = True

    for layer in base_model.layers[:-10]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def train_and_save_model():
    dataset_path = './dataset'

    train_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        dataset_path,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    num_classes = len(class_names)
    model = create_model(num_classes)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    checkpoint_cb = callbacks.ModelCheckpoint(
        MODEL_PATH,
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    early_stopping_cb = callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr_cb = callbacks.ReduceLROnPlateau(patience=3, factor=0.2, verbose=1)

    class_weights = {
        i: 2.0 if class_name != "non-car" else 1.0 for i, class_name in enumerate(class_names)
    }

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        class_weight=class_weights,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    )

    np.save(CLASS_NAMES_PATH, class_names)


def load_model_and_classes():
    global model, class_names
    if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = np.load(CLASS_NAMES_PATH)
        print("Kayıtlı model ve sınıflar yüklendi.")
    else:
        print("Kayıtlı model bulunamadı, eğitim başlatılıyor...")
        train_and_save_model()
        model = tf.keras.models.load_model(MODEL_PATH)
        class_names = np.load(CLASS_NAMES_PATH)
        print("Model eğitildi ve yüklendi.")


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "Lütfen bir resim yükleyin."

        file = request.files['image']
        if file.filename == '':
            return "Geçerli bir resim dosyası seçiniz."

        temp_path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()) + file.filename)
        file.save(temp_path)

        img = tf.keras.preprocessing.image.load_img(temp_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        preds = model.predict(img_array)
        preds = preds[0]
        predicted_idx = np.argmax(preds)
        predicted_class = translate_class_name(class_names[predicted_idx])
        predicted_prob = preds[predicted_idx] * 100

        class_probabilities = {
            translate_class_name(cn): f"{prob*100:.2f}%" for cn, prob in zip(class_names, preds)
        }

        if predicted_prob < THRESHOLD:
            return render_template('index.html',
                                   predicted_class="Model bu görüntüyü yeterince güvenli tahmin edemedi.",
                                   predicted_prob=predicted_prob,
                                   class_probabilities=class_probabilities,
                                   show_result=True,
                                   is_unsure=True)
        else:
            return render_template('index.html',
                                   predicted_class=predicted_class,
                                   predicted_prob=predicted_prob,
                                   class_probabilities=class_probabilities,
                                   show_result=True,
                                   is_unsure=False)
    else:
        return render_template('index.html', show_result=False, is_unsure=False)


def translate_class_name(class_name):
    return CLASS_TRANSLATIONS.get(class_name, class_name)


if __name__ == '__main__':
    load_model_and_classes()
    app.run(debug=True)
