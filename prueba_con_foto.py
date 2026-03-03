import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json
import os
from google.colab.patches import cv2_imshow  # For displaying images in Colab
from IPython.display import display, Javascript
from google.colab import output

# Cargar el modelo entrenado
try:
    model = load_model('garbage_classifier_model.h5')
    print("Modelo cargado correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    # No usamos sys.exit() para evitar errores en Colab

# Cargar el mapeo de clases
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Invertir el diccionario para obtener etiquetas a partir de índices
    class_labels = {v: k for k, v in class_indices.items()}
    print("Clases disponibles:", class_labels)
except Exception as e:
    print(f"Error al cargar el mapeo de clases: {e}")
    # Usar las clases definidas anteriormente como respaldo
    clases = ['battery', 'biological', 'cardboard', 'glass',
              'metal', 'paper', 'plastic', 'trash']
    class_labels = {i: clase for i, clase in enumerate(clases)}
    print("Usando clases predefinidas:", class_labels)

# Función para preprocesar la imagen
def preprocess_image(img):
    img = cv2.resize(img, (224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Función para clasificar una imagen desde archivo
def classify_image_from_file(image_path):
    try:
        # Cargar y preprocesar la imagen
        img = cv2.imread(image_path)
        if img is None:
            print(f"No se pudo cargar la imagen: {image_path}")
            return

        # Hacer una copia para mostrar
        display_img = img.copy()

        # Preprocesar
        processed_img = preprocess_image(img)

        # Realizar predicción
        prediction = model.predict(processed_img, verbose=0)
        predicted_class_index = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class_index] * 100

        # Obtener la etiqueta de la clase
        predicted_class = class_labels[predicted_class_index]

        # Mostrar resultado en la imagen
        label = f"{predicted_class}: {confidence:.2f}%"
        cv2.putText(display_img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Mostrar la imagen en Colab
        print(f"Predicción: {label}")
        cv2_imshow(display_img)

    except Exception as e:
        print(f"Error al clasificar la imagen: {e}")

# Opción 1: Clasificar imágenes de prueba del dataset
print("\n--- Clasificación de imágenes de prueba ---")
test_dir = os.path.join('garbage_dataset', 'test')

# Verificar si existe el directorio de prueba
if os.path.exists(test_dir):
    # Seleccionar algunas imágenes aleatorias para probar
    import random

    # Obtener todas las clases
    test_classes = os.listdir(test_dir)

    # Seleccionar una imagen aleatoria de cada clase
    for class_name in test_classes:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.isdir(class_dir):
            images = os.listdir(class_dir)
            if images:
                # Seleccionar una imagen aleatoria
                random_image = random.choice(images)
                image_path = os.path.join(class_dir, random_image)

                print(f"\nClasificando imagen de la clase '{class_name}': {random_image}")
                classify_image_from_file(image_path)
else:
    print("No se encontró el directorio de prueba. Asegúrate de que el dataset esté correctamente estructurado.")

# Opción 2: Subir una imagen para clasificar
from google.colab import files
print("\n--- Subir una imagen para clasificar ---")
print("Puedes subir una imagen para clasificarla.")

try:
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"\nClasificando imagen subida: {filename}")
        classify_image_from_file(filename)
except Exception as e:
    print(f"Error al subir o procesar la imagen: {e}")

print("\nNota: La cámara web no está disponible en Google Colab de forma estándar.")
print("Para usar la cámara web, necesitarías ejecutar este código en un entorno local o usar extensiones específicas de Colab.")