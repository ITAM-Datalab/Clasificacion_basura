import os
import shutil
import random
from google.colab import drive
from tqdm.notebook import tqdm
import zipfile
import requests
import io

# Montar Google Drive
drive.mount('/content/drive')

# Verificar si el dataset ya existe, si no, extraer el ZIP ya subido
original_dataset_dir = '/content/drive/My Drive/Colab Notebooks/Dataset'

if not os.path.exists(original_dataset_dir):
    print("Dataset no encontrado. Extrayendo del ZIP ya subido...")

    # Usar el archivo ZIP que ya fue subido
    zip_file = 'Dataset.zip'  # Nombre del archivo que se ve en la imagen

    if os.path.exists(zip_file):
        try:
            print(f"Extrayendo {zip_file}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall('/content/')
            print("Dataset extraído correctamente.")
        except Exception as e:
            print(f"Error al extraer el dataset: {e}")
            print("Por favor, asegúrate de que el archivo ZIP contiene la estructura correcta.")
            import sys
            sys.exit()
    else:
        print(f"No se encontró el archivo {zip_file}.")
        print("Por favor, asegúrate de que el archivo ZIP esté en el directorio actual.")
        import sys
        sys.exit()

# Verificar la estructura del dataset
if os.path.exists(original_dataset_dir):
    print("Estructura del dataset:")
    for item in os.listdir(original_dataset_dir):
        item_path = os.path.join(original_dataset_dir, item)
        if os.path.isdir(item_path):
            print(f"- {item}: {len(os.listdir(item_path))} archivos")
else:
    print("¡El dataset no se encuentra en la ruta esperada!")
    print("Por favor, asegúrate de que el dataset esté en '/content/Dataset' con las subcarpetas de clases.")
    # Detener la ejecución
    import sys
    sys.exit()

# Definir rutas
base_dir = '/content/garbage_dataset'

# Crear carpetas principales
os.makedirs(base_dir, exist_ok=True)
train_dir = os.path.join(base_dir, 'train')
os.makedirs(train_dir, exist_ok=True)
val_dir = os.path.join(base_dir, 'val')
os.makedirs(val_dir, exist_ok=True)
test_dir = os.path.join(base_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

# Listar las clases (obtener automáticamente de la estructura del dataset)
clases = [d for d in os.listdir(original_dataset_dir) if os.path.isdir(os.path.join(original_dataset_dir, d))]
print(f"Clases detectadas: {clases}")

# Contadores para estadísticas
total_imagenes = 0
imagenes_por_conjunto = {'train': 0, 'val': 0, 'test': 0}

for clase in clases:
    print(f"Procesando clase: {clase}")

    # Crear carpetas para cada conjunto
    os.makedirs(os.path.join(train_dir, clase), exist_ok=True)
    os.makedirs(os.path.join(val_dir, clase), exist_ok=True)
    os.makedirs(os.path.join(test_dir, clase), exist_ok=True)

    # Obtener rutas de las imágenes
    dir_clase = os.path.join(original_dataset_dir, clase)

    # Verificar si el directorio existe
    if not os.path.exists(dir_clase):
        print(f"¡Advertencia! El directorio {dir_clase} no existe. Saltando esta clase.")
        continue

    imagenes = os.listdir(dir_clase)
    random.shuffle(imagenes)

    # Calcular índices para división
    total = len(imagenes)
    total_imagenes += total
    train_end = int(0.7 * total)
    val_end = int(0.85 * total)

    # Dividir imágenes
    train_imagenes = imagenes[:train_end]
    val_imagenes = imagenes[train_end:val_end]
    test_imagenes = imagenes[val_end:]

    # Actualizar contadores
    imagenes_por_conjunto['train'] += len(train_imagenes)
    imagenes_por_conjunto['val'] += len(val_imagenes)
    imagenes_por_conjunto['test'] += len(test_imagenes)

    # Copiar imágenes a las carpetas correspondientes con barra de progreso
    print("Copiando imágenes de entrenamiento...")
    for img in tqdm(train_imagenes, desc=f"Train - {clase}"):
        src = os.path.join(dir_clase, img)
        dst = os.path.join(train_dir, clase, img)
        shutil.copyfile(src, dst)

    print("Copiando imágenes de validación...")
    for img in tqdm(val_imagenes, desc=f"Val - {clase}"):
        src = os.path.join(dir_clase, img)
        dst = os.path.join(val_dir, clase, img)
        shutil.copyfile(src, dst)

    print("Copiando imágenes de prueba...")
    for img in tqdm(test_imagenes, desc=f"Test - {clase}"):
        src = os.path.join(dir_clase, img)
        dst = os.path.join(test_dir, clase, img)
        shutil.copyfile(src, dst)

    print(f"  - {len(train_imagenes)} imágenes para entrenamiento")
    print(f"  - {len(val_imagenes)} imágenes para validación")
    print(f"  - {len(test_imagenes)} imágenes para prueba")

# Mostrar resumen final
print("\nResumen del dataset:")
print(f"Total de imágenes procesadas: {total_imagenes}")
print(f"Imágenes de entrenamiento: {imagenes_por_conjunto['train']} ({imagenes_por_conjunto['train']/total_imagenes*100:.1f}%)")
print(f"Imágenes de validación: {imagenes_por_conjunto['val']} ({imagenes_por_conjunto['val']/total_imagenes*100:.1f}%)")
print(f"Imágenes de prueba: {imagenes_por_conjunto['test']} ({imagenes_por_conjunto['test']/total_imagenes*100:.1f}%)")