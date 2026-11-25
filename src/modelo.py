"""Datos, aumento y armado del modelo. Lo usan la etapa 1 y la etapa 2.

Las redes de Keras que usamos (MobileNetV3, EfficientNet) normalizan por
dentro: reciben píxeles RGB en 0-255 (float32), SIN dividir entre 255.
"""

import csv
import json
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from config import CLASES, DIR_DATOS, DIR_IMAGENES, SEMILLA, TAM_IMAGEN

BASES = {
    "MobileNetV3Large": keras.applications.MobileNetV3Large,
    "EfficientNetB0": keras.applications.EfficientNetB0,
    "EfficientNetV2B0": keras.applications.EfficientNetV2B0,
}


# ---------------------------------------------------------------- datos

def leer_split(particion: str, dir_datos: Path = DIR_DATOS):
    """Devuelve (rutas_relativas, etiquetas) de una partición de splits.csv."""
    rutas, etiquetas = [], []
    with open(Path(dir_datos) / "splits.csv", encoding="utf-8") as f:
        for fila in csv.DictReader(f):
            if fila["particion"] == particion:
                rutas.append(fila["ruta"])
                etiquetas.append(CLASES.index(fila["clase"]))
    return rutas, etiquetas


def leer_class_weights(dir_datos: Path = DIR_DATOS) -> dict[int, float]:
    with open(Path(dir_datos) / "class_weights.json", encoding="utf-8") as f:
        return {int(k): v for k, v in json.load(f).items()}


def cargar_imagen(ruta: tf.Tensor) -> tf.Tensor:
    """Lee un JPEG y lo pasa a 224x224 RGB en uint8 (0-255).

    Se guarda en uint8 para que el caché quepa en la RAM de Colab
    (~1.3 GB para train) y se convierte a float32 justo antes del modelo.
    """
    img = tf.io.decode_jpeg(tf.io.read_file(ruta), channels=3)
    img = tf.image.resize(img, TAM_IMAGEN, antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def aumento_datos() -> keras.Sequential:
    """Rotaciones, giros, zoom y brillo. Solo se aplica a train."""
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=SEMILLA),
        layers.RandomRotation(0.25, fill_mode="reflect", seed=SEMILLA),
        layers.RandomZoom(0.2, fill_mode="reflect", seed=SEMILLA),
        layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEMILLA),
    ], name="aumento_datos")


def crear_dataset(particion: str, batch_size: int = 32,
                  dir_imagenes: Path = DIR_IMAGENES,
                  dir_datos: Path = DIR_DATOS,
                  limite: int | None = None) -> tf.data.Dataset:
    """tf.data.Dataset de (imagen, etiqueta one-hot) para una partición.

    En train se barajan los datos y se aplica el aumento. `limite` sirve
    para pruebas rápidas con pocas imágenes.
    """
    rutas, etiquetas = leer_split(particion, dir_datos)
    if limite:
        rutas, etiquetas = rutas[:limite], etiquetas[:limite]
    rutas = [str(Path(dir_imagenes) / r) for r in rutas]

    ds = tf.data.Dataset.from_tensor_slices((rutas, etiquetas))
    ds = ds.map(lambda r, y: (cargar_imagen(r), tf.one_hot(y, len(CLASES))),
                num_parallel_calls=tf.data.AUTOTUNE).cache()
    if particion == "train":
        ds = ds.shuffle(len(rutas), seed=SEMILLA,
                        reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y))
    if particion == "train":
        aum = aumento_datos()
        ds = ds.map(lambda x, y: (aum(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- modelo

def construir_modelo(nombre_base: str, pesos: str | None = "imagenet",
                     lr: float = 1e-3) -> keras.Model:
    """Entrada 224x224x3 (0-255) -> base congelada -> GAP -> Dense(256)
    -> Dropout(0.5) -> Dense(8, softmax). Ya compilado para la etapa 1."""
    tf.keras.utils.set_random_seed(SEMILLA)
    base = BASES[nombre_base](include_top=False, weights=pesos,
                              input_shape=(*TAM_IMAGEN, 3))
    base.trainable = False

    entrada = keras.Input(shape=(*TAM_IMAGEN, 3), name="imagen")
    x = base(entrada, training=False)  # BatchNorm siempre en inferencia
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    salida = layers.Dense(len(CLASES), activation="softmax",
                          name="clase")(x)
    modelo = keras.Model(entrada, salida, name=f"residuos_{nombre_base}")
    compilar(modelo, lr)
    return modelo


def compilar(modelo: keras.Model, lr: float) -> None:
    modelo.compile(optimizer=keras.optimizers.Adam(lr),
                   loss="categorical_crossentropy",
                   metrics=["accuracy"])


def descongelar_ultimas(modelo: keras.Model, n_capas: int,
                        lr: float = 1e-5) -> None:
    """Etapa 2 (fine-tuning): descongela las últimas `n_capas` de la base,
    deja las BatchNormalization congeladas y recompila con lr bajo."""
    base = next(c for c in modelo.layers if isinstance(c, keras.Model))
    base.trainable = True
    for capa in base.layers[:-n_capas]:
        capa.trainable = False
    for capa in base.layers[-n_capas:]:
        if isinstance(capa, layers.BatchNormalization):
            capa.trainable = False
    compilar(modelo, lr)


def callbacks(ruta_modelo: str, paciencia: int = 5,
              umbral_inicial: float | None = None) -> list:
    """Early stopping y checkpoint sobre val_loss.

    `umbral_inicial`: el checkpoint solo se guarda si val_loss baja de este
    valor. En el fine-tuning se pasa el val_loss del modelo de partida, para
    no guardar un modelo peor que el que ya teníamos.
    """
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=paciencia,
                                      restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(ruta_modelo, monitor="val_loss",
                                        save_best_only=True,
                                        initial_value_threshold=umbral_inicial),
    ]
