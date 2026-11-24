"""Exporta el modelo final a TensorFlow Lite y verifica la conversión.

Parte del checkpoint del fine-tuning y genera:

    resultados/tflite/modelo.tflite    el modelo convertido (no se sube a git)
    resultados/tflite/etiquetas.json   clases, contenedores y cómo preparar la
                                       entrada; lo lee src/inferir.py
    resultados/tflite/resumen.json     tamaño, velocidad y comparación con Keras

Cuantización (--cuantizacion), medida con el modelo final en val y test:
    float16   pesos en float16 (por defecto). 6.6 MB, 2.3 ms/imagen en CPU;
              cambia la clase de 1-2 imágenes de ~1,865 frente a Keras y deja
              las métricas iguales.
    dinamica  pesos en int8 (la que se usó en la prueba de redes). 3.6 MB pero
              4.8 ms/imagen; cambia la clase de 24-26 imágenes y baja la
              accuracy de test de 0.947 a 0.943.
    ninguna   float32, idéntico a Keras. 12.9 MB, 2.2 ms/imagen.
En los tres casos la entrada sigue siendo float32 con píxeles RGB en 0-255
(SIN dividir entre 255): MobileNetV3 normaliza por dentro.

Verificación: se pasan las mismas imágenes de validación y test por Keras y
por TFLite y se compara la clase predicha y las probabilidades. También se
mide el tiempo por imagen en CPU (batch 1).

Uso (en WSL con el entorno activado, desde la raíz del repo):
    python src/exportar_tflite.py
    python src/exportar_tflite.py --cuantizacion float16
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from tensorflow import keras

from config import (CLASES, CLASES_A_CONTENEDORES, DIR_DATOS, DIR_IMAGENES,
                    TAM_IMAGEN)
from evaluar import A_CONTENEDOR
from modelo import crear_dataset


def convertir(modelo: keras.Model, cuantizacion: str) -> bytes:
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    if cuantizacion in ("dinamica", "float16"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if cuantizacion == "float16":
        conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def probs_tflite(interp: tf.lite.Interpreter, x: np.ndarray) -> np.ndarray:
    """Probabilidades de TFLite para un lote, una imagen a la vez."""
    ent = interp.get_input_details()[0]["index"]
    sal = interp.get_output_details()[0]["index"]
    salida = []
    for img in x:
        interp.set_tensor(ent, img[None].astype(np.float32))
        interp.invoke()
        salida.append(interp.get_tensor(sal)[0])
    return np.array(salida)


def comparar(modelo: keras.Model, interp: tf.lite.Interpreter,
             particion: str, **kw) -> dict:
    """Keras contra TFLite sobre la misma entrada ya preprocesada."""
    ys, pk, pt = [], [], []
    for x, y in crear_dataset(particion, **kw):
        x = x.numpy()
        ys.append(y.numpy().argmax(1))
        pk.append(modelo.predict_on_batch(x))
        pt.append(probs_tflite(interp, x))
    y, pk, pt = np.concatenate(ys), np.concatenate(pk), np.concatenate(pt)
    ck, ct = pk.argmax(1), pt.argmax(1)
    distintas = np.flatnonzero(ck != ct)
    return {
        "n_imagenes": int(len(y)),
        "misma_clase_que_keras": round(float((ck == ct).mean()), 4),
        "imagenes_con_otra_clase": int(len(distintas)),
        "dif_max_probabilidad": round(float(np.abs(pk - pt).max()), 4),
        "keras": {"accuracy": round(accuracy_score(y, ck), 4),
                  "f1_macro": round(f1_score(y, ck, average="macro"), 4),
                  "mcc": round(float(matthews_corrcoef(y, ck)), 4)},
        "tflite": {"accuracy": round(accuracy_score(y, ct), 4),
                   "f1_macro": round(f1_score(y, ct, average="macro"), 4),
                   "mcc": round(float(matthews_corrcoef(y, ct)), 4),
                   "accuracy_contenedor": round(accuracy_score(
                       A_CONTENEDOR[y], A_CONTENEDOR[ct]), 4)},
    }


def medir_ms(interp: tf.lite.Interpreter, imagenes: np.ndarray,
             repeticiones: int = 100) -> float:
    ent = interp.get_input_details()[0]["index"]
    for img in imagenes[:5]:  # calentamiento
        interp.set_tensor(ent, img[None])
        interp.invoke()
    t0 = time.perf_counter()
    for i in range(repeticiones):
        interp.set_tensor(ent, imagenes[i % len(imagenes)][None])
        interp.invoke()
    return (time.perf_counter() - t0) / repeticiones * 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modelo", type=Path,
                    default=Path("resultados/fine_tuning/"
                                 "modelo_fine_tuning.keras"))
    ap.add_argument("--cuantizacion", default="float16",
                    choices=["float16", "dinamica", "ninguna"])
    ap.add_argument("--salida", type=Path, default=Path("resultados/tflite"))
    ap.add_argument("--hilos", type=int, default=4,
                    help="hilos de CPU para medir el tiempo de TFLite")
    ap.add_argument("--imagenes", type=Path, default=DIR_IMAGENES)
    ap.add_argument("--datos", type=Path, default=DIR_DATOS)
    ap.add_argument("--limite", type=int, default=None,
                    help="usar solo N imágenes por partición (para probar)")
    args = ap.parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    modelo = keras.models.load_model(args.modelo)
    ruta_tfl = args.salida / "modelo.tflite"
    ruta_tfl.write_bytes(convertir(modelo, args.cuantizacion))
    mb = ruta_tfl.stat().st_size / 1e6
    print(f"TFLite ({args.cuantizacion}): {ruta_tfl} ({mb:.1f} MB)")

    interp = tf.lite.Interpreter(model_path=str(ruta_tfl),
                                 num_threads=args.hilos)
    interp.allocate_tensors()
    ent = interp.get_input_details()[0]
    sal = interp.get_output_details()[0]

    etiquetas = {
        "clases": CLASES,
        "clases_a_contenedores": CLASES_A_CONTENEDORES,
        "entrada": {
            "alto": TAM_IMAGEN[0], "ancho": TAM_IMAGEN[1], "canales": 3,
            "orden_canales": "RGB",
            "tipo": np.dtype(ent["dtype"]).name,
            "rango": [0, 255],
            "normalizacion": "ninguna: NO dividir entre 255; el modelo "
                             "normaliza por dentro",
        },
        "salida": "probabilidades (softmax) en el orden de 'clases'",
    }
    with open(args.salida / "etiquetas.json", "w", encoding="utf-8") as f:
        json.dump(etiquetas, f, indent=2, ensure_ascii=False)

    kw = dict(dir_imagenes=args.imagenes, dir_datos=args.datos,
              limite=args.limite)
    print("Comparando Keras y TFLite en val y test...")
    comparacion = {p: comparar(modelo, interp, p, **kw)
                   for p in ("val", "test")}
    muestra = np.concatenate(
        [x.numpy() for x, _ in crear_dataset("val", **kw).take(2)])

    resumen = {
        "modelo_keras": args.modelo.as_posix(),
        "cuantizacion": args.cuantizacion,
        "tflite_mb": round(mb, 2),
        "keras_mb": round(args.modelo.stat().st_size / 1e6, 2),
        "ms_por_imagen_cpu": round(medir_ms(interp, muestra), 1),
        "hilos": args.hilos,
        "entrada": {"forma": [int(d) for d in ent["shape"]],
                    "tipo": np.dtype(ent["dtype"]).name},
        "salida": {"forma": [int(d) for d in sal["shape"]],
                   "tipo": np.dtype(sal["dtype"]).name},
        "comparacion": comparacion,
    }
    with open(args.salida / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    print(f"\nArchivos en {args.salida}/")


if __name__ == "__main__":
    main()
