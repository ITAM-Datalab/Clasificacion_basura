"""Prueba corta para elegir la red preentrenada (etapa 1, paso 5).

Entrena cada candidata SOLO con la base congelada y pocas épocas, sobre el
mismo split, y compara en VALIDACIÓN (test se reserva para el modelo final):
  - F1 macro y F1 por clase (en especial biological)
  - accuracy y F1 macro a nivel de los 3 contenedores
  - tamaño y tiempo de inferencia del modelo en TensorFlow Lite (CPU)

Uso (en Colab con GPU, desde la raíz del repo y después de preparar_datos):
    python src/prueba_redes.py
    python src/prueba_redes.py --bases MobileNetV3Large EfficientNetB0 --epocas 8

Resultados en `resultados/prueba_redes/`.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402

from config import CLASES, CLASES_A_CONTENEDORES, DIR_DATOS, DIR_IMAGENES  # noqa: E402
from modelo import (callbacks, construir_modelo, crear_dataset,  # noqa: E402
                    leer_class_weights)

CONTENEDORES = sorted(set(CLASES_A_CONTENEDORES.values()))
A_CONTENEDOR = np.array([CONTENEDORES.index(CLASES_A_CONTENEDORES[c])
                         for c in CLASES])


def a_tflite(modelo: tf.keras.Model, ruta: Path) -> Path:
    """Convierte a TFLite con cuantización de rango dinámico (pesos int8)."""
    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    ruta.write_bytes(conv.convert())
    return ruta


def medir_tflite(ruta: Path, imagenes: np.ndarray, hilos: int = 4,
                 repeticiones: int = 50) -> float:
    """Milisegundos promedio por imagen (batch 1) en CPU."""
    interp = tf.lite.Interpreter(model_path=str(ruta), num_threads=hilos)
    interp.allocate_tensors()
    ent = interp.get_input_details()[0]["index"]
    for img in imagenes[:5]:  # calentamiento
        interp.set_tensor(ent, img[None])
        interp.invoke()
    t0 = time.perf_counter()
    for i in range(repeticiones):
        interp.set_tensor(ent, imagenes[i % len(imagenes)][None])
        interp.invoke()
    return (time.perf_counter() - t0) / repeticiones * 1000


def graficar(historia: dict, nombre: str, salida: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(10, 4))
    for eje, m in zip(ejes, ("loss", "accuracy")):
        eje.plot(historia[m], label="train")
        eje.plot(historia[f"val_{m}"], label="val")
        eje.set_title(f"{nombre}: {m}")
        eje.set_xlabel("época")
        eje.legend()
    fig.tight_layout()
    fig.savefig(salida / f"curvas_{nombre}.png", dpi=120)
    plt.close(fig)


def probar(nombre: str, args, ds_train, ds_val, y_val, pesos_clase,
           salida: Path) -> dict:
    print(f"\n===== {nombre} =====")
    modelo = construir_modelo(nombre, pesos=args.pesos, lr=args.lr)
    ruta_keras = salida / f"{nombre}.keras"

    t0 = time.time()
    hist = modelo.fit(ds_train, validation_data=ds_val, epochs=args.epocas,
                      class_weight=pesos_clase,
                      callbacks=callbacks(str(ruta_keras), paciencia=3),
                      verbose=2)
    minutos = (time.time() - t0) / 60
    graficar(hist.history, nombre, salida)

    prob = modelo.predict(ds_val, verbose=0)
    y_pred = prob.argmax(1)
    f1_clases = f1_score(y_val, y_pred, average=None,
                         labels=range(len(CLASES)))
    c_val, c_pred = A_CONTENEDOR[y_val], A_CONTENEDOR[y_pred]

    imgs = np.concatenate([x.numpy() for x, _ in ds_val.take(2)])
    ruta_tfl = a_tflite(modelo, salida / f"{nombre}.tflite")

    res = {
        "red": nombre,
        "epocas": len(hist.history["loss"]),
        "min_entrenamiento": round(minutos, 1),
        "val_accuracy": round(accuracy_score(y_val, y_pred), 4),
        "val_f1_macro": round(f1_score(y_val, y_pred, average="macro"), 4),
        "val_f1_biological": round(float(f1_clases[CLASES.index("biological")]), 4),
        "val_acc_contenedor": round(accuracy_score(c_val, c_pred), 4),
        "val_f1_macro_contenedor": round(
            f1_score(c_val, c_pred, average="macro"), 4),
        "tflite_mb": round(ruta_tfl.stat().st_size / 1e6, 1),
        "tflite_ms_cpu": round(medir_tflite(ruta_tfl, imgs), 1),
        "f1_por_clase": {c: round(float(f), 4)
                         for c, f in zip(CLASES, f1_clases)},
    }
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--bases", nargs="+",
                    default=["MobileNetV3Large", "EfficientNetB0"])
    ap.add_argument("--epocas", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imagenes", type=Path, default=DIR_IMAGENES)
    ap.add_argument("--datos", type=Path, default=DIR_DATOS)
    ap.add_argument("--salida", type=Path,
                    default=Path("resultados/prueba_redes"))
    ap.add_argument("--limite", type=int, default=None,
                    help="usar solo N imágenes por partición (para probar)")
    ap.add_argument("--pesos", default="imagenet",
                    help="'imagenet' o 'none' (sin pesos, solo para probar)")
    args = ap.parse_args()
    args.pesos = None if args.pesos == "none" else args.pesos
    args.salida.mkdir(parents=True, exist_ok=True)

    kw = dict(batch_size=args.batch, dir_imagenes=args.imagenes,
              dir_datos=args.datos, limite=args.limite)
    ds_train = crear_dataset("train", **kw)
    ds_val = crear_dataset("val", **kw)
    y_val = np.concatenate([y.numpy().argmax(1) for _, y in ds_val])
    pesos_clase = leer_class_weights(args.datos)

    resultados = [probar(b, args, ds_train, ds_val, y_val, pesos_clase,
                         args.salida) for b in args.bases]

    columnas = [k for k in resultados[0] if k != "f1_por_clase"]
    with open(args.salida / "resumen.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columnas, extrasaction="ignore")
        w.writeheader()
        w.writerows(resultados)
    with open(args.salida / "f1_por_clase.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clase", *args.bases])
        for c in CLASES:
            w.writerow([c, *[r["f1_por_clase"][c] for r in resultados]])

    print("\n===== Resumen =====")
    for r in resultados:
        print(f"{r['red']:<18} F1 macro={r['val_f1_macro']:.3f}  "
              f"F1 biological={r['val_f1_biological']:.3f}  "
              f"acc contenedor={r['val_acc_contenedor']:.3f}  "
              f"TFLite={r['tflite_ms_cpu']} ms / {r['tflite_mb']} MB")
    print(f"Archivos en {args.salida}/")


if __name__ == "__main__":
    main()
