"""Etapa 2, paso 1: entrenamiento completo con la base congelada.

Solo se entrenan las capas nuevas (la "cabeza"); la red preentrenada no se
modifica. El modelo que sale de aquí es el punto de partida del fine-tuning.

- Early stopping sobre `val_loss` con paciencia de 4 épocas: el entrenamiento
  se detiene cuando pasan 4 épocas seguidas sin que `val_loss` mejore.
- Checkpoint: se guarda el modelo cada vez que `val_loss` mejora, así que el
  archivo final corresponde a la mejor época (el punto donde el modelo dejó
  de mejorar), no a la última.
- El error (loss) y la accuracy de cada época se guardan en `historial.csv`
  conforme avanza el entrenamiento, y al final se grafica el error contra
  las épocas.

Uso (en Colab con GPU, desde la raíz del repo y después de preparar_datos):
    python src/entrenar_base_congelada.py
    python src/entrenar_base_congelada.py --epocas 60 --paciencia 4

Resultados en `resultados/base_congelada/`:
    modelo_base_congelada.keras   modelo de la mejor época (no se sube a git)
    historial.csv                 loss, accuracy, val_loss y val_accuracy por época
    curva_error.png               error de train y val contra las épocas
    resumen.json                  mejor época y sus métricas
    metricas_val.json, por_clase_val.csv, matriz_confusion_val.csv
                                  métricas del checkpoint en validación,
                                  incluido MCC (ver src/evaluar.py)
"""

import argparse
import json
import time
from pathlib import Path

from tensorflow import keras

from config import BASE_ELEGIDA, DIR_DATOS, DIR_IMAGENES
from curvas import graficar_error, leer_historial
from evaluar import evaluar
from modelo import (callbacks, construir_modelo, crear_dataset,
                    leer_class_weights)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--base", default=BASE_ELEGIDA)
    ap.add_argument("--epocas", type=int, default=50,
                    help="máximo de épocas; normalmente para antes por "
                         "early stopping")
    ap.add_argument("--paciencia", type=int, default=4,
                    help="épocas sin mejora en val_loss antes de detenerse")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imagenes", type=Path, default=DIR_IMAGENES)
    ap.add_argument("--datos", type=Path, default=DIR_DATOS)
    ap.add_argument("--salida", type=Path,
                    default=Path("resultados/base_congelada"))
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

    modelo = construir_modelo(args.base, pesos=args.pesos, lr=args.lr)
    ruta_modelo = args.salida / "modelo_base_congelada.keras"
    ruta_historial = args.salida / "historial.csv"

    # callbacks() ya trae EarlyStopping(val_loss, restore_best_weights) y
    # ModelCheckpoint(val_loss, save_best_only). CSVLogger escribe cada época
    # al terminarla, así que el historial no se pierde si Colab se desconecta.
    cbs = callbacks(str(ruta_modelo), paciencia=args.paciencia)
    cbs.append(keras.callbacks.CSVLogger(str(ruta_historial)))

    t0 = time.time()
    modelo.fit(ds_train, validation_data=ds_val, epochs=args.epocas,
               class_weight=leer_class_weights(args.datos),
               callbacks=cbs, verbose=2)
    minutos = (time.time() - t0) / 60

    historia = leer_historial(ruta_historial)
    val_loss = historia["val_loss"]
    mejor_epoca = val_loss.index(min(val_loss)) + 1  # épocas desde 1
    epocas_corridas = len(val_loss)
    paro_temprano = epocas_corridas < args.epocas

    graficar_error(historia, mejor_epoca, args.paciencia,
                   f"{args.base}, base congelada: error por época",
                   args.salida / "curva_error.png")

    resumen = {
        "red": args.base,
        "epocas_corridas": epocas_corridas,
        "epocas_maximas": args.epocas,
        "paciencia": args.paciencia,
        "paro_por_early_stopping": paro_temprano,
        "mejor_epoca": mejor_epoca,
        "mejor_val_loss": round(val_loss[mejor_epoca - 1], 4),
        "val_accuracy_mejor_epoca": round(
            historia["val_accuracy"][mejor_epoca - 1], 4),
        "train_loss_mejor_epoca": round(
            historia["loss"][mejor_epoca - 1], 4),
        "min_entrenamiento": round(minutos, 1),
        "lr": args.lr,
        "batch": args.batch,
    }
    with open(args.salida / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    if not paro_temprano and epocas_corridas - mejor_epoca < args.paciencia:
        print(f"\nAVISO: llegó al máximo de {args.epocas} épocas y val_loss "
              f"mejoró en las últimas {args.paciencia}; el modelo quizá "
              "seguía mejorando. Considera subir --epocas.")
    print(f"\nCheckpoint (época {mejor_epoca}): {ruta_modelo}")

    # Se evalúa el archivo del checkpoint (no el modelo en memoria), así las
    # métricas corresponden exactamente a lo que usará el fine-tuning.
    evaluar(ruta_modelo, "val", args.salida, **kw)
    print(f"\nArchivos en {args.salida}/")


if __name__ == "__main__":
    main()
