"""Etapa 2, paso 2: fine-tuning sobre el modelo con la base congelada.

Parte del checkpoint de `entrenar_base_congelada.py`, descongela las últimas
capas de la red preentrenada con `descongelar_ultimas` (las BatchNormalization
siguen congeladas) y las reentrena junto con la cabeza con una tasa de
aprendizaje baja (1e-5 por defecto, 100 veces menor que la de la etapa
anterior), para ajustarlas a imágenes de residuos sin perder lo aprendido en
ImageNet.

Capas a descongelar (--capas), contadas desde el final de MobileNetV3Large
(187 capas). Estos valores cortan justo entre bloques:
    18  bloque 14 + convolución final
    33  bloques 13-14 + convolución final
    48  bloques 12-14 + convolución final: toda la última etapa (7x7, 160
        canales). Es el valor por defecto, cerca del 26 % de la base.

- Antes de entrenar se mide el val_loss del modelo de partida. El checkpoint
  solo se guarda si el fine-tuning lo supera; si nunca lo supera, no se
  guarda modelo y el de la base congelada sigue siendo el mejor.
- Early stopping sobre val_loss con paciencia 4, igual que la etapa anterior.

Uso (en WSL con el entorno activado, desde la raíz del repo):
    python src/fine_tuning.py
    python src/fine_tuning.py --capas 33 --lr 1e-5 --epocas 30

Resultados en `resultados/fine_tuning/`:
    modelo_fine_tuning.keras   modelo de la mejor época (no se sube a git)
    historial.csv              loss, accuracy, val_loss y val_accuracy por época
    curva_error.png            error por época, con el val_loss de partida
    resumen.json               configuración, mejor época y comparación
    metricas_val.json, por_clase_val.csv, matriz_confusion_val.csv
                               métricas del checkpoint en validación (con MCC)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tensorflow import keras

from config import DIR_DATOS, DIR_IMAGENES
from curvas import graficar_error, leer_historial
from evaluar import evaluar
from modelo import (callbacks, crear_dataset, descongelar_ultimas,
                    leer_class_weights)


def contar_entrenables(modelo: keras.Model) -> int:
    return int(sum(np.prod(w.shape) for w in modelo.trainable_weights))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modelo-base", type=Path,
                    default=Path("resultados/base_congelada/"
                                 "modelo_base_congelada.keras"))
    ap.add_argument("--capas", type=int, default=48,
                    help="capas a descongelar desde el final de la base")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epocas", type=int, default=30)
    ap.add_argument("--paciencia", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imagenes", type=Path, default=DIR_IMAGENES)
    ap.add_argument("--datos", type=Path, default=DIR_DATOS)
    ap.add_argument("--salida", type=Path,
                    default=Path("resultados/fine_tuning"))
    ap.add_argument("--limite", type=int, default=None,
                    help="usar solo N imágenes por partición (para probar)")
    args = ap.parse_args()
    if not args.modelo_base.exists():
        raise SystemExit(f"No existe {args.modelo_base}: corre primero "
                         "src/entrenar_base_congelada.py")
    args.salida.mkdir(parents=True, exist_ok=True)

    kw = dict(batch_size=args.batch, dir_imagenes=args.imagenes,
              dir_datos=args.datos, limite=args.limite)
    ds_train = crear_dataset("train", **kw)
    ds_val = crear_dataset("val", **kw)

    modelo = keras.models.load_model(args.modelo_base)
    entrenables_antes = contar_entrenables(modelo)
    val_loss_partida = float(
        modelo.evaluate(ds_val, verbose=0, return_dict=True)["loss"])
    print(f"val_loss del modelo de partida: {val_loss_partida:.4f}")

    descongelar_ultimas(modelo, args.capas, lr=args.lr)
    entrenables = contar_entrenables(modelo)
    print(f"Parámetros entrenables: {entrenables_antes:,} -> "
          f"{entrenables:,} (últimas {args.capas} capas de la base)")

    ruta_modelo = args.salida / "modelo_fine_tuning.keras"
    ruta_historial = args.salida / "historial.csv"
    if ruta_modelo.exists():
        ruta_modelo.unlink()  # que no quede uno de una corrida anterior
    cbs = callbacks(str(ruta_modelo), paciencia=args.paciencia,
                    umbral_inicial=val_loss_partida)
    cbs.append(keras.callbacks.CSVLogger(str(ruta_historial)))

    t0 = time.time()
    modelo.fit(ds_train, validation_data=ds_val, epochs=args.epocas,
               class_weight=leer_class_weights(args.datos),
               callbacks=cbs, verbose=2)
    minutos = (time.time() - t0) / 60

    historia = leer_historial(ruta_historial)
    val_loss = historia["val_loss"]
    epocas_corridas = len(val_loss)
    mejor = int(np.argmin(val_loss)) + 1  # épocas desde 1
    mejoro = ruta_modelo.exists()

    graficar_error(historia, mejor if mejoro else None, args.paciencia,
                   f"Fine-tuning (últimas {args.capas} capas, lr {args.lr:g}):"
                   " error por época",
                   args.salida / "curva_error.png",
                   referencia=val_loss_partida)

    resumen = {
        "modelo_base": args.modelo_base.as_posix(),
        "capas_descongeladas": args.capas,
        "parametros_entrenables": entrenables,
        "lr": args.lr,
        "batch": args.batch,
        "paciencia": args.paciencia,
        "epocas_corridas": epocas_corridas,
        "epocas_maximas": args.epocas,
        "paro_por_early_stopping": epocas_corridas < args.epocas,
        "val_loss_partida": round(val_loss_partida, 4),
        "mejoro_sobre_partida": mejoro,
        "mejor_epoca": mejor if mejoro else None,
        "mejor_val_loss": round(val_loss[mejor - 1], 4),
        "val_accuracy_mejor_epoca": round(
            historia["val_accuracy"][mejor - 1], 4),
        "min_entrenamiento": round(minutos, 1),
    }
    with open(args.salida / "resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
    print(json.dumps(resumen, indent=2, ensure_ascii=False))

    if not mejoro:
        print(f"\nEl fine-tuning no bajó el val_loss de partida "
              f"({val_loss_partida:.4f}); no se guardó modelo. El de "
              f"{args.modelo_base} sigue siendo el mejor.")
        return
    print(f"\nCheckpoint (época {mejor}): {ruta_modelo}")
    evaluar(ruta_modelo, "val", args.salida, **kw)
    print(f"\nArchivos en {args.salida}/")


if __name__ == "__main__":
    main()
