"""Evalúa un modelo guardado sobre una partición (por defecto validación).

Métricas:
  - accuracy, F1 macro y MCC con las 8 clases
  - por clase: precisión, recall, F1 y MCC de esa clase contra el resto
  - con los 3 contenedores: accuracy, F1 macro, MCC y F1 por contenedor
  - matrices de confusión de 8 clases y de 3 contenedores (CSV y PNG)

MCC (coeficiente de correlación de Matthews) va de -1 a 1: 1 es predicción
perfecta, 0 equivale a adivinar y -1 es desacuerdo total. A diferencia de la
accuracy, usa toda la matriz de confusión (aciertos y errores de cada clase),
así que no se infla cuando las clases están desbalanceadas, como aquí (glass
tiene 3 veces más imágenes que trash o battery).

Test se reserva para el modelo final: durante el desarrollo se evalúa en val.

Uso (desde la raíz del repo):
    python src/evaluar.py --modelo resultados/base_congelada/modelo_base_congelada.keras

Guarda junto al modelo (o en --salida):
    metricas_<particion>.json          métricas globales y por contenedor
    por_clase_<particion>.csv          precisión, recall, F1 y MCC por clase
    matriz_confusion_<particion>.csv   8 clases; filas = real, columnas = predicha
    matriz_confusion_<particion>.png
    matriz_confusion_contenedor_<particion>.csv   lo mismo con los 3 contenedores
    matriz_confusion_contenedor_<particion>.png

En las imágenes el color es el porcentaje de cada fila (de la clase real), para
comparar clases de distinto tamaño; el número de cada celda es el conteo.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix,  # noqa: E402
                             f1_score, matthews_corrcoef,
                             precision_recall_fscore_support)
from tensorflow import keras  # noqa: E402

from config import (CLASES, CLASES_A_CONTENEDORES, DIR_DATOS,  # noqa: E402
                    DIR_IMAGENES)
from modelo import crear_dataset  # noqa: E402

CONTENEDORES = sorted(set(CLASES_A_CONTENEDORES.values()))
A_CONTENEDOR = np.array([CONTENEDORES.index(CLASES_A_CONTENEDORES[c])
                         for c in CLASES])
NOMBRES_CONTENEDOR = {
    "inorganico_no_reciclable": "Inorgánico\nno reciclable",
    "inorganico_reciclable": "Inorgánico\nreciclable",
    "organico": "Orgánico",
}

# Rampa secuencial azul (un solo tono, claro -> oscuro), pasos 100-700.
AZULES = LinearSegmentedColormap.from_list("azules", [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
    "#0d366b"])
TINTA = "#1a1a19"           # texto principal
TINTA_SECUNDARIA = "#52514e"
FONDO = "#ffffff"


def predecir(modelo: keras.Model, ds) -> tuple[np.ndarray, np.ndarray]:
    """(y_real, y_predicha) como índices de clase, recorriendo `ds` una vez."""
    reales, predichas = [], []
    for x, y in ds:
        reales.append(y.numpy().argmax(1))
        predichas.append(modelo.predict_on_batch(x).argmax(1))
    return np.concatenate(reales), np.concatenate(predichas)


def calcular_metricas(y: np.ndarray, p: np.ndarray) -> dict:
    etiquetas = list(range(len(CLASES)))
    prec, rec, f1, soporte = precision_recall_fscore_support(
        y, p, labels=etiquetas, zero_division=0)
    por_clase = {
        c: {"precision": round(float(prec[i]), 4),
            "recall": round(float(rec[i]), 4),
            "f1": round(float(f1[i]), 4),
            "mcc": round(float(matthews_corrcoef(y == i, p == i)), 4),
            "soporte": int(soporte[i])}
        for i, c in enumerate(CLASES)
    }
    cy, cp = A_CONTENEDOR[y], A_CONTENEDOR[p]
    f1_cont = f1_score(cy, cp, average=None,
                       labels=range(len(CONTENEDORES)), zero_division=0)
    return {
        "n_imagenes": int(len(y)),
        "accuracy": round(accuracy_score(y, p), 4),
        "f1_macro": round(f1_score(y, p, average="macro"), 4),
        "mcc": round(float(matthews_corrcoef(y, p)), 4),
        "contenedor": {
            "accuracy": round(accuracy_score(cy, cp), 4),
            "f1_macro": round(f1_score(cy, cp, average="macro"), 4),
            "mcc": round(float(matthews_corrcoef(cy, cp)), 4),
            "f1_por_contenedor": {c: round(float(f), 4)
                                  for c, f in zip(CONTENEDORES, f1_cont)},
        },
        "por_clase": por_clase,
    }


def graficar_matriz(matriz: np.ndarray, etiquetas: list[str], titulo: str,
                    ruta: Path) -> None:
    """Mapa de calor de la matriz de confusión.

    Color = porcentaje de la fila (de la clase real), en una rampa secuencial
    de un solo tono. Texto = conteo; las celdas en cero quedan vacías.
    """
    totales = matriz.sum(1, keepdims=True)
    pct = np.divide(matriz, totales, out=np.zeros(matriz.shape),
                    where=totales > 0)
    n = len(etiquetas)
    lado = max(5.0, 1.0 + 0.75 * n)  # mínimo para que quepan las etiquetas
    fig, eje = plt.subplots(figsize=(lado + 1.8, lado))
    vacia = np.ma.masked_where(matriz == 0, pct)
    AZULES.set_bad(FONDO)
    im = eje.imshow(vacia, cmap=AZULES, vmin=0, vmax=1)

    # 2 px de fondo entre celdas
    eje.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    eje.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    eje.grid(which="minor", color=FONDO, linewidth=2)
    eje.tick_params(which="minor", length=0)
    for lado_eje in eje.spines.values():
        lado_eje.set_visible(False)

    for i in range(n):
        for j in range(n):
            if matriz[i, j] == 0:
                continue
            r, g, b = to_rgb(AZULES(pct[i, j]))
            luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
            eje.text(j, i, str(matriz[i, j]), ha="center", va="center",
                     fontsize=9 if n > 4 else 11,
                     fontweight="bold" if i == j else "normal",
                     color=FONDO if luminancia < 0.45 else TINTA)

    eje.set_xticks(range(n), etiquetas, rotation=0 if n <= 4 else 35,
                   ha="center" if n <= 4 else "right", color=TINTA)
    eje.set_yticks(range(n), etiquetas, color=TINTA)
    eje.tick_params(which="major", length=0)
    eje.set_xlabel("Predicha", color=TINTA_SECUNDARIA)
    eje.set_ylabel("Real", color=TINTA_SECUNDARIA)
    eje.set_title(titulo, color=TINTA, loc="left")
    barra = fig.colorbar(im, ax=eje, fraction=0.046, pad=0.03)
    barra.set_label("% de la clase real", color=TINTA_SECUNDARIA)
    barra.ax.yaxis.set_major_formatter(
        PercentFormatter(xmax=1))
    barra.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130, facecolor=FONDO)
    plt.close(fig)


def guardar_matriz_csv(matriz: np.ndarray, etiquetas: list[str],
                       ruta: Path) -> None:
    with open(ruta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["real \\ predicha", *etiquetas])
        for e, fila in zip(etiquetas, matriz):
            w.writerow([e, *fila.tolist()])


def guardar_metricas(metricas: dict, y: np.ndarray, p: np.ndarray,
                     salida: Path, particion: str) -> None:
    salida.mkdir(parents=True, exist_ok=True)
    with open(salida / f"metricas_{particion}.json", "w",
              encoding="utf-8") as f:
        json.dump({k: v for k, v in metricas.items() if k != "por_clase"},
                  f, indent=2, ensure_ascii=False)

    with open(salida / f"por_clase_{particion}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["clase", "contenedor", "precision", "recall", "f1",
                    "mcc", "soporte"])
        for c, m in metricas["por_clase"].items():
            w.writerow([c, CLASES_A_CONTENEDORES[c], m["precision"],
                        m["recall"], m["f1"], m["mcc"], m["soporte"]])

    n, acc = metricas["n_imagenes"], metricas["accuracy"]
    matriz = confusion_matrix(y, p, labels=range(len(CLASES)))
    guardar_matriz_csv(matriz, CLASES,
                       salida / f"matriz_confusion_{particion}.csv")
    graficar_matriz(matriz, CLASES,
                    f"8 clases, {particion} ({n} imágenes, "
                    f"accuracy {acc:.1%})",
                    salida / f"matriz_confusion_{particion}.png")

    cy, cp = A_CONTENEDOR[y], A_CONTENEDOR[p]
    matriz_c = confusion_matrix(cy, cp, labels=range(len(CONTENEDORES)))
    guardar_matriz_csv(
        matriz_c, CONTENEDORES,
        salida / f"matriz_confusion_contenedor_{particion}.csv")
    graficar_matriz(
        matriz_c, [NOMBRES_CONTENEDOR[c] for c in CONTENEDORES],
        f"3 contenedores, {particion} (accuracy "
        f"{metricas['contenedor']['accuracy']:.1%})",
        salida / f"matriz_confusion_contenedor_{particion}.png")


def imprimir(metricas: dict, particion: str) -> None:
    c = metricas["contenedor"]
    print(f"\n== Métricas en {particion} ({metricas['n_imagenes']} imágenes) ==")
    print(f"8 clases:      accuracy={metricas['accuracy']:.4f}  "
          f"F1 macro={metricas['f1_macro']:.4f}  MCC={metricas['mcc']:.4f}")
    print(f"3 contenedores: accuracy={c['accuracy']:.4f}  "
          f"F1 macro={c['f1_macro']:.4f}  MCC={c['mcc']:.4f}")
    for cont, f1c in c["f1_por_contenedor"].items():
        print(f"  F1 {cont:<25} {f1c:.4f}")
    print(f"{'clase':<11} {'prec':>6} {'recall':>6} {'F1':>6} {'MCC':>6}")
    for clase, m in metricas["por_clase"].items():
        print(f"{clase:<11} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m['mcc']:>6.3f}")


def evaluar(ruta_modelo: Path, particion: str = "val",
            salida: Path | None = None, **kw_dataset) -> dict:
    """Carga el .keras, calcula las métricas, las guarda y las imprime."""
    modelo = keras.models.load_model(ruta_modelo)
    ds = crear_dataset(particion, **kw_dataset)
    y, p = predecir(modelo, ds)
    metricas = calcular_metricas(y, p)
    guardar_metricas(metricas, y, p, salida or Path(ruta_modelo).parent,
                     particion)
    imprimir(metricas, particion)
    return metricas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modelo", type=Path, required=True)
    ap.add_argument("--particion", default="val",
                    choices=["train", "val", "test"])
    ap.add_argument("--salida", type=Path, default=None,
                    help="carpeta de salida (por defecto, la del modelo)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imagenes", type=Path, default=DIR_IMAGENES)
    ap.add_argument("--datos", type=Path, default=DIR_DATOS)
    ap.add_argument("--limite", type=int, default=None)
    args = ap.parse_args()
    evaluar(args.modelo, args.particion, args.salida,
            batch_size=args.batch, dir_imagenes=args.imagenes,
            dir_datos=args.datos, limite=args.limite)


if __name__ == "__main__":
    main()
