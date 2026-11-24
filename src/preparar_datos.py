"""Prepara el dataset de forma reproducible.

Parte del zip de Kaggle (Garbage Classification V2) y:
  1. Extrae solo las 8 clases que separa el módulo (descarta clothes y shoes).
  2. Abre cada imagen con Pillow, la pasa a RGB y la guarda como JPEG real.
     Esto corrige los archivos WebP que vienen con extensión .jpg.
  3. Resuelve nombres repetidos: el zip trae pares como `glass_20241.jpg` y
     `glass_20241.png` que irían al mismo `glass_20241.jpg`. Si son la misma
     foto se conserva solo la primera; si son fotos distintas la segunda se
     guarda como `glass_20241(1).jpg`.
  4. Hace la partición 70/15/15 estratificada con semilla fija.
  5. Guarda en `datos/`:
       - splits.csv                  ruta, clase, particion, ancho, alto
       - clases.json                 orden de clases (índice -> nombre)
       - clases_a_contenedores.json  mapeo de 8 clases a 3 contenedores
       - class_weights.json          pesos "balanced" calculados con train
       - resumen_split.csv           conteo por clase y partición

Uso (local o en Colab, desde la raíz del repo):
    python src/preparar_datos.py --zip Datasets/2025-05-10_archive.zip

Las imágenes extraídas quedan en `data/imagenes/` (ignorada por git).
Los archivos de `datos/` sí se suben: son los que usa la etapa 2.
"""

import argparse
import csv
import io
import json
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import (CLASES, CLASES_A_CONTENEDORES, CLASES_DESCARTADAS,
                    DIR_DATOS, DIR_IMAGENES, SEMILLA)

EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Dos imágenes con el mismo nombre se consideran la misma foto si tienen el
# mismo tamaño y la diferencia promedio por píxel (escala 0-255) es menor a
# este umbral. En el zip, los 30 pares .jpg/.png difieren entre 0.2 y 3.2,
# solo por la compresión JPEG; fotos distintas difieren decenas de unidades.
UMBRAL_MISMA_FOTO = 8.0


def misma_foto(a: Image.Image, b: Image.Image) -> bool:
    if a.size != b.size:
        return False
    dif = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return float(dif.mean()) < UMBRAL_MISMA_FOTO


def ruta_libre(salida: Path, usadas: set) -> Path:
    """`nombre(1).jpg`, `nombre(2).jpg`, ... el primero que no se haya usado."""
    n = 1
    while (nueva := salida.with_name(f"{salida.stem}({n}){salida.suffix}")) in usadas:
        n += 1
    return nueva


def extraer_imagenes(ruta_zip: Path, destino: Path) -> list[dict]:
    """Extrae y normaliza las imágenes de las clases que se usan."""
    filas = []
    convertidas = []
    ilegibles = []
    duplicadas = []   # misma foto con otro formato: no se guarda
    renombradas = []  # foto distinta con el mismo nombre: se guarda como (n)
    usadas = set()    # rutas de salida ya escritas en esta corrida
    descartadas = Counter()

    with zipfile.ZipFile(ruta_zip) as zf:
        for nombre in sorted(zf.namelist()):
            p = PurePosixPath(nombre)
            if nombre.endswith("/") or p.suffix.lower() not in EXTENSIONES:
                continue
            if any(parte.startswith((".", "__MACOSX")) for parte in p.parts):
                continue
            clase = p.parent.name
            if clase in CLASES_DESCARTADAS:
                descartadas[clase] += 1
                continue
            if clase not in CLASES:
                continue

            datos = zf.read(nombre)
            try:
                img = Image.open(io.BytesIO(datos))
                formato = img.format
                img = img.convert("RGB")
            except Exception as e:  # noqa: BLE001
                ilegibles.append(f"{nombre}: {e}")
                continue

            salida = destino / clase / (p.stem + ".jpg")
            if salida in usadas:
                with Image.open(salida) as previa:
                    igual = misma_foto(previa.convert("RGB"), img)
                if igual:
                    duplicadas.append(nombre)
                    continue
                salida = ruta_libre(salida, usadas)
                renombradas.append(f"{nombre} -> {salida.name}")
            usadas.add(salida)
            salida.parent.mkdir(parents=True, exist_ok=True)
            if formato == "JPEG" and p.suffix.lower() in {".jpg", ".jpeg"}:
                salida.write_bytes(datos)  # sin recomprimir
            else:
                img.save(salida, "JPEG", quality=95)
                convertidas.append(f"{nombre} ({formato})")

            filas.append({
                "ruta": salida.relative_to(destino).as_posix(),
                "clase": clase,
                "ancho": img.width,
                "alto": img.height,
            })

    print(f"Imágenes extraídas: {len(filas)}")
    for clase, n in sorted(descartadas.items()):
        print(f"  descartadas {clase}: {n}")
    if convertidas:
        print(f"Convertidas a JPEG ({len(convertidas)}):")
        for c in convertidas:
            print(f"  {c}")
    if duplicadas:
        print(f"Duplicadas, no se guardan ({len(duplicadas)}):")
        for c in duplicadas:
            print(f"  {c}")
    if renombradas:
        print(f"Mismo nombre pero foto distinta, renombradas "
              f"({len(renombradas)}):")
        for c in renombradas:
            print(f"  {c}")
    if ilegibles:
        print(f"No se pudieron leer ({len(ilegibles)}):")
        for c in ilegibles:
            print(f"  {c}")
    return filas


def particionar(filas: list[dict], semilla: int) -> None:
    """Agrega la columna 'particion': 70 % train, 15 % val, 15 % test."""
    idx = np.arange(len(filas))
    y = [f["clase"] for f in filas]
    idx_train, idx_resto = train_test_split(
        idx, test_size=0.30, stratify=y, random_state=semilla)
    y_resto = [y[i] for i in idx_resto]
    idx_val, idx_test = train_test_split(
        idx_resto, test_size=0.50, stratify=y_resto, random_state=semilla)
    for nombre, grupo in (("train", idx_train), ("val", idx_val),
                          ("test", idx_test)):
        for i in grupo:
            filas[i]["particion"] = nombre


def guardar(filas: list[dict], dir_datos: Path) -> None:
    dir_datos.mkdir(parents=True, exist_ok=True)

    filas = sorted(filas, key=lambda f: f["ruta"])
    with open(dir_datos / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, lineterminator="\n", fieldnames=["ruta", "clase", "particion",
                                          "ancho", "alto"])
        w.writeheader()
        w.writerows(filas)

    with open(dir_datos / "clases.json", "w", encoding="utf-8") as f:
        json.dump({i: c for i, c in enumerate(CLASES)}, f, indent=2,
                  ensure_ascii=False)

    with open(dir_datos / "clases_a_contenedores.json", "w",
              encoding="utf-8") as f:
        json.dump(CLASES_A_CONTENEDORES, f, indent=2, ensure_ascii=False)

    y_train = [f["clase"] for f in filas if f["particion"] == "train"]
    pesos = compute_class_weight("balanced", classes=np.array(CLASES),
                                 y=np.array(y_train))
    with open(dir_datos / "class_weights.json", "w", encoding="utf-8") as f:
        json.dump({i: round(float(p), 4) for i, p in enumerate(pesos)}, f,
                  indent=2)

    conteo = Counter((f["clase"], f["particion"]) for f in filas)
    with open(dir_datos / "resumen_split.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["clase", "contenedor", "train", "val", "test", "total"])
        for c in CLASES:
            n = [conteo[(c, p)] for p in ("train", "val", "test")]
            w.writerow([c, CLASES_A_CONTENEDORES[c], *n, sum(n)])
            print(f"  {c:<11} train={n[0]:>5} val={n[1]:>4} test={n[2]:>4}")

    print(f"Archivos guardados en {dir_datos}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--zip", type=Path,
                    default=Path("Datasets/2025-05-10_archive.zip"))
    ap.add_argument("--imagenes", type=Path, default=DIR_IMAGENES,
                    help="dónde extraer las imágenes (no se sube a git)")
    ap.add_argument("--datos", type=Path, default=DIR_DATOS,
                    help="dónde guardar splits y json (sí se sube a git)")
    ap.add_argument("--semilla", type=int, default=SEMILLA)
    args = ap.parse_args()

    filas = extraer_imagenes(args.zip, args.imagenes)
    faltan = set(CLASES) - {f["clase"] for f in filas}
    if faltan:
        raise SystemExit(f"No se encontraron imágenes de: {sorted(faltan)}")
    particionar(filas, args.semilla)
    guardar(filas, args.datos)


if __name__ == "__main__":
    main()
