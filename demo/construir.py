"""Construye el sitio estático del demo (lo corre GitHub Actions).

Clasifica cada foto de una partición (test por defecto: fotos que el modelo
no vio al entrenar) con src/inferir.py y modelo.tflite, y escribe en --salida:

    index.html    la página del demo (copia de demo/index.html)
    modelo.tflite el mismo modelo, para clasificar en el navegador (LiteRT.js)
                  las fotos que suba quien use el demo
    datos.json    clases, contenedores y, por foto, la clase real y las
                  probabilidades que dio el modelo
    fotos/        miniaturas JPEG para mostrar (el modelo clasifica la
                  imagen original, no la miniatura)

Uso (desde cualquier carpeta; necesita data/imagenes/ ya extraído):
    python demo/construir.py
    python -m http.server -d sitio 8000      # para verlo en local
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))
from inferir import Clasificador  # noqa: E402


def leer_particion(ruta_splits: Path, particion: str) -> list[dict]:
    with open(ruta_splits, encoding="utf-8") as f:
        return [fila for fila in csv.DictReader(f)
                if fila["particion"] == particion]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modelo", type=Path,
                    default=RAIZ / "resultados/tflite/modelo.tflite")
    ap.add_argument("--particion", default="test", choices=["val", "test"])
    ap.add_argument("--imagenes", type=Path, default=RAIZ / "data/imagenes")
    ap.add_argument("--datos", type=Path, default=RAIZ / "datos")
    ap.add_argument("--salida", type=Path, default=RAIZ / "sitio")
    ap.add_argument("--lado", type=int, default=360,
                    help="lado máximo de las miniaturas, en píxeles")
    ap.add_argument("--limite", type=int, default=None,
                    help="usar solo N fotos (para probar)")
    args = ap.parse_args()

    clf = Clasificador(args.modelo)
    filas = leer_particion(args.datos / "splits.csv", args.particion)
    if args.limite:
        filas = filas[:args.limite]

    if args.salida.exists():
        shutil.rmtree(args.salida)
    (args.salida / "fotos").mkdir(parents=True)

    fotos, t0 = [], time.time()
    for i, fila in enumerate(filas, 1):
        origen = args.imagenes / fila["ruta"]
        with Image.open(origen) as img:
            probs = clf.probabilidades(img)
            miniatura = img.convert("RGB")
            miniatura.thumbnail((args.lado, args.lado))
        destino = args.salida / "fotos" / fila["ruta"]
        destino.parent.mkdir(parents=True, exist_ok=True)
        miniatura.save(destino, "JPEG", quality=82)
        fotos.append({
            "ruta": f"fotos/{fila['ruta']}",
            "real": clf.clases.index(fila["clase"]),
            "probs": [round(float(p), 4) for p in probs],
        })
        if i % 250 == 0 or i == len(filas):
            print(f"  {i}/{len(filas)} fotos ({time.time() - t0:.0f} s)")

    real = np.array([f["real"] for f in fotos])
    pred = np.array([int(np.argmax(f["probs"])) for f in fotos])
    cont = np.array([clf.contenedores[c] for c in clf.clases])
    with open(args.modelo.with_name("resumen.json"), encoding="utf-8") as f:
        resumen_tflite = json.load(f)
    with open(args.modelo.with_name("etiquetas.json"), encoding="utf-8") as f:
        entrada = json.load(f)["entrada"]

    datos = {
        "generado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": os.environ.get("GITHUB_SHA", "")[:7],
        "repositorio": os.environ.get("GITHUB_REPOSITORY", ""),
        "particion": args.particion,
        "modelo": {"cuantizacion": resumen_tflite.get("cuantizacion"),
                   "mb": resumen_tflite.get("tflite_mb")},
        "entrada": {"alto": entrada["alto"], "ancho": entrada["ancho"]},
        "clases": clf.clases,
        "contenedores": clf.contenedores,
        "accuracy": round(float((real == pred).mean()), 4),
        "accuracy_contenedor": round(float(
            (cont[real] == cont[pred]).mean()), 4),
        "fotos": fotos,
    }
    with open(args.salida / "datos.json", "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, separators=(",", ":"))
    shutil.copy(Path(__file__).with_name("index.html"), args.salida)
    # Para clasificar en el navegador las fotos que suba la gente.
    shutil.copy(args.modelo, args.salida / "modelo.tflite")

    print(f"{len(fotos)} fotos de {args.particion}: accuracy "
          f"{datos['accuracy']:.4f}, por contenedor "
          f"{datos['accuracy_contenedor']:.4f}")
    print(f"Sitio en {args.salida}/")


if __name__ == "__main__":
    main()
