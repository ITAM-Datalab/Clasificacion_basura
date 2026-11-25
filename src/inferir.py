"""Clasifica fotos de residuos con el modelo TFLite.

Preparación de la imagen (igual que en el entrenamiento):
  1. Se abre y se convierte a RGB (también fotos en escala de grises, PNG con
     transparencia, WebP, CMYK...).
  2. Se redimensiona a 224x224 con interpolación bilineal, sin recortar
     (igual que tf.image.resize en src/modelo.py).
  3. Se pasa a float32 con valores 0-255, SIN dividir entre 255: MobileNetV3
     normaliza por dentro. Dividir entre 255 da predicciones equivocadas.

Solo necesita numpy, Pillow y un intérprete de TFLite. Usa el primero que
encuentre: ai_edge_litert, tflite_runtime o tensorflow. Así se puede copiar a
la placa del módulo junto con modelo.tflite y etiquetas.json, sin el resto
del repo.

Uso (desde la raíz del repo, después de src/exportar_tflite.py):
    python src/inferir.py foto.jpg
    python src/inferir.py fotos/*.jpg --top 3
    python src/inferir.py foto.jpg --json

Desde otro programa (por ejemplo, el que controla la cámara):
    from inferir import Clasificador
    clf = Clasificador("resultados/tflite/modelo.tflite")
    r = clf.clasificar("foto.jpg")        # o una PIL.Image, o un arreglo RGB
    r["contenedor"]                       # "organico", "inorganico_reciclable"...

Ojo con OpenCV: cv2.imread y las cámaras de OpenCV dan BGR, no RGB.
Convierte antes con cv2.cvtColor(img, cv2.COLOR_BGR2RGB).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

MODELO = Path("resultados/tflite/modelo.tflite")


def cargar_interprete(ruta: Path, hilos: int):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf  # tf.lite se carga de forma diferida
            Interpreter = tf.lite.Interpreter
    return Interpreter(model_path=str(ruta), num_threads=hilos)


class Clasificador:
    def __init__(self, ruta_modelo: Path = MODELO,
                 ruta_etiquetas: Path | None = None, hilos: int = 4):
        ruta_modelo = Path(ruta_modelo)
        ruta_etiquetas = Path(ruta_etiquetas or
                              ruta_modelo.with_name("etiquetas.json"))
        with open(ruta_etiquetas, encoding="utf-8") as f:
            etiquetas = json.load(f)
        self.clases = etiquetas["clases"]
        self.contenedores = etiquetas["clases_a_contenedores"]
        entrada = etiquetas["entrada"]
        self.tam = (entrada["ancho"], entrada["alto"])  # orden de PIL

        self.interp = cargar_interprete(ruta_modelo, hilos)
        self.interp.allocate_tensors()
        self.entrada = self.interp.get_input_details()[0]["index"]
        self.salida = self.interp.get_output_details()[0]["index"]

    def preparar(self, imagen) -> np.ndarray:
        """Ruta, PIL.Image o arreglo HxWx3 RGB -> (1, 224, 224, 3) float32 0-255."""
        if isinstance(imagen, np.ndarray):
            imagen = Image.fromarray(imagen.astype(np.uint8))
        elif not isinstance(imagen, Image.Image):
            imagen = Image.open(imagen)
        imagen = imagen.convert("RGB").resize(self.tam, Image.BILINEAR)
        x = np.asarray(imagen, dtype=np.float32)  # 0-255, sin dividir
        return x[None]

    def probabilidades(self, imagen) -> np.ndarray:
        self.interp.set_tensor(self.entrada, self.preparar(imagen))
        self.interp.invoke()
        return self.interp.get_tensor(self.salida)[0]

    def clasificar(self, imagen, top: int = 3) -> dict:
        t0 = time.perf_counter()
        p = self.probabilidades(imagen)
        ms = (time.perf_counter() - t0) * 1000
        orden = np.argsort(p)[::-1]
        clase = self.clases[orden[0]]
        return {
            "clase": clase,
            "contenedor": self.contenedores[clase],
            "probabilidad": round(float(p[orden[0]]), 4),
            "top": [{"clase": self.clases[i],
                     "probabilidad": round(float(p[i]), 4)}
                    for i in orden[:top]],
            "ms": round(ms, 1),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("imagenes", nargs="+", type=Path)
    ap.add_argument("--modelo", type=Path, default=MODELO)
    ap.add_argument("--etiquetas", type=Path, default=None,
                    help="por defecto, etiquetas.json junto al modelo")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--hilos", type=int, default=4)
    ap.add_argument("--json", action="store_true",
                    help="una línea JSON por imagen")
    args = ap.parse_args()

    clf = Clasificador(args.modelo, args.etiquetas, args.hilos)
    for ruta in args.imagenes:
        r = clf.clasificar(ruta, args.top)
        if args.json:
            print(json.dumps({"imagen": str(ruta), **r}, ensure_ascii=False))
            continue
        otras = ", ".join(f"{t['clase']} {t['probabilidad']:.1%}"
                          for t in r["top"][1:])
        print(f"{ruta}: {r['clase']} ({r['probabilidad']:.1%}) -> "
              f"{r['contenedor']}  [{otras}]  {r['ms']} ms")


if __name__ == "__main__":
    main()
