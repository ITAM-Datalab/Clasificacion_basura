"""Constantes compartidas por la etapa 1 (datos) y la etapa 2 (modelo)."""

from pathlib import Path

SEMILLA = 42

# Orden fijo de las clases: el índice de cada una es la salida del modelo.
CLASES = ["battery", "biological", "cardboard", "glass",
          "metal", "paper", "plastic", "trash"]

# Clases del dataset original que no son residuos que separa el módulo.
CLASES_DESCARTADAS = {"clothes", "shoes"}

ORGANICO = "organico"
RECICLABLE = "inorganico_reciclable"
NO_RECICLABLE = "inorganico_no_reciclable"

CLASES_A_CONTENEDORES = {
    "battery": NO_RECICLABLE,  # manejo especial
    "biological": ORGANICO,
    "cardboard": RECICLABLE,
    "glass": RECICLABLE,
    "metal": RECICLABLE,
    "paper": RECICLABLE,
    "plastic": RECICLABLE,
    "trash": NO_RECICLABLE,
}

TAM_IMAGEN = (224, 224)

DIR_IMAGENES = Path("data/imagenes")  # imágenes extraídas, no van a git
DIR_DATOS = Path("datos")             # splits y json, sí van a git
