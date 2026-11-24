"""Historial de entrenamiento (CSVLogger) y gráfica de error por época."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def leer_historial(ruta: Path) -> dict[str, list[float]]:
    """Lee el CSV que escribe CSVLogger: una fila por época."""
    with open(ruta, encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return {col: [float(fila[col]) for fila in filas]
            for col in filas[0] if col != "epoch"}


def graficar_error(historia: dict, mejor_epoca: int | None, paciencia: int,
                   titulo: str, ruta: Path,
                   referencia: float | None = None) -> None:
    """Error de train y val contra las épocas.

    `mejor_epoca` (desde 1) marca el checkpoint; None si no se guardó ninguno.
    `referencia` dibuja una línea horizontal, p. ej. el val_loss del modelo
    de partida en el fine-tuning.
    """
    epocas = range(1, len(historia["loss"]) + 1)
    fig, eje = plt.subplots(figsize=(8, 5))
    eje.plot(epocas, historia["loss"], marker="o", ms=3, label="train")
    eje.plot(epocas, historia["val_loss"], marker="o", ms=3,
             label="validación")
    if referencia is not None:
        eje.axhline(referencia, color="tab:red", ls=":", lw=1.5,
                    label=f"val_loss de partida ({referencia:.4f})")
    if mejor_epoca is not None:
        mejor_val = historia["val_loss"][mejor_epoca - 1]
        eje.axvline(mejor_epoca, color="gray", ls="--", lw=1)
        eje.scatter([mejor_epoca], [mejor_val], s=80, zorder=3,
                    color="black",
                    label=f"checkpoint (época {mejor_epoca}, "
                          f"val_loss={mejor_val:.4f})")
        ultima = len(historia["loss"])
        if ultima > mejor_epoca:  # épocas de paciencia sin mejora
            eje.axvspan(mejor_epoca, ultima, color="gray", alpha=0.12,
                        label=f"sin mejora (paciencia {paciencia})")
    eje.set_title(titulo)
    eje.set_xlabel("época")
    eje.set_ylabel("error (categorical cross-entropy)")
    eje.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    eje.grid(alpha=0.3)
    eje.legend()
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
