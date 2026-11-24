# Clasificación de residuos (DATALAB)

Modelo de visión para un módulo de basura inteligente. Una cámara cenital toma
una foto del residuo, el modelo lo clasifica en 8 clases y una tabla manda cada
clase a uno de 3 contenedores: orgánico, inorgánico reciclable e inorgánico no
reciclable.

Equipo de software/IA: Emilio Juárez Ávalos y Alejandro Martínez, junto con el
equipo de Makers (estructura física y microcontroladores).

## Estructura

| Ruta | Qué es |
|---|---|
| `src/config.py` | Clases, mapeo a contenedores, semilla y rutas compartidas |
| `src/preparar_datos.py` | Extrae el dataset, corrige formatos, quita duplicados y hace el split |
| `src/modelo.py` | Carga de datos (tf.data), aumento y armado del modelo; lo usan ambas etapas |
| `src/prueba_redes.py` | Prueba corta para elegir la red preentrenada |
| `src/entrenar_base_congelada.py` | Entrenamiento completo con la base congelada (etapa 2, paso 1) |
| `notebooks/prueba_redes.ipynb` | Corre la prueba corta en Colab con GPU |
| `entorno/` | Versiones fijas y script para crear el entorno con GPU en WSL2/Linux |
| `resultados/` | Métricas y gráficas que se suben al repo (los modelos `.keras`/`.tflite` no) |
| `datos/` | Split y archivos JSON generados (se suben a git) |
| `data/imagenes/` | Imágenes extraídas (no se suben, las genera el script) |
| `Datasets/` | Zip original de Kaggle (no se sube, 780 MB) |
| `reportes/` | Reportes internos de avances para DATALAB |
| `viejo_modelo/` | Modelo del semestre anterior, solo como referencia |

## Dataset

[Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
(Suman Kunwar, Kaggle). Se usan 8 de sus 10 clases. Se descartan `clothes` y
`shoes` porque no son residuos que separa el módulo. Quedan 12,428 imágenes
(sin los 30 duplicados `.jpg`/`.png` que trae el zip).

Se usa la **versión 8** del dataset (diciembre 2024); las versiones posteriores
cambian las imágenes y por lo tanto el split.

| Clase | Contenedor |
|---|---|
| biological | Orgánico |
| cardboard, glass, metal, paper, plastic | Inorgánico reciclable |
| battery (manejo especial), trash | Inorgánico no reciclable |

## Entorno

- **Colab:** basta con `pip install -r requirements.txt`.
- **Entrenamiento local con GPU NVIDIA (Linux o WSL2):** usa las versiones
  exactas de `entorno/requirements-gpu.lock` (Python 3.12, TensorFlow 2.21.0,
  CUDA 12.9 y cuDNN 9 instalados por pip). Necesitas el driver de NVIDIA (en
  WSL2 basta el de Windows) y [uv](https://docs.astral.sh/uv/). Dentro de WSL,
  desde la raíz del repo:

  ```bash
  bash entorno/crear_entorno_wsl.sh
  source ~/venvs/clasificacion_basura/bin/activate
  ```

  El script termina mostrando si TensorFlow detecta la GPU. TensorFlow no usa
  GPU en Windows nativo (desde la versión 2.11), por eso se entrena en WSL2.

## Preparar los datos

1. Descarga la versión 8 del zip (no hace falta cuenta de Kaggle) y guárdalo
   como `Datasets/2025-05-10_archive.zip`.
2. Desde la raíz del repo:

```bash
mkdir -p Datasets
curl -L -o Datasets/2025-05-10_archive.zip \
  "https://www.kaggle.com/api/v1/datasets/download/sumn2u/garbage-classification-v2?datasetVersionNumber=8"
pip install -r requirements.txt
python src/preparar_datos.py --zip Datasets/2025-05-10_archive.zip
```

El script:

- extrae solo las 8 clases a `data/imagenes/<clase>/`;
- abre cada imagen y la guarda como JPEG en RGB, lo que corrige los archivos
  con extensión `.jpg` que en realidad son PNG, WebP o MPO;
- resuelve los nombres repetidos (pares como `glass_20241.jpg` y
  `glass_20241.png`): si son la misma foto (mismo tamaño y diferencia promedio
  por píxel menor a 8 en escala 0-255) guarda solo la primera; si son fotos
  distintas, guarda la segunda como `glass_20241(1).jpg`. En la versión 8 los
  30 pares son la misma foto y se descartan;
- hace la partición 70/15/15 estratificada por clase con semilla 42;
- escribe en `datos/`:
  - `splits.csv`, con ruta, clase, partición, ancho y alto de cada imagen;
  - `clases.json`, con el índice de cada clase (el orden de salida del modelo);
  - `clases_a_contenedores.json`;
  - `class_weights.json`, con pesos `balanced` calculados solo con train;
  - `resumen_split.csv`, con el conteo por clase y partición.

Correrlo dos veces con el mismo zip da exactamente el mismo `splits.csv`.

En Colab: clona el repo, sube el zip a `Datasets/` (o enlázalo desde Drive) y
corre `!python src/preparar_datos.py`.

## Elegir la red preentrenada

Abre `notebooks/prueba_redes.ipynb` en Colab con GPU. Entrena MobileNetV3Large y
EfficientNetB0 con la base congelada durante 8 épocas como máximo y compara en
validación el F1 macro, el F1 de biological, la exactitud por contenedor y el
tiempo y tamaño en TensorFlow Lite. Los resultados quedan en
`resultados/prueba_redes/`.

### Resultados (validación, base congelada, 8 épocas)

> Estos resultados se obtuvieron con el split anterior, que tenía 30 fotos
> repetidas (7 de ellas en dos particiones a la vez). Al quitar los duplicados
> el split cambió por completo. Como son 30 imágenes de 12,458, no se espera
> que cambie la elección de la red, pero las cifras no son exactamente
> comparables con las del entrenamiento nuevo.

| Red | F1 macro | F1 biological | Acc. contenedor | TFLite (MB) | TFLite (ms/img, CPU) | Min. entrenamiento (CPU) |
|---|---:|---:|---:|---:|---:|---:|
| **MobileNetV3Large** | 0.916 | **0.974** | 0.975 | **3.6** | **6.6** | **9.3** |
| EfficientNetB0 | **0.922** | 0.970 | **0.976** | 4.9 | 7.7 | 15.0 |

F1 por clase en `resultados/prueba_redes/f1_por_clase.csv` y curvas en
`resultados/prueba_redes/curvas_*.png`. Los tiempos se midieron en un CPU de 4
núcleos con TFLite cuantizado (rango dinámico): sirven para comparar las redes,
no son el tiempo real en la placa del módulo.

**Red elegida: MobileNetV3Large.** La diferencia de F1 macro es de 0.006, menor
que el umbral de 0.02 que fijamos para considerarla relevante. Con el criterio
de desempate, MobileNetV3Large gana: tiene mejor F1 en biological (la clase más
débil), es un 27 % más chica, es un 14 % más rápida en TFLite y entrena en
casi la mitad del tiempo. Como la placa del módulo aún no está definida, conviene
la opción más ligera. EfficientNetB0 fue mejor en cardboard, paper y trash, así
que queda como alternativa si el fine-tuning de MobileNetV3Large se estanca.

En ambas redes la pérdida de validación seguía bajando en la época 8, así que
la etapa 2 (más épocas y fine-tuning) tiene margen de mejora.

## Entrenar con la base congelada

Primer paso de la etapa 2: entrena solo las capas nuevas de la red elegida
(`BASE_ELEGIDA` en `src/config.py`) y deja la red preentrenada sin cambios.

```bash
# en WSL, desde la raíz del repo (ver "Entorno"):
source ~/venvs/clasificacion_basura/bin/activate
python src/entrenar_base_congelada.py            # máx. 50 épocas, paciencia 4
python src/entrenar_base_congelada.py --epocas 80
```

- **Early stopping:** se detiene cuando `val_loss` lleva 4 épocas seguidas sin
  mejorar (`--paciencia`).
- **Checkpoint:** guarda el modelo solo cuando `val_loss` mejora, así que
  `modelo_base_congelada.keras` es el de la mejor época y no el de la última.
  Ese archivo es el punto de partida del fine-tuning.
- Si llega al máximo de épocas sin detenerse, avisa para subir `--epocas`.

Resultados en `resultados/base_congelada/`: `historial.csv` (error y accuracy
por época, se escribe al terminar cada época), `curva_error.png` (error contra
épocas, con la mejor época marcada) y `resumen.json`. El `.keras` no se sube
a git.

## Modelo viejo

`viejo_modelo/` tiene el trabajo anterior. El código (`modelo.py`, una CNN
desde cero con 8 clases) no corresponde al archivo guardado (una MobileNetV2
congelada con 10 clases), no hay métricas y el split no era reproducible. Por
eso el modelo nuevo se construye desde cero sobre este repo.
