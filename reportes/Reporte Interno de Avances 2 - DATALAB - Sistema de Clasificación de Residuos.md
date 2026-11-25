# **Reporte Interno de Avances \- DATALAB**

**Nombre del Proyecto:** Sistema de Clasificación de Residuos  
**Nombres de los Integrantes:** Emilio Juárez Ávalos, Alejandro Martínez  
**Fecha de Entrega:** 28/10/2026

## **1\. ¿Qué hicieron en este periodo?**

Revisamos el modelo que se había entrenado anteriormente y analizamos el dataset de imágenes. Con base en eso decidimos construir un modelo nuevo, propio, partiendo de una red neuronal preentrenada. Definimos la estructura del modelo, las librerías que usaremos y cómo vamos a preparar el dataset. Lo que falta decidir es qué red preentrenada usar como base.

### **1.1 Algoritmo de clasificación**

Mantenemos la decisión de usar una Red Neuronal Convolucional (RNC), ahora con la técnica de ***transfer learning***: tomamos una red que ya fue entrenada con millones de imágenes (el dataset ImageNet), le quitamos su capa final y le agregamos capas propias para que aprenda a reconocer nuestros tipos de residuo. Así aprovechamos lo que la red ya sabe sobre formas, bordes y texturas, y necesitamos muchas menos imágenes que si la entrenáramos desde cero.

**Estructura del modelo:**

Imagen de entrada (224 × 224 píxeles, RGB) → **red preentrenada (por definir)** → GlobalAveragePooling → Dense(256, ReLU) → Dropout(0.5) → Dense(8, softmax)

**Entrenamiento en dos etapas:**

1. **Base congelada:** se entrenan solo las capas nuevas y la red preentrenada no se modifica.
2. ***Fine-tuning*:** se descongelan las últimas capas de la red preentrenada y se reentrenan con una tasa de aprendizaje baja, para ajustarlas a imágenes de residuos.

El modelo anterior solo hizo la primera etapa; agregar la segunda es una de las principales mejoras.

**Otras decisiones:**

* **Salida:** el modelo predice 8 clases de residuo y después una tabla asigna cada clase a un contenedor del módulo (ver sección 1.4). No entrenamos directamente con los 3 contenedores porque juntaría en una sola clase objetos que se ven muy distintos, como vidrio y cartón.
* **Configuración:** optimizador Adam, función de pérdida *categorical cross-entropy*, *early stopping* y guardado automático del mejor modelo.
* **Aumento de datos:** rotaciones, giros, zoom y cambios de brillo, para que el modelo tolere distintas posiciones del residuo en la bandeja.
* **Desbalance de clases:** pesos por clase, para que las clases con pocas imágenes cuenten más durante el entrenamiento.
* **Evaluación:** partición fija del dataset (70 % entrenamiento, 15 % validación, 15 % prueba), estratificada por clase y con semilla fija, para que los resultados se puedan repetir. Mediremos *accuracy*, F1 por clase, matriz de confusión y tiempo de inferencia en el hardware del módulo.

### **1.2 Red preentrenada (por decidir)**

Estas son las redes que estamos considerando. Todas vienen incluidas en Keras con pesos de ImageNet:

| Red | Parámetros | Precisión en ImageNet (top-1) | Observaciones |
| :---- | ----: | ----: | :---- |
| **MobileNetV2** | 3.5 M | 71.3 % | La que usó el modelo anterior. Es la más ligera |
| **MobileNetV3Large** | 5.4 M | 75.6 % | Versión más nueva de MobileNet, diseñada para celulares y placas pequeñas |
| **EfficientNetB0** | 5.3 M | 77.1 % | Más precisa con un tamaño parecido, pero algo más lenta |
| **EfficientNetV2B0** | 7.2 M | 78.7 % | La más precisa de las redes ligeras |
| **ResNet50** | 25.6 M | 74.9 % | Clásica y muy documentada, pero unas 7 veces más pesada que MobileNetV2 |

*Fuente: documentación de Keras Applications.*

**Criterios para elegir:**

* **Hardware del módulo:** si la placa que defina el equipo de Makers es pequeña (por ejemplo, una Raspberry Pi), hay que priorizar las redes ligeras.
* **Velocidad:** el modelo tiene que clasificar el residuo antes de que la bandeja se mueva.
* **Precisión con nuestro dataset:** la precisión en ImageNet es solo una referencia; lo que importa es cómo le va con imágenes de residuos.
* **Compatibilidad con TensorFlow Lite**, para poder correr el modelo en el módulo.

Para decidir, haremos una prueba corta con las 2 o 3 candidatas que mejor se ajusten al hardware, usando la misma partición del dataset, y nos quedaremos con la que tenga mejor F1 y clasifique a tiempo.

### **1.3 Librerías y herramientas**

| Librería | Para qué la usamos |
| :---- | :---- |
| **Python 3** | Lenguaje principal del proyecto |
| **TensorFlow / Keras** | Construir, entrenar y guardar el modelo; aumento de datos; redes preentrenadas (`keras.applications`) |
| **NumPy** | Manejo de las imágenes como arreglos y de las predicciones |
| **scikit-learn** | Partición estratificada del dataset, cálculo de pesos por clase y métricas (reporte de clasificación y matriz de confusión) |
| **Matplotlib y Seaborn** | Curvas de entrenamiento y gráfica de la matriz de confusión |
| **OpenCV** | Captura de imágenes con la cámara del módulo y preprocesamiento (recorte y redimensionado) |
| **Pillow** | Lectura y conversión de imágenes del dataset (por ejemplo, convertir WebP a JPG) |
| **Google Colab** | Entorno de entrenamiento con GPU |

Para ejecutar el modelo en el hardware del módulo planeamos usar **TensorFlow Lite**, que genera una versión más ligera del modelo. Esto depende de la placa que defina el equipo de Makers.

### **1.4 Dataset**

* **Nombre y fuente:** *Garbage Classification V2*, publicado en Kaggle por Suman Kunwar ([kaggle.com/datasets/sumn2u/garbage-classification-v2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)).
* **Tamaño original:** 19,762 imágenes en 10 clases (780 MB comprimido).

**Clases que no sirven para nuestro objetivo.** El dataset incluye muchas clases que no corresponden a lo que queremos que el modelo aprenda. Las más grandes son **ropa (5,327 imágenes)** y **zapatos (1,977 imágenes)**: juntas suman 7,304 imágenes, el **37 % del dataset**. No son residuos que vaya a separar el módulo y no encajan en ninguno de los tres contenedores. Además, como ropa es la clase más grande (27 %), haría que el modelo dedicara buena parte del entrenamiento a algo que no necesitamos. Por eso las quitamos y nos quedamos con **12,458 imágenes en 8 clases**.

| Clase | Imágenes | ¿Se usa? | Contenedor |
| :---- | ----: | :----: | :---- |
| clothes (ropa) | 5,327 | No | — |
| glass (vidrio) | 3,061 | Sí | Inorgánico reciclable |
| plastic (plástico) | 1,984 | Sí | Inorgánico reciclable |
| shoes (zapatos) | 1,977 | No | — |
| cardboard (cartón) | 1,825 | Sí | Inorgánico reciclable |
| paper (papel) | 1,680 | Sí | Inorgánico reciclable |
| metal | 1,020 | Sí | Inorgánico reciclable |
| biological (orgánico) | 997 | Sí | Orgánico |
| trash (basura general) | 947 | Sí | Inorgánico no reciclable |
| battery (baterías) | 944 | Sí | Inorgánico no reciclable (residuo de manejo especial) |

**Imágenes por contenedor (después de quitar ropa y zapatos):**

| Contenedor | Clases | Imágenes | % |
| :---- | :---- | ----: | ----: |
| Inorgánico reciclable | vidrio, plástico, cartón, papel, metal | 9,570 | 76.8 % |
| Inorgánico no reciclable | basura general, baterías | 1,891 | 15.2 % |
| Orgánico | orgánico | 997 | 8.0 % |
| **Total** | **8 clases** | **12,458** | **100 %** |

**Calidad de las imágenes (8 clases que se usan):**

* **Formato:** casi todas en JPG; algunas en PNG. Encontramos 3 imágenes de la clase *paper* que en realidad son WebP con extensión `.jpg`; hay que convertirlas antes de entrenar.
* **Duplicados:** comparamos el contenido de todos los archivos y no hay imágenes repetidas.
* **Resolución:** muy variable, desde 51 hasta 7,786 píxeles de ancho; la típica es de unos 334 × 306 píxeles. 4,101 imágenes (33 %) miden menos de 224 píxeles en su lado más corto y el modelo tendrá que ampliarlas.
* **Tipo de fotos:** es una mezcla de fotos de catálogo de internet (algunas con marca de agua), escenas con varios objetos y fotos de cerca sobre fondo liso. Ninguna se parece a lo que verá la cámara cenital del módulo.

**Puntos a considerar:**

* **Orgánico es la categoría más débil:** solo tiene el 8 % de las imágenes y el 92 % de ellas son pequeñas, aunque es uno de los tres contenedores del módulo. Vamos a buscar más imágenes de residuos orgánicos.
* **Desbalance:** casi 77 % de las imágenes son reciclables. Por eso usaremos pesos por clase y mediremos el F1 de cada clase, no solo la *accuracy* general.
* **Fotos propias:** empezaremos a tomar fotos con la cámara y la bandeja reales junto con el equipo de Makers, para que el modelo también aprenda de imágenes como las que verá en el módulo.

## **2\. ¿Qué problemas resolvieron sobre la marcha?**

* **El código no correspondía al modelo guardado.** Al revisar el archivo del modelo anterior vimos que era una MobileNetV2 congelada con 10 clases, mientras que el código del repositorio era una RNC simple con 8 clases. Para evitarlo, el modelo nuevo tendrá su código y sus resultados guardados juntos en el repositorio.
* **El dataset traía clases que no sirven.** Ropa y zapatos ocupaban el 37 % de las imágenes y no corresponden a lo que separará el módulo; las quitamos antes de entrenar.
* **No había métricas ni una partición fija.** La división anterior en entrenamiento, validación y prueba era aleatoria y no se podía repetir, así que no había forma confiable de saber si un cambio mejoraba el modelo. Definimos una partición estratificada con semilla fija.
* **8 clases contra 3 contenedores.** Lo resolvimos clasificando en 8 clases y asignando cada una a un contenedor con una tabla.
* **Archivos con formato incorrecto.** Detectamos 3 imágenes WebP guardadas como `.jpg`, que pueden detener el entrenamiento; se convertirán antes de usarlas.

## **3\. ¿Qué aprendieron?**

* Qué es el *transfer learning*, por qué conviene cuando se tienen pocas imágenes y la diferencia entre usar la red preentrenada congelada o reentrenar sus últimas capas (*fine-tuning*).
* Que elegir la red base es un equilibrio entre precisión, tamaño y velocidad, y que la precisión en ImageNet no garantiza que sea la mejor para residuos.
* Que un dataset público no siempre se ajusta al problema: hay que revisarlo y filtrar las clases que no sirven antes de entrenar.
* Por qué la *accuracy* sola engaña cuando las clases están desbalanceadas: un modelo puede acertar mucho en la clase más grande y fallar en las pequeñas.
* Que la calidad de las imágenes importa: como la resolución típica cambia mucho entre clases, el modelo podría aprender a distinguir el tamaño o la calidad de la foto en lugar del objeto.

## **4\. Evidencia Visual**

* Tabla de redes preentrenadas candidatas (sección 1.2).
* Tablas del dataset: clases que se usan, clases descartadas e imágenes por contenedor (sección 1.4).
* Para la siguiente entrega: curvas de entrenamiento y matriz de confusión del modelo nuevo, y fotos de las primeras pruebas con la cámara del módulo.
