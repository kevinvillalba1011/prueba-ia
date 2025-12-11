# Clasificación de oficios judiciales

Este proyecto entrena y utiliza un modelo basado en **DistilBERT** para clasificar textos de oficios judiciales en distintas categorías (por ejemplo: Desembargo, Requerimiento, Embargo, Traslado).

El repositorio contiene dos scripts principales:

- `train.py`: entrena el modelo a partir de un archivo CSV.
- `inference.py`: carga el modelo ya entrenado y realiza predicciones sobre nuevos textos.

---

## 1. Requisitos

Se recomienda usar Python 3.9+ y crear un entorno virtual.

Dependencias principales:

- `transformers`
- `datasets`
- `torch`
- `scikit-learn`
- `pandas`

Instalación de dependencias (ejemplo usando `pip`):

```bash
pip install transformers datasets torch scikit-learn pandas
```

> Nota: en Windows, puede ser recomendable instalar `torch` siguiendo las instrucciones oficiales de PyTorch según tu GPU/CPU: https://pytorch.org/

---

## 2. Estructura del proyecto

- `train.py`: script de entrenamiento del modelo.
- `inference.py`: script de inferencia para probar el modelo.
- `oficios_judiciales_200.csv`: dataset de ejemplo con textos y tipo de documento.
- `modelo_oficios/`: carpeta donde se guarda el modelo entrenado y los checkpoints.
- `resultados_inferencia.json`: ejemplo de archivo de salida con predicciones.

---

## 3. Entrenamiento del modelo

Si quieres entrenar el modelo desde cero (o volver a entrenarlo), asegúrate de tener el archivo `oficios_judiciales_200.csv` en la raíz del proyecto, con al menos las columnas:

- `texto`: contenido del oficio judicial.
- `tipo_documento`: etiqueta de la clase (por ejemplo, Desembargo, Requerimiento, Embargo, Traslado).

Luego ejecuta:

```bash
python train.py
```

Esto hará lo siguiente:

1. Leer el CSV y mapear cada `tipo_documento` a un número (label).
2. Dividir el dataset en entrenamiento (80%) y prueba (20%).
3. Tokenizar los textos usando el modelo base `distilbert-base-multilingual-cased`.
4. Entrenar un modelo de clasificación (`AutoModelForSequenceClassification`).
5. Guardar el modelo final y el tokenizador en la carpeta `modelo_oficios/`.

Después del entrenamiento, en `modelo_oficios/` tendrás:

- Los pesos finales del modelo (`model.safetensors`).
- Archivos del tokenizador (`tokenizer.json`, `vocab.txt`, etc.).
- Carpetas de checkpoints de cada época (`checkpoint-20`, `checkpoint-40`, ...).

---

## 4. Probar el modelo (inferencia)

Una vez que tengas la carpeta `modelo_oficios/` con el modelo entrenado (ya sea porque la generaste con `train.py` o porque la descargaste/lista de otro lugar), puedes probar el modelo con:

```bash
python inference.py
```

El script `inference.py` hace lo siguiente:

1. Carga el tokenizador y el modelo desde `modelo_oficios/`.
2. Define una función `predecir(texto)` que:
   - Tokeniza el texto.
   - Obtiene las logits del modelo.
   - Calcula las probabilidades (`softmax`).
   - Devuelve el tipo de documento predicho y su probabilidad.
3. Evalúa una lista de ejemplos de texto (definida en el propio script).
4. Guarda los resultados de ejemplo en `resultados_inferencia.json`.

Ejemplo de salida (estructura) en `resultados_inferencia.json`:

```json
[
  {
    "texto": "Se ordena embargo sobre las tierras",
    "prediccion": {
      "tipo_documento": "Embargo",
      "probabilidad": 0.95
    }
  }
]
```

---

## 5. Usar el modelo en tu propio código

Puedes reutilizar la función `predecir` de `inference.py` en otros scripts. Ejemplo básico:

```python
from inference import predecir

texto = "Por medio del presente oficio se comunica..."
resultado = predecir(texto)
print(resultado)
```

Esto imprimirá un diccionario con el texto, la etiqueta predicha y la probabilidad asociada.

---

## 6. Notas adicionales

- El dataset de ejemplo es pequeño (200 instancias), por lo que las métricas pueden mejorar si se entrena con más datos.
- La carpeta `modelo_oficios/` puede ser pesada (por los pesos del modelo y los checkpoints). Normalmente se recomienda **no subir los pesos al repositorio** y, en su lugar, compartirlos por otro medio (por ejemplo, un enlace de descarga o un repositorio de modelos como Hugging Face Hub).
- Ajustes como el número de épocas, la tasa de aprendizaje y el tamaño máximo de secuencia se pueden modificar en `train.py` dentro de los `TrainingArguments`.
