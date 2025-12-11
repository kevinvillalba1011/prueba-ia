import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ===============================
# 1. Cargar el Dataset (tu CSV)
# ===============================
df = pd.read_csv("oficios_judiciales_200.csv")

# Creamos un diccionario para convertir las clases a números
label2id = {label: i for i, label in enumerate(df["tipo_documento"].unique())}
id2label = {v: k for k, v in label2id.items()}

# Añadimos una columna numérica llamada "label"
df["label"] = df["tipo_documento"].map(label2id)

# Convertimos el DataFrame a Dataset de Hugging Face
dataset = Dataset.from_pandas(df)

# Dividir en entrenamiento (80%) y prueba (20%)
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_ds = dataset["train"]
test_ds = dataset["test"]


# ===============================
# 2. Preparar Tokenizer
# ===============================
MODEL_NAME = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["texto"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

# aplicar tokenización al dataset completo
train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Borrar columnas que no necesita el modelo
train_ds = train_ds.remove_columns(["texto", "tipo_documento"])
test_ds = test_ds.remove_columns(["texto", "tipo_documento"])

# Convertir a formato PyTorch
train_ds.set_format("torch")
test_ds.set_format("torch")


# ===============================
# 3. Cargar modelo preentrenado
# ===============================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)


# ===============================
# 4. Función de métricas
# ===============================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    accuracy = accuracy_score(labels, preds)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ===============================
# 5. Parámetros del entrenamiento
# ===============================
training_args = TrainingArguments(
    output_dir="./modelo_oficios",
    eval_strategy="epoch",     # evaluar cada época
    save_strategy="epoch",           # guardar cada época
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=20
)


# ===============================
# 6. Crear el Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# ===============================
# 7. Iniciar entrenamiento
# ===============================
trainer.train()

# Guardamos el modelo entrenado
trainer.save_model("./modelo_oficios")
tokenizer.save_pretrained("./modelo_oficios")
