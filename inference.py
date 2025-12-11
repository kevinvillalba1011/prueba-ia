import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "modelo_oficios"

# cargar modelo y tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

def predecir(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    prob, idx = torch.max(probs, dim=1)

    return {
        "texto": texto,
        "prediccion": {
            "tipo_documento": model.config.id2label[idx.item()],
            "probabilidad": round(float(prob.item()), 2)
        }
    }

# textos de prueba
ejemplos = [
    "Se ordena embargo sobre las tierras",
    "Por medio del presente oficio se informa...",
    "Se remite copia certificada del acta de audiencia...",
    "Se requiere al demandado para comparecer...",
    "Comunicamos el levantamiento del embargo..."
]

resultados = [predecir(t) for t in ejemplos]

with open("resultados_inferencia.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=4, ensure_ascii=False)
