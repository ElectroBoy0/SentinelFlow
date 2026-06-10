import json
import asyncio
from kafka import KafkaConsumer, KafkaProducer
import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import cleaner # Our compiled C++ extension

ONNX_MODEL_DIR = "onnx_model"
KAFKA_BROKER = "localhost:9092"
RAW_TOPIC = "raw-text-stream"
TELEMETRY_TOPIC = "inference-telemetry"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_DIR)
    model = ORTModelForFeatureExtraction.from_pretrained(ONNX_MODEL_DIR)
    return tokenizer, model

def process_text(text, tokenizer, model):
    # 1. Clean via C++ Pybind11 extension
    cleaned_text = cleaner.clean_text(text)
    
    # 2. Tokenize
    inputs = tokenizer(cleaned_text, return_tensors="np", padding=True, truncation=True)
    
    # 3. ONNX Inference
    outputs = model(**inputs)
    # Get the embedding (pooler output or mean of last hidden state)
    # BGE models use the [CLS] token (index 0) of the last hidden state
    last_hidden_state = outputs.last_hidden_state
    embedding = last_hidden_state[0, 0, :].tolist()
    
    return {
        "original_text": text,
        "cleaned_text": cleaned_text,
        "embedding": embedding
    }

async def consume_and_predict():
    print("Loading ONNX Model...")
    tokenizer, model = load_model()
    
    print(f"Connecting to Redpanda at {KAFKA_BROKER}...")
    consumer = KafkaConsumer(
        RAW_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='latest'
    )
    
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print("Serving Daemon Started. Waiting for messages...")
    for message in consumer:
        raw_event = message.value
        text_id = raw_event.get("id")
        text = raw_event.get("text", "")
        
        result = process_text(text, tokenizer, model)
        result["id"] = text_id
        
        # Publish telemetry
        producer.send(TELEMETRY_TOPIC, result)
        print(f"Processed message {text_id} and published to telemetry.")

if __name__ == "__main__":
    asyncio.run(consume_and_predict())
