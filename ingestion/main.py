import json
import time
import random
import uuid
from kafka import KafkaProducer

KAFKA_BROKER = "localhost:9092"
RAW_TOPIC = "raw-text-stream"

SAMPLE_TEXTS = [
    "The new flagship smartphone features a brilliant <bold>120Hz OLED display</bold>!!!",
    "Stock markets rallied today after the Federal Reserve announced steady rates.",
    "Researchers have discovered a novel protein folding mechanism using AI...",
    "The recipe calls for 2 cups of flour, some sugar, and a pinch of salt.",
    "A massive earthquake struck the coastal region, triggering tsunami warnings."
]

def stream_data():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    print(f"Starting Ingestion Engine. Publishing to {RAW_TOPIC} at {KAFKA_BROKER}...")
    try:
        while True:
            text = random.choice(SAMPLE_TEXTS)
            event = {
                "id": str(uuid.uuid4()),
                "text": text,
                "timestamp": time.time()
            }
            producer.send(RAW_TOPIC, event)
            print(f"Sent: {event['id']}")
            time.sleep(0.1) # Simulate 10 messages per second
    except KeyboardInterrupt:
        print("Ingestion stopped.")

if __name__ == "__main__":
    stream_data()
