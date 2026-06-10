import json
import time
import numpy as np
import os
from kafka import KafkaConsumer
from prometheus_client import start_http_server, Gauge
from scipy.stats import wasserstein_distance
from alibi_detect.cd import MMDDrift
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

KAFKA_BROKER = "localhost:9092"
TELEMETRY_TOPIC = "inference-telemetry"
WINDOW_SIZE = 100
DIM = 384

# Prometheus metrics
MMD_GAUGE = Gauge('drift_mmd_distance', 'Maximum Mean Discrepancy between baseline and streaming window')
WASSERSTEIN_GAUGE = Gauge('drift_wasserstein_distance', 'Wasserstein distance between baseline and streaming window')

def load_baseline():
    baseline_path = "baseline_data/baseline_vectors.npy"
    if not os.path.exists(baseline_path):
        from baseline import generate_dummy_baseline
        generate_dummy_baseline()
    
    baseline_vectors = np.load(baseline_path)
    return baseline_vectors

def init_qdrant():
    client = QdrantClient("localhost", port=6333)
    # Check if collection exists
    try:
        client.get_collection("drifted_samples")
    except:
        client.create_collection(
            collection_name="drifted_samples",
            vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
        )
    return client

def monitor_drift():
    print("Starting Observability Engine...")
    baseline_vectors = load_baseline()
    
    # Initialize MMD detector using alibi-detect
    detector = MMDDrift(baseline_vectors[:500], backend='pytorch', p_val=.05)
    
    q_client = init_qdrant()
    
    print("Starting Prometheus HTTP server on port 8000...")
    start_http_server(8000)
    
    consumer = KafkaConsumer(
        TELEMETRY_TOPIC,
        bootstrap_servers=KAFKA_BROKER,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    
    window = []
    
    print("Listening for telemetry data...")
    for msg in consumer:
        event = msg.value
        window.append(event)
        
        if len(window) >= WINDOW_SIZE:
            print(f"Processing window of {WINDOW_SIZE} events...")
            embeddings = np.array([e['embedding'] for e in window], dtype=np.float32)
            
            # Wasserstein (using 1D mean approximations)
            window_mean = np.mean(embeddings, axis=0)
            baseline_mean = np.load("baseline_data/mean_vec.npy")
            w_dist = wasserstein_distance(window_mean, baseline_mean)
            WASSERSTEIN_GAUGE.set(w_dist)
            
            # MMD via Alibi Detect
            pred = detector.predict(embeddings)
            mmd_dist = pred['data']['distance']
            is_drift = pred['data']['is_drift']
            MMD_GAUGE.set(mmd_dist)
            
            print(f"Metrics -> MMD: {mmd_dist:.4f}, Wasserstein: {w_dist:.4f}, Drift: {is_drift}")
            
            if is_drift:
                print(">>> OOD Event Detected! Pushing window to Qdrant...")
                points = [
                    PointStruct(
                        id=idx,
                        vector=e['embedding'],
                        payload={"id": e['id'], "text": e['cleaned_text'], "original": e['original_text']}
                    ) for idx, e in enumerate(window)
                ]
                q_client.upsert(
                    collection_name="drifted_samples",
                    points=points
                )
            
            window = [] # Reset window

if __name__ == "__main__":
    monitor_drift()
