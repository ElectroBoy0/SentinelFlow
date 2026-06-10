import numpy as np
from qdrant_client import QdrantClient
from sklearn.cluster import KMeans
import requests

QDRANT_HOST = "localhost"
OLLAMA_URL = "http://localhost:11434/api/generate"
COLLECTION_NAME = "drifted_samples"

# Policy Weights
ALPHA = 0.4 # Uncertainty
BETA = 0.3 # Drift
GAMMA = 0.3 # Diversity

def calculate_uncertainty(logits=None):
    # For now, we simulate uncertainty if we don't have logits stored.
    # In a full system, entropy of softmax(logits) is used.
    return np.random.uniform(0.5, 1.0)

def diversity_scoring(embeddings, n_clusters=10):
    """
    Use KMeans++ initialization logic or actual KMeans to find diverse core-sets.
    Returns indices of diverse samples.
    """
    if len(embeddings) < n_clusters:
        return list(range(len(embeddings)))
        
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1).fit(embeddings)
    # Get the closest point to each cluster center
    diverse_indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(embeddings - center, axis=1)
        diverse_indices.append(np.argmin(distances))
    return diverse_indices

def run_active_learning_cycle():
    print("Connecting to Qdrant...")
    client = QdrantClient(QDRANT_HOST, port=6333)
    
    # Scroll to get all recently drifted samples
    try:
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=500,
            with_payload=True,
            with_vectors=True
        )
    except Exception as e:
        print(f"Could not connect to Qdrant or collection missing: {e}")
        return
    
    if not records:
        print("No drifted samples to process.")
        return
        
    print(f"Retrieved {len(records)} drifted samples from Qdrant.")
    
    embeddings = np.array([r.vector for r in records])
    
    # 1. Diversity filter (Core-sets)
    diverse_indices = diversity_scoring(embeddings, n_clusters=min(10, len(embeddings)))
    
    scored_samples = []
    for idx in diverse_indices:
        r = records[idx]
        # 2. Uncertainty Score
        uncertainty = calculate_uncertainty()
        
        # 3. Drift Score (Distance to cluster center could be used, or magnitude)
        drift_score = np.linalg.norm(r.vector) 
        
        # 4. Total Policy Score
        # Normalize scores in a real scenario
        score = ALPHA * uncertainty + BETA * drift_score + GAMMA * 1.0 # Diversity is 1.0 for selected
        
        scored_samples.append({
            "score": score,
            "text": r.payload['text'],
            "original_text": r.payload['original']
        })
        
    # Sort by highest score
    scored_samples.sort(key=lambda x: x['score'], reverse=True)
    
    top_samples = scored_samples[:5]
    print(f"Selected Top {len(top_samples)} High-Value Samples. Sending to Ollama...")
    
    # Self-Healing Loop: Call Ollama LLaMA-3
    for sample in top_samples:
        prompt = f"Categorize the following text into a domain (e.g., Tech, Finance, Science): '{sample['original_text']}'. Respond with ONLY the category name."
        
        print(f"Querying Ollama for: {sample['original_text'][:50]}...")
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            })
            if response.status_code == 200:
                label = response.json().get('response', '').strip()
                print(f"Silver-Standard Label: {label}")
                # Here we would save this to the new training dataset DB
            else:
                print(f"Ollama failed: {response.text}")
        except Exception as e:
            print(f"Could not reach Ollama: {e}")

if __name__ == "__main__":
    run_active_learning_cycle()
