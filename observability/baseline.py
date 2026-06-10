import numpy as np
import os

BASELINE_DIR = "baseline_data"

def generate_dummy_baseline(dim=384, num_samples=1000):
    print("Generating baseline dataset profile...")
    os.makedirs(BASELINE_DIR, exist_ok=True)
    
    # Simulate some normal distribution for training data
    baseline_vectors = np.random.normal(loc=0.0, scale=1.0, size=(num_samples, dim)).astype(np.float32)
    
    mean_vec = np.mean(baseline_vectors, axis=0)
    cov_matrix = np.cov(baseline_vectors, rowvar=False)
    
    np.save(os.path.join(BASELINE_DIR, "baseline_vectors.npy"), baseline_vectors)
    np.save(os.path.join(BASELINE_DIR, "mean_vec.npy"), mean_vec)
    np.save(os.path.join(BASELINE_DIR, "cov_matrix.npy"), cov_matrix)
    
    print("Baseline generated successfully.")

if __name__ == "__main__":
    generate_dummy_baseline()
