import time
import tracemalloc
import numpy as np

def benchmark_inference():
    print("Starting automated benchmarking script for Vayra...")
    batch_sizes = [1, 16, 64, 256]
    
    for batch in batch_sizes:
        tracemalloc.start()
        start_time = time.time()
        
        # Simulate inference
        dummy_data = np.random.rand(batch, 128)
        _ = np.dot(dummy_data, np.random.rand(128, 64))
        
        latency = (time.time() - start_time) * 1000  # ms
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Batch Size: {batch:3d} | Inference Latency: {latency:.2f} ms | Peak Memory Usage: {peak / 1024:.2f} KB")

if __name__ == '__main__':
    benchmark_inference()
