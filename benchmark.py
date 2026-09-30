import time
import tracemalloc
import numpy as np
from vayra_types import vayra_risk_assessment

def benchmark_inference():
    print("Starting automated VAYRA system benchmark...")
    batch_sizes = [1, 16, 64, 256]
    
    for batch in batch_sizes:
        tracemalloc.start()
        start_time = time.time()
        
        # Benchmark actual VAYRA inference function
        input_tensor = np.random.rand(batch, 128)
        _ = vayra_risk_assessment(input_tensor, threshold=0.5)
        
        latency = (time.time() - start_time) * 1000  # ms
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Batch Size: {batch:3d} | Vayra Inference Latency: {latency:.2f} ms | Peak Memory Usage: {peak / 1024:.2f} KB")

if __name__ == '__main__':
    benchmark_inference()
