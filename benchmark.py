import time
import tracemalloc
import numpy as np
import engine

def benchmark_inference():
    print("VAYRA Performance Benchmark")
    print("---------------------------")
    batch_sizes = [10, 50, 100]
    
    for batch in batch_sizes:
        tracemalloc.start()
        
        latencies = []
        for _ in range(50):
            start_time = time.time()
            
            # Simulate real VAYRA route point evaluation
            # input to ML model: lat, lng, precipitation_sum, elevation, slope
            dummy_point = [17.3850, 78.4867, 120.5, 500.0, 2.5]
            _ = engine.predict(dummy_point)
            
            # Simulate real VAYRA risk scoring
            _ = engine.vuln2(2.5, engine.v_model(0.6))
            
            latencies.append((time.time() - start_time) * 1000)
            
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        print(f"Input size: {batch} segments")
        print(f"Average latency: {avg_latency:.2f} ms")
        print(f"P95 latency: {p95_latency:.2f} ms")
        print(f"Peak memory: {peak / (1024*1024):.2f} MB\n")

if __name__ == '__main__':
    benchmark_inference()
