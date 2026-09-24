import time, torch

def bench_gpu(fn, warmup=10, iters=50):
    for _ in range(warmup):            # ← 关键：烧掉首次开销
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()           # ← 关键：等设备真正算完
    return start.elapsed_time(end) / iters      # ms

def bench_cpu(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1000   # ms

N = 4096
x = torch.randn(N, N)
flops = 2 * N**3

ms_cpu = bench_cpu(lambda: x @ x)
if torch.cuda.is_available():
    xg = x.cuda()
    ms_gpu = bench_gpu(lambda: xg @ xg)
    print(f"cpu: {ms_cpu:8.2f} ms  {flops/ms_cpu/1e9:7.2f} TFLOPS")
    print(f"gpu: {ms_gpu:8.2f} ms  {flops/ms_gpu/1e9:7.2f} TFLOPS")
    print(f"speedup: {ms_cpu/ms_gpu:.1f}x")