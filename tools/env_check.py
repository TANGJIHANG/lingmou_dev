import time
import torch

print("torch      :", torch.__version__)
print("cuda build :", torch.version.cuda)
print("available  :", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA 不可用"

print("device     :", torch.cuda.get_device_name(0))
print("capability :", torch.cuda.get_device_capability(0))
print("arch list  :", torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list(), "构建里没有 sm_120 kernel"

x = torch.randn(4096, 4096)

t0 = time.perf_counter()
y_cpu = x @ x
t_cpu = time.perf_counter() - t0

xg = x.cuda()
t0 = time.perf_counter()
y_gpu = xg @ xg
torch.cuda.synchronize()
t_gpu = time.perf_counter() - t0

diff = (y_gpu.cpu() - y_cpu).abs().max().item()
print(f"cpu matmul : {t_cpu*1000:8.1f} ms")
print(f"gpu matmul : {t_gpu*1000:8.1f} ms")
print(f"speedup    : {t_cpu/t_gpu:8.1f} x")
print(f"max diff   : {diff:.3e}")
assert diff < 1e-2, "CPU/GPU 结果不一致，算子有问题"
print("PASS")