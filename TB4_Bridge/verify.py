import mlx.core as mx
import mlx.core.distributed as dist
import time


def benchmark():
    # Initialize distributed environment
    group = dist.init()
    rank = group.rank()
    world_size = group.size()

    # Configuration: 512MB tensor (Adjust size if needed)
    # 1024 * 1024 * 128 * 4 bytes (float32) = 512 MB
    shape = (1024, 1024, 128)
    data_size_gb = (1024 * 1024 * 128 * 4) / (1024 ** 3)

    # Create a tensor on each device
    a = mx.ones(shape, dtype=mx.float32) * (rank + 1)

    # Warm-up rounds to initialize the connection
    if rank == 0:
        print(f"[*] Starting warm-up...")
    for _ in range(3):
        dist.all_sum(a)
    mx.eval(a)

    # Benchmark rounds
    iters = 10
    if rank == 0:
        print(f"[*] Benchmarking {iters} iterations with {data_size_gb * 1024:.0f}MB tensor...")

    # Synchronize before starting timer
    mx.eval(a)

    start_time = time.perf_counter()

    for _ in range(iters):
        # all_sum triggers data exchange across the Thunderbolt bridge
        res = dist.all_sum(a)
        mx.eval(res)  # Ensure the operation is finished

    end_time = time.perf_counter()

    # Calculate statistics
    total_time = end_time - start_time
    avg_time = total_time / iters

    # Bandwidth calculation:
    # In all_sum for 2 nodes, each node sends and receives its own data size.
    # Effective bandwidth = Data Size / Average Time
    bandwidth_gps = data_size_gb / avg_time  # GB/s
    bandwidth_gbps = bandwidth_gps * 8  # Gbps

    if rank == 0:
        print("-" * 50)
        print(f"🚀 Benchmark Results (Rank {rank})")
        print(f" - Total Time: {total_time:.4f} sec")
        print(f" - Avg Time per Transfer: {avg_time:.4f} sec")
        print(f" - Effective Bandwidth: {bandwidth_gps:.2f} GB/s")
        print(f" - Effective Bandwidth: {bandwidth_gbps:.2f} Gbps")
        print("-" * 50)
        print(f"Note: Thunderbolt 4 theoretical max is 40 Gbps.")
        print(f"Practical IP-over-Thunderbolt usually hits 15~25 Gbps.")


if __name__ == "__main__":
    benchmark()