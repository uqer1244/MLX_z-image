import mlx.core as mx
import mlx.core.distributed as dist


def main():
    group = dist.init()

    rank = group.rank()
    size = group.size()

    a = mx.ones((1024, 1024)) * (rank + 1)

    total_sum = dist.all_sum(a)

    print(f"[Device {rank}/{size}] Local Value: {a[0, 0].item()}, Reduced Sum: {total_sum[0, 0].item()}")

if __name__ == "__main__":
    main()