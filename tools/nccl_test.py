import torch
import torch.distributed as dist
import time

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.cuda.set_device(rank)
    x = torch.ones(1).cuda(rank) * rank

    print(f"[Rank {rank}] before broadcast: {x.item()}", flush=True)

    # test broadcast
    dist.broadcast(x, src=0)

    print(f"[Rank {rank}] after broadcast: {x.item()}", flush=True)

    time.sleep(1)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
