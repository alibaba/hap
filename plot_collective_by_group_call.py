import config
import sys
import datetime
import time
import torch
import torch.fx
import numpy as np
import hap
import collectives

def run(global_rank, local_rank, max_ratio, queue):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank, timeout=datetime.timedelta(hours=2))

    total_length = 4 * 1024 # 4MB
    sharding_lengths = [max_ratio] + [(1 - max_ratio) / (config.world_size - 1)] * (config.world_size - 1)
    sharding_lengths = [ x / sum(sharding_lengths) for x in sharding_lengths]
    hap.sharding_round(total_length, sharding_lengths)

    if local_rank == 0:
        print("sharding_lengths:", sharding_lengths)

    tensor = torch.rand(256, sharding_lengths[global_rank]).to(local_rank)

    result_times = []
    last_iter_time = time.time()
    for iter in range(config.run_iter):
        collectives.all_gather(tensor, 1, sharding_lengths, global_rank)
        # collectives.all_gather_by_group_call(tensor, 1, sharding_lengths, global_rank)
        # torch.cuda.synchronize()
        dist.barrier()

        if local_rank == 0:
            iter_duration = time.time() - last_iter_time
            result_times.append(iter_duration)
            last_iter_time += iter_duration
            print("iter time: ", iter_duration)
            print("avg±std:", np.mean(result_times[-config.avg_iter:]), np.std(result_times[-config.avg_iter:]))

    if local_rank == 0:
        queue.put(np.mean(result_times[-config.avg_iter:]))

if __name__ == '__main__':
    ranks = [ int(x) for x in sys.argv[1].split(',') ]

    # if torch.cuda.device_count() != len(ranks):
    #     print("forget to set CUDA_VISIBLE_DEVICES")
    #     raise SystemExit

    import os
    os.environ['MASTER_ADDR'] = str(config.master_addr)
    os.environ['MASTER_PORT'] = str(config.master_port)
    os.environ['WORLD_SIZE'] = str(config.world_size)

    import torch.multiprocessing as mp
    ctx = mp.get_context('spawn')
    queue = ctx.Queue(1)

    result = []

    for max_ratio in np.linspace(1 / config.world_size, 1, 100, endpoint=False):
        for local_rank, global_rank in enumerate(ranks):
            ctx.Process(target=run, args=(global_rank, local_rank, max_ratio, queue)).start()

        for p in mp.active_children():
            p.join()

        result.append((max_ratio, queue.get()))
        print(result[-1])

    print(result)
