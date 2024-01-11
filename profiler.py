import config

from sys import argv

import math
import torch
import torch.fx
import collectives
import time
import hap

def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

class FlopsProfiler:
    def __init__(self, model: torch.fx.GraphModule, *input_data) -> None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-8)

        for _ in range(11):
            loss = model(*input_data)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()

        start_time = time.time()
        loss = model(*input_data)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        duration = time.time() - start_time

        flops = hap.stat(model, {
            "input_shape": config.input_shape()
        })

        eprint(f"Profiling finished. Total flops: {flops}, wall time: {duration}")
        self.device_flops = math.floor(flops / duration)
        eprint("device flops: ", self.device_flops)

class BandwidthProfiler:
    def __init__(self, config, ranks, skewness) -> None:
        self.bandwidth = {}
        self.skewness = skewness

        for op in (
            collectives.all_gather,
            collectives.all_gather_by_group_call,
            collectives.all_reduce,
            collectives.reduce_scatter,
            collectives.reduce_scatter_by_group_call,
            collectives.all_to_all
        ):
            estimation = []
            for size in (4*1024*1024, 16*1024*1024, 64*1024*1024, 256*1024*1024):
                ts = [ self.run_collective(config, ranks, op, size) for _ in range(5) ]
                eprint((size, sorted(ts)))
                estimation.append(size / sorted(ts)[2])
            self.bandwidth[op.__name__] = math.floor(sum(estimation) / len(estimation))
        eprint(self.bandwidth)

    def run_collective(self, config, ranks, op, size: int) -> float:
        import os
        os.environ['MASTER_ADDR'] = str(config.master_addr)
        os.environ['MASTER_PORT'] = str(config.master_port)
        os.environ['WORLD_SIZE'] = str(config.world_size)

        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        queue = ctx.Queue(1)

        for local_rank, global_rank in enumerate(ranks):
            ctx.Process(target=_run_collective_worker, args=(op, size, self.skewness, queue, global_rank, local_rank)).start()

        for p in mp.active_children():
            p.join()

        return queue.get()

def _run_collective_worker(op, size: int, skewness: float, queue, global_rank: int, local_rank: int):
    import torch.distributed as dist
    dist.init_process_group('nccl', rank=global_rank)

    if op is collectives.all_reduce:
        tensor = torch.rand(256, size // 1024).to(local_rank)
        op_args = ()

    if op in (collectives.all_gather, collectives.all_gather_by_group_call):
        total_length = size // 1024
        sharding_lengths = [skewness] + [1] * (config.world_size - 1)
        sharding_lengths = [ x / sum(sharding_lengths) for x in sharding_lengths]
        hap.sharding_round(total_length, sharding_lengths)

        tensor = torch.rand(256, sharding_lengths[global_rank]).to(local_rank)
        op_args = (1, sharding_lengths, global_rank)

    if op in (collectives.reduce_scatter, collectives.reduce_scatter_by_group_call):
        total_length = size // 1024
        sharding_lengths = [skewness] + [1] * (config.world_size - 1)
        sharding_lengths = [ x / sum(sharding_lengths) for x in sharding_lengths]
        hap.sharding_round(total_length, sharding_lengths)

        tensor = torch.rand(256, total_length).to(local_rank)
        op_args = (1, sharding_lengths, global_rank)

    if op is collectives.all_to_all:
        total_length = size // 1024
        split_sharding_lengths = [skewness] + [1] * (config.world_size - 1)
        split_sharding_lengths = [ x / sum(split_sharding_lengths) for x in split_sharding_lengths]
        hap.sharding_round(256, split_sharding_lengths)

        cat_sharding_lengths = [skewness] + [1] * (config.world_size - 1)
        cat_sharding_lengths = [ x / sum(cat_sharding_lengths) for x in cat_sharding_lengths]
        hap.sharding_round(total_length, cat_sharding_lengths)

        tensor = torch.rand(256, cat_sharding_lengths[global_rank]).to(local_rank)
        op_args = (0, 1, split_sharding_lengths, cat_sharding_lengths, global_rank)

    for _ in range(5): # 4 warmup rounds
        start_time = time.time()
        op(tensor, *op_args)
        torch.cuda.synchronize(local_rank)
        duration = time.time() - start_time

    if local_rank == 0:
        queue.put(duration)


if __name__ == '__main__':
    if len(argv) >= 2:
        ranks = [ int(x) for x in argv[1].split(',') ]
        skewness = 1 # float(argv[2])

        # if torch.cuda.device_count() != len(ranks):
        #     eprint("forget to set CUDA_VISIBLE_DEVICES")
        #     raise SystemExit

        profiler = BandwidthProfiler(config, ranks, skewness)
        # save("bandwidth_profiler", profiler)
        raise SystemExit

    # assert config.world_size == 1

    model = hap.trace(config.get_model()).cuda(0)
    x, y = next(config.get_data()[1])
    profiler = FlopsProfiler(model, x.cuda(0), y.cuda(0))
    # save("flops_profiler", profiler)
