from __future__ import annotations

import torch
import torch.distributed as dist

def sharded_shape(shape, dim, length):
    return shape[:dim] + (length,) + shape[dim+1:]

def padded(tensor, dim, length):
    if tensor.shape[dim] == length:
        return tensor
    return torch.cat([ tensor, torch.empty(*sharded_shape(tensor.shape, dim, length - tensor.shape[dim]), device=tensor.device) ], dim=dim)

class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, sharding_lengths, rank):
        ctx.dim, ctx.sharding_lengths, ctx.rank = dim, sharding_lengths, rank
        out_tensor_slices = [ torch.empty(*sharded_shape(tensor.shape, dim, max(sharding_lengths)), device=tensor.device) for _ in sharding_lengths ]
        dist.all_gather(out_tensor_slices, padded(tensor, dim, max(sharding_lengths)).contiguous())
        return torch.cat([ out_tensor_slice.split([sharding_length, max(sharding_lengths) - sharding_length], dim=dim)[0] for out_tensor_slice, sharding_length in zip(out_tensor_slices, sharding_lengths) ], dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_slices = torch.split(grad_output.contiguous(), ctx.sharding_lengths, dim=ctx.dim)
        grad_output_slices_padded = [ padded(grad_output_slice, ctx.dim, max(ctx.sharding_lengths)).contiguous() for grad_output_slice in grad_output_slices ]
        grad = torch.empty_like(grad_output_slices_padded[0])
        dist.reduce_scatter(grad, [ x.contiguous() for x in grad_output_slices_padded ])
        grad = grad.split([ctx.sharding_lengths[ctx.rank], max(ctx.sharding_lengths) - ctx.sharding_lengths[ctx.rank]], dim=ctx.dim)[0]
        return grad, None, None, None

class AllGatherByGroupCall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, sharding_lengths, rank):
        ctx.dim, ctx.sharding_lengths, ctx.rank = dim, sharding_lengths, rank
        tensor_slices = [
            torch.empty(*sharded_shape(tensor.shape, dim, sharding_length), device=tensor.device)
            if i != rank else tensor
            for i, sharding_length in enumerate(sharding_lengths)
        ]

        reqs = []
        with dist.distributed_c10d._coalescing_manager(None, reqs):
            for i, t in enumerate(tensor_slices):
                req = dist.broadcast(t, i, async_op=True) # TODO: try sync version?
                reqs.append(req)

        for req in reqs: # https://pytorch.org/docs/stable/distributed.html#synchronous-and-asynchronous-collective-operations In the case of CUDA collectives, will block until the operation has been successfully enqueued onto a CUDA stream and the output can be utilized on the default stream without further synchronization.
            req.wait()

        return torch.cat(tensor_slices, dim=dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_slices = torch.split(grad_output, ctx.sharding_lengths, dim=ctx.dim)
        grad_output_slices = [ x.contiguous() for x in grad_output_slices ]

        reqs = []
        with dist.distributed_c10d._coalescing_manager(None, reqs):
            for i, grad_output_slice in enumerate(grad_output_slices):
                req = dist.reduce(grad_output_slice, i, async_op=True)
                reqs.append(req)

        for req in reqs:
            req.wait()

        return grad_output_slices[ctx.rank], None, None, None


# aliasing prevents assigning __module__, which is required by fx.node.Node.__repr__, otherwise it crashes
def all_gather(tensor, dim, sharding_lengths, rank): return AllGather.apply(tensor, dim, sharding_lengths, rank)
def all_gather_by_group_call(tensor, dim, sharding_lengths, rank): return AllGatherByGroupCall.apply(tensor, dim, sharding_lengths, rank)

class AllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        out_tensor = tensor.contiguous()
        dist.all_reduce(out_tensor)
        return out_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.clone()
        dist.all_reduce(grad)
        return grad

def all_reduce(tensor): return AllReduce.apply(tensor)

class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, sharding_lengths, rank):
        ctx.dim, ctx.sharding_lengths, ctx.rank = dim, sharding_lengths, rank
        tensor_slices = torch.split(tensor.contiguous(), sharding_lengths, dim=dim)
        tensor_slices_padded = [ padded(tensor_slice, dim, max(sharding_lengths)).contiguous() for tensor_slice in tensor_slices ]
        out = torch.empty_like(tensor_slices_padded[0])
        dist.reduce_scatter(out, [ x.contiguous() for x in tensor_slices_padded ])
        return out.split([sharding_lengths[rank], max(sharding_lengths) - sharding_lengths[rank]], dim=dim)[0]

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_slices = [ torch.empty(*sharded_shape(grad_output.shape, ctx.dim, max(ctx.sharding_lengths)), device=grad_output.device) for _ in ctx.sharding_lengths ]
        dist.all_gather(grad_output_slices, padded(grad_output.contiguous(), ctx.dim, max(ctx.sharding_lengths)).contiguous())
        grad = torch.cat([ grad_output_slice.split([sharding_length, max(ctx.sharding_lengths) - sharding_length], dim=ctx.dim)[0] for grad_output_slice, sharding_length in zip(grad_output_slices, ctx.sharding_lengths) ], dim=ctx.dim)
        return grad, None, None, None

class ReduceScatterByGroupCall(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, sharding_lengths, rank):
        ctx.dim, ctx.sharding_lengths, ctx.rank = dim, sharding_lengths, rank
        tensor_slices = torch.split(tensor, sharding_lengths, dim=dim)
        tensor_slices = [ x.contiguous() for x in tensor_slices ]

        reqs = []
        with dist.distributed_c10d._coalescing_manager(None, reqs):
            for i, t in enumerate(tensor_slices):
                req = dist.reduce(t, i, async_op=True)
                reqs.append(req)

        for req in reqs:
            req.wait()

        return tensor_slices[rank]

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_slices = [
            torch.empty(*sharded_shape(grad_output.shape, ctx.dim, sharding_length), device=grad_output.device)
            if i != ctx.rank else grad_output.contiguous()
            for i, sharding_length in enumerate(ctx.sharding_lengths)
        ]

        reqs = []
        with dist.distributed_c10d._coalescing_manager(None, reqs):
            for i, grad_output_slice in enumerate(grad_output_slices):
                req = dist.broadcast(grad_output_slice, i, async_op=True)
                reqs.append(req)

        for req in reqs:
            req.wait()

        return torch.cat(grad_output_slices, dim=ctx.dim), None, None, None

def reduce_scatter(tensor, dim, sharding_lengths, rank): return ReduceScatter.apply(tensor, dim, sharding_lengths, rank)
def reduce_scatter_by_group_call(tensor, dim, sharding_lengths, rank): return ReduceScatterByGroupCall.apply(tensor, dim, sharding_lengths, rank)

# Not really a collective operator
def dynamic_slice(tensor, dim, sharding_lengths, rank):
    tensor_slices = torch.split(tensor, sharding_lengths, dim=dim)
    return tensor_slices[rank].contiguous()

# Actually there is an "all_to_all_single" that do the chunking and cating for us: https://github.com/pytorch/pytorch/blob/master/torch/distributed/distributed_c10d.py#L2404
# similar versions exist for other collectives. They should be preferred in terms of performance (and deepspeed uses them)
class AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, split_dim, cat_dim, split_sharding_lengths, cat_sharding_lengths, rank):
        ctx.split_dim, ctx.cat_dim, ctx.split_sharding_lengths, ctx.cat_sharding_lengths, ctx.rank = split_dim, cat_dim, split_sharding_lengths, cat_sharding_lengths, rank
        tensor_slices = torch.split(tensor.contiguous(), split_sharding_lengths, dim=split_dim)
        out_slices = [ torch.empty(*sharded_shape(tensor_slices[rank].shape, cat_dim, length), device=tensor.device) for length in cat_sharding_lengths ]
        dist.all_to_all(out_slices, [ x.contiguous() for x in tensor_slices ])
        return torch.cat(out_slices, dim=cat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_slices = torch.split(grad_output.contiguous(), ctx.cat_sharding_lengths, dim=ctx.cat_dim)
        grad_slices = [ torch.empty(*sharded_shape(grad_output_slices[ctx.rank].shape, ctx.split_dim, length), device=grad_output.device) for length in ctx.split_sharding_lengths ]
        dist.all_to_all(grad_slices, [ x.contiguous() for x in grad_output_slices ])
        return torch.cat(grad_slices, dim=ctx.split_dim), None, None, None, None, None

def all_to_all(tensor, split_dim, cat_dim, split_sharding_lengths, cat_sharding_lengths, rank): return AllToAll.apply(tensor, split_dim, cat_dim, split_sharding_lengths, cat_sharding_lengths, rank)

# the "f" function in the Megatron paper, which is identical function in forward, but all-reduce in backward. It is used for tensors that are replicated before entering the funtion (input and parameters)
class Replicate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad = grad_output.contiguous()
        dist.all_reduce(grad)
        return grad

def replicate(tensor): return Replicate.apply(tensor)


# simple tests on 4 GPUs

def test(rank):
    import torch.distributed as dist
    import datetime
    dist.init_process_group('nccl', rank=rank, timeout=datetime.timedelta(hours=2))

    class Mod1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(5, rank+1))

        def forward(self):
            return all_gather(self.p, 1, [1,2,3,4], rank).sum() / 4

    print("testing all_gather")
    mod = Mod1().cuda(rank)
    loss = mod.forward()
    loss.backward()
    print(rank, loss, mod.p.grad, flush=True) # expecting: losses are the same, grads are 1
    dist.barrier()

    class Mod1v2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(5, rank+1))

        def forward(self):
            return all_gather_by_group_call(self.p, 1, [1,2,3,4], rank).sum() / 4

    print("testing all_gather_by_group_call")
    mod = Mod1v2().cuda(rank)
    loss = mod.forward()
    loss.backward()
    print(rank, loss, mod.p.grad, flush=True) # expecting: losses are the same, grads are 1
    dist.barrier()

    class Mod2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(5, 4) * rank)

        def forward(self):
            return all_reduce(self.p).sum() / 4

    print("testing all_reduce")
    mod = Mod2().cuda(rank)
    loss = mod.forward()
    loss.backward()
    print(rank, loss, mod.p.grad, flush=True) # expecting: losses are the same, grads are 1
    dist.barrier()

    class Mod3(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(5, 10))

        def forward(self):
            return reduce_scatter(self.p, 1, [1,2,3,4], rank).sum()

    print("testing reduce_scatter")
    mod = Mod3().cuda(rank)
    loss = mod.forward()
    loss.backward()
    print(rank, loss, mod.p.grad, flush=True) # expecting: losses propotional to rank, grads are 1
    dist.barrier()

    class Mod3v2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(5, 10))

        def forward(self):
            return reduce_scatter_by_group_call(self.p, 1, [1,2,3,4], rank).sum()

    print("testing reduce_scatter_by_group_call")
    mod = Mod3v2().cuda(rank)
    loss = mod.forward()
    loss.backward()
    print(rank, loss, mod.p.grad, flush=True) # expecting: losses propotional to rank, grads are 1
    dist.barrier()

    class Mod4(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(5, 10))

        def forward(self):
            return dynamic_slice(self.p, 1, [1,2,3,4], rank).sum()

    print("testing dynamic_slice")
    mod = Mod4().cuda(rank)
    loss = mod.forward()
    loss.backward()
    print(rank, loss, mod.p.grad, flush=True) # expecting: losses propotional to rank, grads are partially 1
    dist.barrier()

    class Mod5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(10, rank+1))

        def forward(self):
            return all_to_all(self.p, 0, 1, [4,3,2,1], [1,2,3,4], rank).sum()

    print("testing all_to_all")
    mod = Mod5().cuda(rank)
    loss = mod.forward()
    loss.backward()
    print(rank, loss, mod.p.grad, flush=True) # expecting: losses are reverse propotional to rank, grads are 1
    dist.barrier()

    class Mod5_v2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(10, [0,2,0,8][rank]))

        def forward(self):
            return all_to_all(self.p, 0, 1, [5,0,5,0], [0,2,0,8], rank)

    print("testing all_to_all with zero")
    mod = Mod5_v2().cuda(rank)
    result = mod.forward()
    print(rank, mod.p.shape, result.shape, flush=True)
    dist.barrier()

    class Mod6(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.ones(5, 5))

        def forward(self):
            return replicate(self.p).sum() / 4

    print("testing replicate")
    mod = Mod6().cuda(rank)
    loss = mod.forward()
    loss.backward()
    print(rank, loss, mod.p.grad, flush=True) # expecting: losses are the same, grads are 1
    dist.barrier()

if __name__ == '__main__':
    if torch.cuda.device_count() < 4:
        print("Not enough GPUs")
        raise SystemExit

    import os
    os.environ['MASTER_ADDR'] = "127.0.0.1"
    os.environ['MASTER_PORT'] = "39393"
    os.environ['WORLD_SIZE'] = "4"

    import torch.multiprocessing as mp
    mp.set_start_method('spawn')

    for rank in range(4):
        mp.Process(target=test, args=(rank, )).start()

    for p in mp.active_children():
        p.join()
