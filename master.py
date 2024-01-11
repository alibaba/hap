import config
import torch
import torch.fx
import math
import time
import hap

def eprint(*args, **kwargs):
    import sys
    print(*args, file=sys.stderr, **kwargs)

model = hap.trace(config.get_model(seed=39))

eprint(model.code)
eprint("Total Number of Ops:", len(model.graph.nodes))
eprint("Total parameters:", sum(math.prod(p.size()) for p in model.parameters()))

flops = hap.stat(model, {
    "input_shape": config.input_shape()
})

eprint("Total flops:", flops)

start_time = time.time()

dgraph = hap.main(model, {
    "input_shape": config.input_shape(),
    # "device_flops": [ 3858755112937 ] * round(config.world_size / 8 * 2) + [ 2149250936815 ] * round(config.world_size / 8 * 6),
    "device_flops": [ 5966645808610 ] * config.world_size,
    # "device_flops": [ 5712013967207, 1, 3858755112937, 1 ],
    # "device_flops": [ 3858755112937, 3858755112937, 1, 1, 1 ] * 2 + [ 2149250936815, 2149250936815, 1, 1, 1 ] * 6,
    "all_gather_bandwidth": 9463184466,
    "all_gather_by_group_call_bandwidth": 9146232360,
    "all_reduce_bandwidth": 5906161310,
    "reduce_scatter_bandwidth": 9740459122,
    "reduce_scatter_by_group_call_bandwidth": 8756865799,
    "all_to_all_bandwidth": 24588140722,

    "extra_ps": False,
    "group_collective": False,

    "rank": 0,
})

eprint(dgraph)

eprint("\nTime: ", time.time() - start_time)

# dmodel = torch.fx.GraphModule(model, dgraph)
