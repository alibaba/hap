import os
import sys
import math

rootpath = "/root/hap"
sys.path.insert(1, rootpath)

# model_name = "Tmlp"
# model_name = "Ttransformer"
model_name = "Rtransformer"
# model_name = "Rmoe"
# model_name = "Rswitch"
# model_name = "Vtransformer"
# model_name = "Vmoe"
# model_name = "Vswitch"
# model_name = "Vvgg"

world_size = 8
nlayers = 12
n_expert = 2 * world_size
batch_size = 32 * world_size
seqlen = 128
if model_name.startswith('V'):
    seqlen = 64
capacity_factor = 1.25
if model_name.endswith('moe'):
    capacity_factor *= 2
capacity = math.ceil(seqlen / n_expert * capacity_factor)
emsize = 768
# emsize = 960
nhid = emsize * 4

dropout = 0.1
nheads = 12

master_addr = "127.0.0.1"
master_port = 39266

# segmentation = True
segmentation = False

# trace = True
trace = False

report_per_iter_time = True
# report_per_iter_time = False

lr = 5e-4

run_iter = 100
avg_iter = 50
log_iter = 100

def get_model(seed=None):
    import models

    if seed is not None:
        import torch
        torch.manual_seed(seed)

    if model_name == 'Tmlp':
        return models.TMLP(nhid=emsize, nlayers=nlayers, segmentation=segmentation)
    if model_name == 'Tmlp2':
        return models.TMLP2(nhid=emsize, nlayers=nlayers, segmentation=segmentation)
    if model_name == 'Ttransformer':
        return models.TTransformer(emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, nlayers=nlayers, segmentation=segmentation)
    if model_name == 'Tmoe':
        return models.TMoE(emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers, segmentation=segmentation)

    if model_name == 'Rtransformer':
        ntokens, *_ = get_data()
        return models.RTransformer(ntokens=ntokens, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, nlayers=nlayers, segmentation=segmentation)
    if model_name == 'Rmoe':
        ntokens, *_ = get_data()
        return models.RMoE(ntokens=ntokens, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers, segmentation=segmentation)
    if model_name == 'Rswitch':
        ntokens, *_ = get_data()
        return models.RSwitch(ntokens=ntokens, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers, segmentation=segmentation)

    if model_name == 'Vtransformer':
        nclasses, *_ = get_data()
        return models.VTransformer(nclasses=nclasses, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, nlayers=nlayers, segmentation=segmentation)
    if model_name == 'Vmoe':
        nclasses, *_ = get_data()
        return models.VMoE(nclasses=nclasses, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers, segmentation=segmentation)
    if model_name == 'Vswitch':
        nclasses, *_ = get_data()
        return models.VSwitch(nclasses=nclasses, seqlen=seqlen, emsize=emsize, nheads=nheads, nhid=nhid, dropout=dropout, n_expert=n_expert, capacity=capacity, nlayers=nlayers, segmentation=segmentation)
    if model_name == 'Vvgg':
        nclasses, *_ = get_data()
        return models.VVGG(nclasses=nclasses, dropout=dropout, segmentation=segmentation)

def get_data():
    if model_name.startswith('R'):
        return wikitext2()

    if model_name.startswith('V'):
        return cifar10()

    if model_name.startswith('T'):
        import torch
        x = torch.rand(batch_size, seqlen, emsize) / 6
        y = torch.rand(batch_size)
        def rep():
            while True:
                yield x, y
        return 0, rep()

def wikitext2():
    sys.path.insert(1, f"{rootpath}/wikitext")
    import data
    corpus = data.Corpus(f"{rootpath}/wikitext")
    train_data = data.segmentify(data.batchify(corpus.train, batch_size), seqlen)
    test_data = data.segmentify(data.batchify(corpus.test, batch_size), seqlen)
    valid_data = data.segmentify(data.batchify(corpus.valid, batch_size), seqlen)
    ntokens = world_size * (len(corpus.dictionary) // world_size + 1) # we have to ensure that it is dividable
    return ntokens, train_data, test_data, valid_data

def cifar10():
    import torch
    import torchvision
    def it(data):
        loader = torch.utils.data.DataLoader(data, batch_size=batch_size, drop_last=True)
        while True:
            yield from iter(loader)
    train_data = torchvision.datasets.CIFAR10(f"{rootpath}/cifar10", train=True, transform=torchvision.transforms.ToTensor()) #, download=True
    test_data = torchvision.datasets.CIFAR10(f"{rootpath}/cifar10", train=False, transform=torchvision.transforms.ToTensor()) #, download=True
    return 10, it(train_data), it(test_data)

def input_shape():
    if model_name.startswith('R'):
        return { 'x': (batch_size, seqlen), 'y': (batch_size, seqlen) }
    if model_name.startswith('V'):
        return { 'x': (batch_size, 3, 32, 32), 'y': (batch_size,) }
    if model_name.startswith('T'):
        return { 'x': (batch_size, seqlen, emsize), 'y': (batch_size,) }
