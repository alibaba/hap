from sys import argv
import json

trace = json.load(open(argv[1]))

class StreamInfo:
    def __init__(self) -> None:
        self.time = 0.
        self.ops = set()

stream_info = {}

for event in trace['traceEvents']:
    if event['pid'] != 0 or event['tid'] == 0:
        continue

    stream = event['tid']
    if stream not in stream_info:
        stream_info[stream] = StreamInfo()

    if 'dur' not in event:
        continue

    info = stream_info[stream]
    info.time += event['dur'] / 1_000_000
    info.ops.add(event['name'])

for stream, info in stream_info.items():
    print(f"=== {stream} ===")
    print(f"total_time: {info.time:.4f}s")
    for name in sorted(list(info.ops), key=lambda x: len(x))[:5]:
        print(name)
    print()

nccl_time = 0.
copy_device_to_device_time = 0.

for event in trace['traceEvents']:
    if event['pid'] != 0 or event['tid'] == 0 or 'dur' not in event:
        continue

    if 'nccl' in event['name']:
        nccl_time += event['dur'] / 1_000_000

    if 'copy_device_to_device' in event['name']:
        copy_device_to_device_time += event['dur'] / 1_000_000

print("Total Nccl time:", nccl_time)
print("Total Copy time:", copy_device_to_device_time)

steps = []
for event in trace['traceEvents']:
    if 'ProfilerStep' in event['name']:
        steps.append(event['dur'] / 1_000_000)

print("Avg Wall time: ", sum(steps) / len(steps))

comm = nccl_time + copy_device_to_device_time
comp = sum([x.time for x in stream_info.values()]) - comm
print("Comp time:", comp)
print("Comm time:", comm)
print("Comp-Comm ratio:", comp / comm)
