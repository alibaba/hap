HAP: SPMD DNN Training on Heterogeneous GPU Clusters with Automated Program Synthesis
=====================================================================================

## Build From Source

HAP is partially implemented in Rust and requires a Rust compiler to build. The building toolchain can be installed by
following https://rustup.rs. Currently HAP requires a nightly build of the Rust compiler.

Run the following command to build HAP:

```
cargo build --release
```

## Installing Python Dependencies

HAP is implemented on top of PyTorch 1.13.1. It can be installed, along with its own dependencies, by following
https://pytorch.org/get-started/previous-versions/#v1131.

Alternatively, we provides an environment file `environment.yml` for reproducing the experiment environment with conda
(https://conda.io).


## Contributing

See CONTRIBUTING.md for details.

## Contributors

HAP is developed by Alibaba Group and HKU Netexplo. This work is supported by [Alibaba Innovative Research(AIR)](https://damo.alibaba.com/air/).

## License

HAP is licensed under the Apache License (Version 2.0). See LICENSE file.
This product contains some third-party testcases under other open source licenses.
See the NOTICE file for more information.


