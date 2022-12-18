Requirements
-------------
Unix-like environment 
(tested on Arch Linux)

Nvidia GPU with Cuda 6.0 compatibility 
(tested using RTX 3060, though much earlier cards should work)


Compilation
-----------
(Only tested on Linux, though should be cross-compatible)

Install rustup/cargo (the Rust toolchain)
Install `cuda` package (nvcc must be installed)

Within the root folder (contains Cargo.toml, src/, resources/)

```
cargo build --release
```
Packages will be fetched automatically by cargo and resources/kernel.cu will
be compiled to a .ptx by nvcc


The resultant executable is placed within ./target/release as `wfc`

Execution
---------
Usage:
./wfc <SAMPLE> --flags

./wfc --help for detailed information

Example:
./wfc samples/sample_islands.txt -W64 -H64 -m ac3-cuda
Generates 64x64 unicode output to wfc_out using sample_islands.txt as unicode sample source

Visualization
-------------
Install Python 3 and the Pillow package
Modify the variables at the top of utilities/img_converter.py to point to the correct files

Within the utilities folder run

```
python img_converter.python
```
