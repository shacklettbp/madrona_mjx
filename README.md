# Madrona MJX Prototype

## Installation

```sh
git clone --recursive git@github.com:shacklettbp/madrona_mjxtype.git

cd madrona_mjxtype
mkdir build
cd build
cmake ..
make -j

cd ..
pip install -e .
```


## Usage


Launch the viewer with the following command:

```sh
MADRONA_MWGPU_KERNEL_CACHE=build/cache python scripts/viewer.py --num-worlds 10 --window-width 2730 --window-height 1536
```

Headless profiling:

```sh
MADRONA_MWGPU_KERNEL_CACHE=build/cache python scripts/headless.py --num-worlds 16384 --num-steps 1000
```
