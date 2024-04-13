# Madrona MJX Prototype

## Installation

```sh
git clone --recursive git@github.com:shacklettbp/madrona_mjx.git

cd madrona_mjx
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
MADRONA_MWGPU_KERNEL_CACHE=build/cache python scripts/viewer.py --num-worlds 16 --window-width 2730 --window-height 1536 --batch-render-view-width 64 --batch-render-view-height 64
```

Headless profiling:

```sh
MADRONA_MWGPU_KERNEL_CACHE=build/cache python scripts/headless.py --num-worlds 1024 --num-steps 1000 --batch-render-view-width 64 --batch-render-view-height 64
```
