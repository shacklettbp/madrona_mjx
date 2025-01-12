# Madrona MJX

## Overview
Madrona MJX is an integration of the [Madrona](https://madrona-engine.github.io) engine with
Mujoco MJX to provide high-throughput batch rendering within mjx for training vision-based
policies. See this document for a detailed [walkthrough](hhttps://github.com/shacklettbp/madrona_mjx/blob/main/docs/MADRONA.md) of the project.

## Features
- High throughput Batch Renderer
- Raytracer (recommended) and Rasterizer backend options
- Integration with MJX and Brax pipelines
- Domain Randomization capabilities for vision properties

## Installation

Currently Madrona-MJX requires building the renderer from source, then locally installing a Python wrapper for the renderer.

### Prerequisites

**Cuda**
The Cuda toolkit is required to build the renderer. Use [Cuda 12.5.1]([url](https://developer.nvidia.com/cuda-12-5-1-download-archive)) or earlier. [Cudnn](https://developer.nvidia.com/cudnn) will be required for the upcoming local Jax install.

**Jax**
Naively running `pip install jax["cuda12"]` can result in Jax pulling a different version of the Cuda binaries than the toolkit binaries used to build the wrapper. To prevent this, after installing the Cuda Toolkit and Cudnn, run `pip install jax["cuda12_local"]`. 

**Cmake**
If `cmake..` below does not work, check your cmake version with `cmake --version` and try updating to at least [cmake 3.31.0](https://github.com/Kitware/CMake/releases/download/v3.31.0-rc2/cmake-3.31.0-rc2-linux-x86_64.sh).

### Source Installation
```sh
git clone https://github.com/shacklettbp/madrona_mjx.git

cd madrona_mjx
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j

cd ..
pip install -e .
```

## Usage

Madrona-MJX can be used in two ways, by launching a viewer that is hooked to 
a visualization loop, or headless inside of a training loop. Currently 
visualization during training is not supported

### Launching the viewer

A viewer script is provided that can be used to view mjcf files using the Madrona
viewer. Let's launch the viewer with a cartpole mjcf.

```sh
python scripts/viewer.py --mjcf data/cartpole.xml --num-worlds 16 --window-width 2730 --window-height 1536 --batch-render-view-width 64 --batch-render-view-height 64
```

The viewer includes a world view and a batch render view. *The viewer is not representative of what the true
batch renderers are outputting, as it uses a different renderer*. To visualize what the batch 
renderers are visualizing, we have included a small debug window that shows the output 
of the rgb and depth for the current world. The worlds can by cycled in the options UI.

We can also load robotics examples examples from the Mujoco Menagerie. Let's 
try loading a Franka Emika Panda robot scene.

```sh
./scripts/create_franka_example

python scripts/viewer.py --mjcf mujoco_menagerie/franka_emika_panda/mjx_single_cube_camera.xml --num-worlds 16 --window-width 2730 --window-height 1536 --batch-render-view-width 64 --batch-render-view-height 64
```

### Training

An example vision training pipeline is provided in the following colab.

Madrona-MJX integrates directly into brax training pipelines. To include the batch rendererer into a new environments you must:
1. Create the batch renderer when your environment is created, passing along the correct arguments.
2. Call the .init() method inside your environment reset, passing along the correct arguments.
3. Call the .render() method inside your environment step to recieve rgb + depth outputs.

To make integration easier, a specialized wrapper is provided that replaces the typical Brax VmapWrapper and DomainRandomizationWrapper.
The MadronaWrapper should be used in replacement to those other wrappers to vmap and initialize the renderer properly. MadronaWrapper 
optionally takes in a randomization function that can be used to randomize your model. viewer.py includes a domain randomization example for randomizing the size
and color of the floor.


## Important tips

- USE CACHE
- A camera must be included in the mjcf
- Domain randimization must include the correct setting of geom_rgba, geom_size, and geom_matid. Please see the viewer for an example.
- Only one renderer can be initialized at a time. This means two environments cannot be created that both use their own batch renderer instances. (e.g. train/eval)
- The number of worlds must be known and initialized at the very beginning. All future rendering will render all worlds, there is no way to disable or not render certain worlds. (implication is that train/eval must have same batch size)

## Feature Parity
Not every feature of MJX is carried over into the renderer.

The following features are supported:
- Rigid body position and rotation changes
- Camera position and rotation changes

The following features are on the way:
- Light position and rotation changes
- Light parameter domain randomization

The following features are *not* supported:
- Deformable bodies
- Particle systems
- Musles, Tendons, Composites (Except for and rigid body components)

