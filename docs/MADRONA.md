# Madrona MJX Walkthrough

## Overview

Madrona MJX integrates the Madrona engine with Mujoco MJX to provide high-throughput batch rendering within MJX for training vision-based policies. 

At its core, the renderer features the following:
- The ability to render images in a batched fashion, meaning that it can render thousands of images at the same time at high throughput.
- Simple lighting with shadows (configurable).
- Both directional and spotlights which can move.
- Variable number of cameras per world.

This walkthrough will guide you through the key components and structure of this project.

Madrona utilizes a GPU-based entity-component-system (ECS) where environment state is stored in a unique pattern in contiguous GPU memory. This state-of-the-art GPU memory structure allows for high throughput rendering through efficient data access by the Madrona Batch Renderer.

Even though Madrona is itself a batch simulation framework, we can utilize just its rendering capabilities as a standalone component to attach to other simulation frameworks. Everything under the `src` directory serves to translate MJX's state representation into Madrona's own state representation, which will allow for the high-throughput rendering. Because the end user (who performs their simulation in MJX) will just interface with the `madrona_mjx` python modules to do their rendering, we will refer to this module as the renderer interface.

Fundamentally, the end user of the rendering interface should expect the following usage pattern (pseudocode):

```python
# Initialize the physics simulator
my_simulator = Simulator(...)

# Initialize the Madrona renderer
renderer = MadronaBatchRenderer(geometry, num_instances, ...)
renderer.init(my_simulator.initial_state())

for _ in range(num_steps):
    # Step the simulation
    my_simulator.step()

    # Render the output of the current simulation step
    renderer.render(my_simulator.get_state())

    # Retrieve the output tensors
    rendered_output_tensors = renderer.rgbd_tensors()
```

Now, we go into detail as to how the interface works.

## Code Walkthrough

### Bindings `src/bindings.cpp`

The `bindings.cpp` file defines classes and functions which can be invoked from Python once the end user imports `_madrona_mjx_batch_renderer`. The primary class that is declared here is `MadronaBatchRenderer` whose constructor takes in all the geometry (triangle meshes, textures, uvs, etc...) that will be rendered during the course of the simulation rollout. It provides simple wrappers to expose RGB and depth tensors, as well as instance position/rotation tensors. The main functions the end user will be concerned with are `init` and `render`. The `init` function not only initializes instance transforms but also cameras and lights information across all the worlds. The `render` function takes in the new instance and camera transforms which should be used to render the next batch of frames.

The class `MadronaBatchRenderer` itself, can be thought of as a thin wrapper layer over the C++ class `Manager` which we describe below.

### Manager `src/mgr.cpp`

The `Manager` class is the orchestration class. Its responsibilities are to:

- Initialize Madrona's core internal components
- Load the geometry (do some data post-processing)
- Perform copies of the simulator state into pre-allocated buffers for further processing (done in `sim.cpp`)
- Invoke its batched renderer
- Expose important data to be passed up the chain to Python through the previously mentioned `bindings.cpp`

In order to invoke the batch renderer (which also entails doing the further processing on copied in transforms), the `Manager` class needs to provide an entry point to the functionality that is declared in `sim.cpp`.

### `sim.cpp` and invoking the batch renderer

The code in `sim.cpp` serves one main purpose: it defines the `Taskgraph`, a series of functions which will run on the GPU, which in this case, serves to transform the data provided by the simulator into something that the Madrona renderer can understand. Namely, the Madrona renderer requires that all rendered instances and cameras be described using components and archetypes (which concretely translate into specially managed contiguous arrays of data residing on the GPU).

## Integration with other simulators

There are a few main considerations when integrating Madrona's renderer with a custom simulator.

### Loading geometry

The first is loading geometry in a way that Madrona understands. MJX has its own format for describing geometry for a model. Perhaps the simulator you'd like to integrate has its own geomtery data format as well. Therefore, the first step is modifying this process. This will require some manual modifications in `mgr.cpp` to do this. Particularly, you will want to modify the `loadRenderObjects` function [here](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/mgr.cpp#L471). Madrona's geometry format works as follows:

First, there's the mesh. A mesh is a collection of arrays. One for triangles, one for normals, UVs etc... This could represent a sphere, capsule, or any other mesh.

Next, there are objects. These are a group of meshes. All meshes within an object will be rendered as a single thing. For instance, if you defined a robot object such that its legs, arms, torso and head are all in different vertex arrays, you could create 4 meshes and attach them all to a single object to be rendered as a single thing.

Finally, there are instances. Say you have 4 robot entities in a world. Each entity will be an instance of the robot object and when rendering happens, one just needs to inform the renderer of the instance's object ID (you can see this being configured in the `Sim` class's [constructor](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/sim.cpp#L145)).

The core responsibility of the asset loading is to create the [`ImportedAssets`](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/mgr.cpp#L499) structure which can be created in two ways:

- Through a Madrona-provided class [`AssetImporter`](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/mgr.cpp#L488) which can load geomtry from disk (obj, gltf, etc...).
- By manually filling out the `ImportedAssets` structure which is what we do for the MJX model.

### Configuring the scene

In order to configure the scene, you will need to create the proper ECS constructs in `sim.cpp`. This can be seen inside the `Sim` class's [constructor](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/sim.cpp#L135). The main idea is to simply create entities for all instances that need to be rendered, configure their object ID and other basic properties. The order in which you create these entities is important and needs to stay consistent as you copy in updated transforms during the simulation rollout - we describe this in the next subsection.

### Passing in updated transforms to the render function

In order to invoke the renderer, one needs to call `MadronaBatchRenderer`'s `render` function which takes in tensors of instance position/rotation and camera position/rotation.

Take for instance the position tensor. This needs to be a contiguous tensor of all x/y/z positions of all instances across all worlds. All positions need to first be ordered by world. Within each world, they need to be ordered in the order you defined in the `Sim` constructor we described previously.

The same can be said for rotations though instead of x/y/z, we expect w/x/y/z quaternions.

### Retrieving the rendered output tensors

The rendered output tensors can be retrieved through `MadronaBatchRenderer`'s tensor getter methods. The only thing to note is that the order of these images will be consistent with the ordering defined in `Sim`'s constructor.

### Deep learning frameworks

One can pass tensors from either PyTorch or Jax.