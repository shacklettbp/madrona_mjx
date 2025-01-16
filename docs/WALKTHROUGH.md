# Madrona MJX Code Walkthrough

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

We go into more detail regarding what exactly `sim.cpp` must do in the [integration](https://github.com/shacklettbp/madrona_mjx/blob/main/docs/INTEGRATION.md) document.