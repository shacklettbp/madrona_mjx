# Integration with other simulators

Before reading this document, we suggest you read the [walkthrough](https://github.com/shacklettbp/madrona_mjx/blob/main/docs/WALKTHROUGH.md) document.

There are a few main considerations when integrating Madrona's renderer with a custom simulator.

## Define the state

There are a couple important terms that we need to define in order to fully specify all parts of the state:
- *Environments*: a namespace to encapsulate a set of object instances and views.
- *Objects*: 3D assets (made of triangle meshes and optionally textures). They are referred to by a global ID, and aren't specific to any environment.
- *Instances*: these are associated with a specific environment and maintain a position and rotation to define its transform in the environemnt it belongs to.
A single *object* may be instanced many times, across many environments. Instances therefore also maintain its *object* ID that defines asset to render
for this instance.
- *View*: camera parameters which are also specific to an environment. Each environment can have multiple views. The batch renderer is responsible for rendering
output frames for all views across all environments. Currently, views share the same width/height and FOV.

## Defining the Objects (Geometry)

This is the first step in integrating another simulator. Geometry needs to be loaded in a way that Madrona understands.
MJX has its own format for describing geometry for a model. Perhaps the simulator you'd like to integrate has its own geomtery data format as well. Therefore, the first step is modifying this process. This will require some manual modifications in `mgr.cpp` to do this. Particularly, you will want to modify the `loadRenderObjects` function [here](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/mgr.cpp#L471). Madrona's geometry format works as follows:

First, there's the mesh. A mesh is a collection of arrays. One for triangles, one for normals, UVs etc... This could represent a sphere, capsule, or any other mesh.

Next, there are objects, as we defined in the previous subsection. These are a group of meshes. All meshes within an object will be rendered as a single thing. For instance, if you defined a robot object such that its legs, arms, torso and head are all in different vertex arrays, you could create 4 meshes and attach them all to a single object to be rendered as a single thing.

The core responsibility of your custom asset loading code is to create an array of `SourceObject` (which represents an object) that can be created in two ways:

- Through a Madrona-provided class [`AssetImporter`](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/mgr.cpp#L488) which can load geomtry from disk (obj, gltf, etc...). This will create the `ImportedAssets` structure which will have such an array created already.
- By manually filling out this array.
- A mixture of both using the built-in `AssetImporter` and manual configuration which is what we do in the MJX case.

Once this array has been created, you can pass it to the `AssetProcessor`'s BVH utility functions to create the [final structures](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/mgr.cpp#L704) which need to be passed to the renderer's [initialization](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/mgr.cpp#L779).

## Configuring the environments

In order to configure the environment, you will need to create the proper ECS constructs in `sim.cpp`. This can be seen inside the `Sim` class's [constructor](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/sim.cpp#L135). The main idea is to simply create entities for all instances that need to be rendered, configure their object ID and other basic properties (such as initial position). The order in which you [create](https://github.com/shacklettbp/madrona_mjx/blob/cf7a70291ba2d24f8c4f2bc76488a3dcfa9b3481/src/sim.cpp#144) these entities is important and needs to stay consistent as you copy in updated transforms during the simulation rollout.

## Passing in updated transforms to the render function

In order to invoke the renderer, one needs to call `MadronaBatchRenderer`'s `render` function which takes in tensors of instance position/rotation and camera position/rotation.

Take for example the position tensor. This needs to be a contiguous tensor of all x/y/z positions of all instances across all worlds. All positions need to first be ordered by world. Within each world, they need to be ordered in the order you created in the `Sim` constructor we described previously.

The same can be said for rotations though instead of x/y/z, we expect w/x/y/z quaternions.

## Passing in updated transforms to the `init` function

To enable domain randomization across worlds, `madrona_mjx` implements a way to set
unique data per world such that you can have different visual properties per world.
These can be different lighting conditions, textures, colors, sizes, etc...

The visual properties are only updated once in the `init` function and other
state like the instance or camera transforms are updated in `render` every frame.

If you wish to update visual properties every frame as well, you would need
to modify the `render` function to take in those properties and update them
accordingly in `sim.cpp`.

## Retrieving the rendered output tensors

The rendered output tensors can be retrieved through `MadronaBatchRenderer`'s tensor getter methods. The only thing to note is that the order of these images will be consistent with the ordering defined in `Sim`'s constructor.

## Deep learning frameworks

One can pass tensors from either PyTorch or Jax.