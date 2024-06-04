#pragma once
#ifdef madmjx_mgr_EXPORTS
#define MGR_EXPORT MADRONA_EXPORT
#else
#define MGR_EXPORT MADRONA_IMPORT
#endif

#include <memory>

#include <madrona/py/utils.hpp>
#include <madrona/exec_mode.hpp>

#include <madrona/render/render_mgr.hpp>

namespace madMJX {

struct VisualizerGPUHandles {
    madrona::render::APIBackend *renderAPI;
    madrona::render::GPUDevice *renderDev;
};

struct MJXModelGeometry {
    madrona::math::Vector3 *vertices;
    uint32_t *indices;
    uint32_t *vertexOffsets;
    uint32_t *triOffsets;
    uint32_t numVertices;
    uint32_t numTris;
    uint32_t numMeshes;
};

struct MJXModel {
    MJXModelGeometry meshGeo;
    int32_t *geomTypes;
    int32_t *geomDataIDs;
    madrona::math::Vector3 *geomSizes;
    uint32_t numGeoms;
    uint32_t numCams;
};

// The Manager class encapsulates the linkage between the outside training
// code and the internal simulation state (src/sim.hpp / src/sim.cpp)
//
// Manager is responsible for initializing the simulator, loading physics
// and rendering assets off disk, and mapping ECS components to tensors
// for learning
class Manager {
public:
    struct Config {
        madrona::ExecMode execMode; // CPU or CUDA
        int gpuID; // Which GPU for CUDA backend?
        uint32_t numWorlds; // Simulation batch size
        uint32_t batchRenderViewWidth;
        uint32_t batchRenderViewHeight;
        bool addCamDebugGeometry = false;
    };

    MGR_EXPORT Manager(
        const Config &cfg,
        const MJXModel &mjx_model,
        madrona::Optional<VisualizerGPUHandles> viz_gpu_hdls =
            madrona::Optional<VisualizerGPUHandles>::none());
    MGR_EXPORT ~Manager();

    MGR_EXPORT void init(madrona::math::Vector3 *geom_pos,
                         madrona::math::Quat *geom_rot,
                         madrona::math::Vector3 *cam_pos,
                         madrona::math::Quat *cam_rot);
    MGR_EXPORT void render(madrona::math::Vector3 *geom_pos,
                           madrona::math::Quat *geom_rot,
                           madrona::math::Vector3 *cam_pos,
                           madrona::math::Quat *cam_rot);

#ifdef MADRONA_CUDA_SUPPORT
    MGR_EXPORT void gpuStreamInit(cudaStream_t strm, void **buffers);
    MGR_EXPORT void gpuStreamRender(cudaStream_t strm, void **buffers);
#endif

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    //
    MGR_EXPORT madrona::py::Tensor instancePositionsTensor() const;
    MGR_EXPORT madrona::py::Tensor instanceRotationsTensor() const;
    MGR_EXPORT madrona::py::Tensor cameraPositionsTensor() const;
    MGR_EXPORT madrona::py::Tensor cameraRotationsTensor() const;

    MGR_EXPORT madrona::py::Tensor rgbTensor() const;
    MGR_EXPORT madrona::py::Tensor depthTensor() const;

    MGR_EXPORT uint32_t numWorlds() const;
    MGR_EXPORT madrona::render::RenderManager & getRenderManager();

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
