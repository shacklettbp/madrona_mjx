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
        uint32_t maxEpisodeLength;
        bool enableBatchRenderer;
        uint32_t batchRenderViewWidth = 64;
        uint32_t batchRenderViewHeight = 64;
    };

    MGR_EXPORT Manager(
        const Config &cfg,
        madrona::Optional<VisualizerGPUHandles> viz_gpu_hdls =
            madrona::Optional<VisualizerGPUHandles>::none());
    MGR_EXPORT ~Manager();

    MGR_EXPORT void init();
    MGR_EXPORT void processActions();
    MGR_EXPORT void postPhysics();

#ifdef MADRONA_CUDA_SUPPORT
    MGR_EXPORT void processActionsAsync(cudaStream_t strm);
    MGR_EXPORT void postPhysicsAsync(cudaStream_t strm);
#endif

    // These functions export Tensor objects that link the ECS
    // simulation state to the python bindings / PyTorch tensors (src/bindings.cpp)
    MGR_EXPORT madrona::py::Tensor resetTensor() const;
    MGR_EXPORT madrona::py::Tensor doneTensor() const;

    MGR_EXPORT madrona::py::Tensor actionTensor() const;
    MGR_EXPORT madrona::py::Tensor rewardTensor() const;

    MGR_EXPORT madrona::py::Tensor rigidBodyPositionsTensor() const;
    MGR_EXPORT madrona::py::Tensor rigidBodyRotationsTensor() const;
    MGR_EXPORT madrona::py::Tensor jointForcesTensor() const;

    MGR_EXPORT madrona::py::Tensor rgbTensor() const;
    MGR_EXPORT madrona::py::Tensor depthTensor() const;

    // These functions are used by the viewer to control the simulation
    // with keyboard inputs in place of DNN policy actions
    MGR_EXPORT void triggerReset(int32_t world_idx);
    MGR_EXPORT void setAction(int32_t world_idx, int32_t move_amount);

    MGR_EXPORT uint32_t numWorlds() const;
    MGR_EXPORT madrona::render::RenderManager & getRenderManager();

private:
    struct Impl;
    struct CPUImpl;
    struct CUDAImpl;

    std::unique_ptr<Impl> impl_;
};

}
