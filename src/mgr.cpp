#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;

namespace madMJX {

struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};


static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    if (viz_gpu_hdls.has_value() || !mgr_cfg.enableBatchRenderer) {
        return Optional<RenderGPUState>::none();
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (!viz_gpu_hdls.has_value() && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        assert(viz_gpu_hdls.has_value());

        render_api = viz_gpu_hdls->renderAPI;
        render_dev = viz_gpu_hdls->renderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = 1,
        .maxInstancesPerWorld = 10,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    WorldReset *worldResetBuffer;
    Action *agentActionsBuffer;
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;

    inline Impl(const Manager::Config &mgr_cfg,
                WorldReset *reset_buffer,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr)
        : cfg(mgr_cfg),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr))
    {}

    inline virtual ~Impl() {}

    virtual void init() = 0;
    virtual void processActions() = 0;
    virtual void postPhysics() = 0;

#ifdef MADRONA_CUDA_SUPPORT
    virtual void processActionsAsync(cudaStream_t strm) = 0;
    virtual void postPhysicsAsync(cudaStream_t strm) = 0;
#endif

    inline void renderStep()
    {
        if (renderMgr.has_value()) {
            renderMgr->readECS();
        }

        if (cfg.enableBatchRenderer) {
            renderMgr->batchRender();
        }
    }

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * make(
        const Config &cfg,
        const Optional<VisualizerGPUHandles> &viz_gpu_hdls);
};

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg,
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr)),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final {}

    inline virtual void init() final
    {
        cpuExec.runTaskGraph(TaskGraphID::Init);
        renderStep();
    }

    inline virtual void processActions() final
    {
        cpuExec.runTaskGraph(TaskGraphID::ProcessActions);
    }

    inline virtual void postPhysics() final
    {
        cpuExec.runTaskGraph(TaskGraphID::PostPhysics);
        renderStep();
    }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void processActionsAsync(cudaStream_t strm) final
    {
        (void)strm;
        FATAL("madMJX TODO: CPU backend integration");
    }

    virtual void postPhysicsAsync(cudaStream_t strm) final
    {
        (void)strm;
        FATAL("madMJX TODO: CPU backend integration");
    }
#endif

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph initGraph;
    MWCudaLaunchGraph processActionsGraph;
    MWCudaLaunchGraph postPhysicsGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec)
        : Impl(mgr_cfg,
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          initGraph(gpuExec.buildLaunchGraph(TaskGraphID::Init)),
          processActionsGraph(
              gpuExec.buildLaunchGraph(TaskGraphID::ProcessActions)),
          postPhysicsGraph(
              gpuExec.buildLaunchGraph(TaskGraphID::PostPhysics))
    {}

    inline virtual ~CUDAImpl() final {}

    inline virtual void init() final
    {
        gpuExec.run(initGraph);
    }

    inline virtual void processActions() final
    {
        gpuExec.run(processActionsGraph);
    }

    inline virtual void postPhysics() final
    {
        gpuExec.run(postPhysicsGraph);
        renderStep();
    }

    inline virtual void processActionsAsync(cudaStream_t strm) final
    {
        gpuExec.runAsync(processActionsGraph, strm);
    }

    inline virtual void postPhysicsAsync(cudaStream_t strm) final
    {
        gpuExec.runAsync(postPhysicsGraph, strm);
        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        REQ_CUDA(cudaStreamSynchronize(strm));
        renderStep();
    }

    virtual inline Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

static void loadRenderObjects(render::RenderManager &render_mgr)
{
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    render_asset_paths[(size_t)SimObject::Pole] =
        (std::filesystem::path(DATA_DIR) / "pole_render.obj").string();
    render_asset_paths[(size_t)SimObject::Cart] =
        (std::filesystem::path(DATA_DIR) / "cart_render.obj").string();
    render_asset_paths[(size_t)SimObject::Backdrop] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();

    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },
        { render::rgb8ToFloat(230, 230, 20), -1, 0.8f, 1.0f },
        { render::rgb8ToFloat(180, 180, 180), -1, 0.8f, 1.0f },
    });

    // Override materials
    render_assets->objects[(CountT)SimObject::Pole].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::Cart].meshes[0].materialIDX = 1;
    render_assets->objects[(CountT)SimObject::Backdrop].meshes[0].materialIDX = 2;

    render_mgr.loadObjects(render_assets->objects, materials, {});

    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

Manager::Impl * Manager::Impl::make(
    const Manager::Config &mgr_cfg,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    Sim::Config sim_cfg;
    sim_cfg.maxStepsPerEpisode = mgr_cfg.maxEpisodeLength;

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg, viz_gpu_hdls);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, viz_gpu_hdls, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = (uint32_t)TaskGraphID::NumGraphs,
            .numExportedBuffers = (uint32_t)ExportID::NumExports, 
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        WorldReset *world_reset_buffer =  (WorldReset *)gpu_exec.getExported(
            (uint32_t)ExportID::EpisodeReset);

        Action *agent_actions_buffer = (Action *)gpu_exec.getExported(
            (uint32_t)ExportID::AgentAction);

        return new CUDAImpl {
            mgr_cfg,
            world_reset_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg, viz_gpu_hdls);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, viz_gpu_hdls, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);

        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
            (uint32_t)TaskGraphID::NumGraphs,
        };

        WorldReset *world_reset_buffer = (WorldReset *)cpu_exec.getExported(
                (uint32_t)ExportID::EpisodeReset);

        Action *agent_actions_buffer = (Action *)cpu_exec.getExported(
            (uint32_t)ExportID::AgentAction);

        auto cpu_impl = new CPUImpl {
            mgr_cfg,
            world_reset_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(cpu_exec),
        };

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

Manager::Manager(const Config &cfg,
                 Optional<VisualizerGPUHandles> viz_gpu_hdls)
    : impl_(Impl::make(cfg, viz_gpu_hdls))
{}

Manager::~Manager() {}

void Manager::init()
{
    impl_->init();
}

void Manager::processActions()
{
    impl_->processActions();
}

void Manager::postPhysics()
{
    impl_->postPhysics();
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::processActionsAsync(cudaStream_t strm)
{
    impl_->processActionsAsync(strm);
}

void Manager::postPhysicsAsync(cudaStream_t strm)
{
    impl_->postPhysicsAsync(strm);
}
#endif

Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::EpisodeReset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::EpisodeDone,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}


Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::AgentAction,
                               TensorElementType::Int32,
        {
            impl_->cfg.numWorlds,
            1,
        });
}

Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::AgentReward,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

Tensor Manager::rigidBodyPositionsTensor() const
{
    return impl_->exportTensor(ExportID::RigidBodyPositions,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   2,
                                   sizeof(Vector3) / sizeof(float),
                               });
}

Tensor Manager::rigidBodyRotationsTensor() const
{
    return impl_->exportTensor(ExportID::RigidBodyRotations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   2,
                                   sizeof(Quat) / sizeof(float),
                               });
}

Tensor Manager::jointForcesTensor() const
{
    return impl_->exportTensor(ExportID::JointForces,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                                   sizeof(JointForce) / sizeof(float),
                               });
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}

void Manager::setAction(int32_t world_idx,
                        int32_t move_amount)
{
    Action action { 
        .move = move_amount,
    };

    auto *action_ptr = impl_->agentActionsBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

uint32_t Manager::numWorlds() const
{
    return impl_->cfg.numWorlds;
}

render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}
