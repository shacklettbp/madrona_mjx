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
    if (viz_gpu_hdls.has_value()) {
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

static inline render::RenderManager initRenderManager(
    const Manager::Config &mgr_cfg,
    const MJXModel &mjx_model,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls,
    const Optional<RenderGPUState> &render_gpu_state)
{
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

    uint32_t max_instances_per_world = mjx_model.numGeoms;
    if (mgr_cfg.addCamDebugGeometry) {
        max_instances_per_world += mjx_model.numCams;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = true,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = mjx_model.numCams,
        .maxInstancesPerWorld = max_instances_per_world,
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

struct Manager::Impl {
    Config cfg;
    uint32_t numGeoms;
    uint32_t numCams;
    Optional<RenderGPUState> renderGPUState;
    render::RenderManager renderMgr;

    inline Impl(const Manager::Config &mgr_cfg,
                uint32_t num_geoms,
                uint32_t num_cams,
                Optional<RenderGPUState> &&render_gpu_state,
                render::RenderManager &&render_mgr)
        : cfg(mgr_cfg),
          numGeoms(num_geoms),
          numCams(num_cams),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr))
    {}

    inline virtual ~Impl() {}

    virtual void init(Vector3 *geom_positions, Quat *geom_rotations,
                      Vector3 *cam_positions, Quat *cam_rotations) = 0;
    virtual void render(Vector3 *geom_positions, Quat *geom_rotations,
                        Vector3 *cam_positions, Quat *cam_rotations) = 0;

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuStreamInit(cudaStream_t strm, void **buffers) = 0;
    virtual void gpuStreamRender(cudaStream_t strm, void **buffers) = 0;
#endif

    inline void renderCommon()
    {
        renderMgr.readECS();
        renderMgr.batchRender();
    }

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * make(
        const Config &cfg,
        const MJXModel &mjx_model,
        const Optional<VisualizerGPUHandles> &viz_gpu_hdls);
};

struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   uint32_t num_geoms,
                   uint32_t num_cams,
                   Optional<RenderGPUState> &&render_gpu_state,
                   render::RenderManager &&render_mgr,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg, num_geoms, num_cams,
               std::move(render_gpu_state), std::move(render_mgr)),
          cpuExec(std::move(cpu_exec))
    {}

    inline virtual ~CPUImpl() final {}

    inline void copyInTransforms(Vector3 *geom_positions,
                                 Quat *geom_rotations,
                                 Vector3 *cam_positions,
                                 Quat *cam_rotations)
    {
        memcpy(cpuExec.getExported((CountT)ExportID::InstancePositions),
               geom_positions,
               sizeof(Vector3) * numGeoms * cfg.numWorlds);
        memcpy(cpuExec.getExported((CountT)ExportID::InstanceRotations),
               geom_rotations,
               sizeof(Quat) * numCams * cfg.numWorlds);

        memcpy(cpuExec.getExported((CountT)ExportID::CameraPositions),
               cam_positions,
               sizeof(Vector3) * numGeoms * cfg.numWorlds);
        memcpy(cpuExec.getExported((CountT)ExportID::CameraRotations),
               cam_rotations,
               sizeof(Quat) * numCams * cfg.numWorlds);
    }

    inline virtual void init(Vector3 *geom_positions,
                             Quat *geom_rotations,
                             Vector3 *cam_positions,
                             Quat *cam_rotations) final
    {
        copyInTransforms(geom_positions, geom_rotations,
                         cam_positions, cam_rotations);
        cpuExec.runTaskGraph(TaskGraphID::Init);
        renderCommon();
    }

    inline virtual void render(Vector3 *geom_positions,
                               Quat *geom_rotations,
                               Vector3 *cam_positions,
                               Quat *cam_rotations) final
    {
        copyInTransforms(geom_positions, geom_rotations,
                         cam_positions, cam_rotations);
        cpuExec.runTaskGraph(TaskGraphID::Render);
        renderCommon();
    }

#ifdef MADRONA_CUDA_SUPPORT
    virtual void gpuStreamInit(cudaStream_t, void **) final
    {
        FATAL("madMJX TODO: CPU backend integration");
    }

    virtual void gpuStreamRender(cudaStream_t, void **) final
    {
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
    MWCudaLaunchGraph renderGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                    uint32_t num_geoms,
                    uint32_t num_cams,
                    Optional<RenderGPUState> &&render_gpu_state,
                    render::RenderManager &&render_mgr,
                    MWCudaExecutor &&gpu_exec)
        : Impl(mgr_cfg, num_geoms, num_cams,
               std::move(render_gpu_state), std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          renderGraph(gpuExec.buildLaunchGraph(TaskGraphID::Render))
    {}

    inline virtual ~CUDAImpl() final {}

    inline void copyInTransforms(Vector3 *geom_positions,
                                 Quat *geom_rotations,
                                 Vector3 *cam_positions,
                                 Quat *cam_rotations,
                                 cudaStream_t strm)
    {
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstancePositions),
            geom_positions,
            sizeof(Vector3) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::InstanceRotations),
            geom_rotations,
            sizeof(Quat) * numGeoms * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);

        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::CameraPositions),
            cam_positions,
            sizeof(Vector3) * numCams * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
        cudaMemcpyAsync(
            gpuExec.getExported((CountT)ExportID::CameraRotations),
            cam_rotations,
            sizeof(Quat) * numCams * cfg.numWorlds,
            cudaMemcpyDeviceToDevice, strm);
    }

    inline virtual void init(Vector3 *geom_positions,
                             Quat *geom_rotations,
                             Vector3 *cam_positions,
                             Quat *cam_rotations) final
    {
        MWCudaLaunchGraph init_graph =
            gpuExec.buildLaunchGraph(TaskGraphID::Init);

        copyInTransforms(geom_positions, geom_rotations,
                         cam_positions, cam_rotations, 0);
        gpuExec.run(init_graph);
        renderCommon();
    }

    inline virtual void render(Vector3 *geom_positions,
                               Quat *geom_rotations,
                               Vector3 *cam_positions,
                               Quat *cam_rotations) final
    {
        copyInTransforms(geom_positions, geom_rotations,
                         cam_positions, cam_rotations, 0);

        gpuExec.run(renderGraph);
        renderCommon();
    }

    struct JAXIO {
        Vector3 *geomPositions;
        Quat *geomRotations;
        Vector3 *camPositions;
        Quat *camRotations;

        uint8_t *rgbOut;
        float *depthOut;

        static inline JAXIO make(void **buffers)
        {
            CountT buf_idx = 0;
            auto geom_positions = (Vector3 *)buffers[buf_idx++];
            auto geom_rotations = (Quat *)buffers[buf_idx++];
            auto cam_positions = (Vector3 *)buffers[buf_idx++];
            auto cam_rotations = (Quat *)buffers[buf_idx++];
            auto rgb_out = (uint8_t *)buffers[buf_idx++];
            auto depth_out = (float *)buffers[buf_idx++];

            return JAXIO {
                .geomPositions = geom_positions,
                .geomRotations = geom_rotations,
                .camPositions = cam_positions,
                .camRotations = cam_rotations,
                .rgbOut = rgb_out,
                .depthOut = depth_out,
            };
        }
    };

    inline void copyOutRendered(uint8_t *rgb_out, float *depth_out,
                                cudaStream_t strm)
    {
        cudaMemcpyAsync(rgb_out, renderMgr.batchRendererRGBOut(),
                        sizeof(uint8_t) * 4 *
                        (size_t)cfg.batchRenderViewWidth *
                        (size_t)cfg.batchRenderViewHeight *
                        (size_t)cfg.numWorlds *
                        (size_t)numCams,
                        cudaMemcpyDeviceToDevice, strm);

        cudaMemcpyAsync(depth_out, renderMgr.batchRendererDepthOut(),
                        sizeof(float) *
                        (size_t)cfg.batchRenderViewWidth *
                        (size_t)cfg.batchRenderViewHeight *
                        (size_t)cfg.numWorlds *
                        (size_t)numCams,
                        cudaMemcpyDeviceToDevice, strm);
    }

    virtual void gpuStreamInit(cudaStream_t strm, void **buffers) final
    {
        MWCudaLaunchGraph init_graph =
            gpuExec.buildLaunchGraph(TaskGraphID::Init);

        JAXIO jax_io = JAXIO::make(buffers);

        copyInTransforms(jax_io.geomPositions, jax_io.geomRotations,
                         jax_io.camPositions, jax_io.camRotations, strm);

        gpuExec.runAsync(init_graph, strm);
        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        REQ_CUDA(cudaStreamSynchronize(strm));
        renderCommon();

        copyOutRendered(jax_io.rgbOut, jax_io.depthOut, strm);
    }

    virtual void gpuStreamRender(cudaStream_t strm, void **buffers) final
    {
        JAXIO jax_io = JAXIO::make(buffers);

        copyInTransforms(jax_io.geomPositions, jax_io.geomRotations,
                         jax_io.camPositions, jax_io.camRotations, strm);

#if 0
        Vector3 *readback_pos = (Vector3 *)malloc(sizeof(Vector3) * numGeoms * cfg.numWorlds);
        cudaMemcpy(jax_io.geomPositions,
                   readback_pos,
                   sizeof(Vector3) * numGeoms * cfg.numWorlds,
                   cudaMemcpyHostToDevice);
        printf("%f %f %f\n",
            readback_pos[1].x,
            readback_pos[1].y,
            readback_pos[1].z);
#endif

        gpuExec.runAsync(renderGraph, strm);
        // Currently a CPU sync is needed to read back the total number of
        // instances for Vulkan
        REQ_CUDA(cudaStreamSynchronize(strm));
        renderCommon();

        copyOutRendered(jax_io.rgbOut, jax_io.depthOut, strm);
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

static void loadRenderObjects(
    const MJXModelGeometry &geo,
    render::RenderManager &render_mgr)
{
    using namespace imp;

    std::array<std::string, (size_t)RenderPrimObjectIDs::NumPrims> 
        render_asset_paths;
    render_asset_paths[(size_t)RenderPrimObjectIDs::DebugCam] =
        (std::filesystem::path(DATA_DIR) / "debugcam.obj").string();
    render_asset_paths[(size_t)RenderPrimObjectIDs::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)RenderPrimObjectIDs::Sphere] =
        (std::filesystem::path(DATA_DIR) / "sphere.obj").string();
    render_asset_paths[(size_t)RenderPrimObjectIDs::Box] =
        (std::filesystem::path(DATA_DIR) / "box.obj").string();

    std::array<const char *, render_asset_paths.size()> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    std::array<char, 1024> import_err;
    auto disk_render_assets = imp::ImportedAssets::importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    if (!disk_render_assets.has_value()) {
        FATAL("Failed to load render assets from disk: %s", import_err);
    }


    HeapArray<SourceMesh> meshes(geo.numMeshes);

    const CountT num_disk_objs = disk_render_assets->objects.size();
    HeapArray<SourceObject> objs(num_disk_objs + geo.numMeshes);

    for (CountT i = 0; i < num_disk_objs; i++) {
        objs[i] = disk_render_assets->objects[i];
        for (auto &mesh : objs[i].meshes) {
            mesh.materialIDX = 0;
        }
    }

    // Color axes
    disk_render_assets->objects[0].meshes[0].materialIDX = 1;
    disk_render_assets->objects[0].meshes[1].materialIDX = 2;
    disk_render_assets->objects[0].meshes[2].materialIDX = 3;

    const CountT num_meshes = (CountT)geo.numMeshes;
    for (CountT mesh_idx = 0; mesh_idx < num_meshes; mesh_idx++) {
        uint32_t mesh_vert_offset = geo.vertexOffsets[mesh_idx];
        uint32_t next_vert_offset = mesh_idx < num_meshes - 1 ?
            geo.vertexOffsets[mesh_idx + 1] : geo.numVertices;

        uint32_t mesh_tri_offset = geo.triOffsets[mesh_idx];
        uint32_t next_tri_offset = mesh_idx < num_meshes - 1 ?
            geo.triOffsets[mesh_idx + 1] : geo.numTris;

        uint32_t mesh_num_verts = next_vert_offset - mesh_vert_offset;
        uint32_t mesh_num_tris = next_tri_offset - mesh_tri_offset;
        uint32_t mesh_idx_offset = mesh_tri_offset * 3;

        meshes[mesh_idx] = {
            .positions = geo.vertices + mesh_vert_offset,
            .normals = nullptr,
            .tangentAndSigns = nullptr,
            .uvs = nullptr,
            .indices = geo.indices + mesh_idx_offset,
            .faceCounts = nullptr,
            .faceMaterials = nullptr,
            .numVertices = mesh_num_verts,
            .numFaces = mesh_num_tris,
            .materialIDX = 0,
        };

        objs[num_disk_objs + mesh_idx] = {
            .meshes = Span<SourceMesh>(&meshes[mesh_idx], 1),
        };
    }

    auto materials = std::to_array<imp::SourceMaterial>({
        { render::rgb8ToFloat(255, 255, 255), -1, 0.8f, 0.2f },
        { render::rgb8ToFloat(50, 50, 255), -1, 0.8f, 0.2f },
        { render::rgb8ToFloat(255, 50, 50), -1, 0.8f, 0.2f },
        { render::rgb8ToFloat(50, 255, 50), -1, 0.8f, 0.2f },
    });

    render_mgr.loadObjects(objs, materials, {});

    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

Manager::Impl * Manager::Impl::make(
    const Manager::Config &mgr_cfg,
    const MJXModel &mjx_model,
    const Optional<VisualizerGPUHandles> &viz_gpu_hdls)
{
    Sim::Config sim_cfg;
    sim_cfg.numGeoms = mjx_model.numGeoms;
    sim_cfg.numCams = mjx_model.numCams;
    sim_cfg.useDebugCamEntity = mgr_cfg.addCamDebugGeometry;

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg, viz_gpu_hdls);

        render::RenderManager render_mgr =
            initRenderManager(mgr_cfg, mjx_model,
                              viz_gpu_hdls, render_gpu_state);

        loadRenderObjects(mjx_model.meshGeo, render_mgr);
        sim_cfg.renderBridge = render_mgr.bridge();

        int32_t *geom_types_gpu = (int32_t *)cu::allocGPU(
            sizeof(int32_t) * mjx_model.numGeoms);
        int32_t *geom_data_ids_gpu = (int32_t *)cu::allocGPU(
            sizeof(int32_t) * mjx_model.numGeoms);
        Vector3 *geom_sizes_gpu = (Vector3 *)cu::allocGPU(
            sizeof(Vector3) * mjx_model.numGeoms);

        REQ_CUDA(cudaMemcpy(geom_types_gpu, mjx_model.geomTypes,
            sizeof(int32_t) * mjx_model.numGeoms, cudaMemcpyHostToDevice));
        REQ_CUDA(cudaMemcpy(geom_data_ids_gpu, mjx_model.geomDataIDs,
            sizeof(int32_t) * mjx_model.numGeoms, cudaMemcpyHostToDevice));
        REQ_CUDA(cudaMemcpy(geom_sizes_gpu, mjx_model.geomSizes,
            sizeof(Vector3) * mjx_model.numGeoms, cudaMemcpyHostToDevice));

        sim_cfg.geomTypes = geom_types_gpu;
        sim_cfg.geomDataIDs = geom_data_ids_gpu;
        sim_cfg.geomSizes = geom_sizes_gpu;

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

        cu::deallocGPU(geom_types_gpu);
        cu::deallocGPU(geom_data_ids_gpu);
        cu::deallocGPU(geom_sizes_gpu);

        return new CUDAImpl {
            mgr_cfg,
            mjx_model.numGeoms,
            mjx_model.numCams,
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

        render::RenderManager render_mgr =
            initRenderManager(mgr_cfg, mjx_model,
                              viz_gpu_hdls, render_gpu_state);

        loadRenderObjects(mjx_model.meshGeo, render_mgr);
        sim_cfg.renderBridge = render_mgr.bridge();

        sim_cfg.geomTypes = mjx_model.geomTypes;
        sim_cfg.geomDataIDs = mjx_model.geomDataIDs;
        sim_cfg.geomSizes = mjx_model.geomSizes;

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

        auto cpu_impl = new CPUImpl {
            mgr_cfg,
            mjx_model.numGeoms,
            mjx_model.numCams,
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
                 const MJXModel &mjx_model,
                 Optional<VisualizerGPUHandles> viz_gpu_hdls)
    : impl_(Impl::make(cfg, mjx_model, viz_gpu_hdls))
{}

Manager::~Manager() {}

void Manager::init(math::Vector3 *geom_pos, math::Quat *geom_rot,
                   math::Vector3 *cam_pos, math::Quat *cam_rot)
{
    impl_->init(geom_pos, geom_rot, cam_pos, cam_rot);
}

void Manager::render(math::Vector3 *geom_pos, math::Quat *geom_rot,
                     math::Vector3 *cam_pos, math::Quat *cam_rot)
{
    impl_->render(geom_pos, geom_rot, cam_pos, cam_rot);
}

#ifdef MADRONA_CUDA_SUPPORT
void Manager::gpuStreamInit(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamInit(strm, buffers);
}

void Manager::gpuStreamRender(cudaStream_t strm, void **buffers)
{
    impl_->gpuStreamRender(strm, buffers);
}
#endif

Tensor Manager::instancePositionsTensor() const
{
    return impl_->exportTensor(ExportID::InstancePositions,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numGeoms,
                                   sizeof(Vector3) / sizeof(float),
                               });
}

Tensor Manager::instanceRotationsTensor() const
{
    return impl_->exportTensor(ExportID::InstanceRotations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numGeoms,
                                   sizeof(Quat) / sizeof(float),
                               });
}

Tensor Manager::cameraPositionsTensor() const
{
    return impl_->exportTensor(ExportID::CameraPositions,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numCams,
                                   sizeof(Vector3) / sizeof(float),
                               });
}

Tensor Manager::cameraRotationsTensor() const
{
    return impl_->exportTensor(ExportID::CameraRotations,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   impl_->numCams,
                                   sizeof(Quat) / sizeof(float),
                               });
}

Tensor Manager::rgbTensor() const
{
    const uint8_t *rgb_ptr = impl_->renderMgr.batchRendererRGBOut();

    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr.batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        impl_->numCams,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

uint32_t Manager::numWorlds() const
{
    return impl_->cfg.numWorlds;
}

render::RenderManager & Manager::getRenderManager()
{
    return impl_->renderMgr;
}

}
