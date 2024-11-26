#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace madMJX {

// Register all the ECS components and archetypes that will be
// used in the simulation
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);
    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    registry.registerArchetype<RenderEntity>();

    registry.exportColumn<RenderEntity, Position>(
        (uint32_t)ExportID::InstancePositions);
    registry.exportColumn<RenderEntity, Rotation>(
        (uint32_t)ExportID::InstanceRotations);
    registry.exportColumn<RenderEntity, Scale>(
        (uint32_t)ExportID::InstanceScales);
    registry.exportColumn<RenderEntity, MaterialOverride>( 
        (uint32_t)ExportID::InstanceMatOverrides);
    registry.exportColumn<RenderEntity, ColorOverride>( 
        (uint32_t)ExportID::InstanceColorOverrides);
    
    if (cfg.useDebugCamEntity) {
        registry.registerArchetype<DebugCameraEntity>();

        registry.exportColumn<DebugCameraEntity, Position>(
            (uint32_t)ExportID::CameraPositions);
        registry.exportColumn<DebugCameraEntity, Rotation>(
            (uint32_t)ExportID::CameraRotations);
    } else {
        registry.registerArchetype<CameraEntity>();

        registry.exportColumn<CameraEntity, Position>(
            (uint32_t)ExportID::CameraPositions);
        registry.exportColumn<CameraEntity, Rotation>(
            (uint32_t)ExportID::CameraRotations);
    }

    if (cfg.useRT) {
        registry.exportColumn<render::RaycastOutputArchetype, 
            render::DepthOutputBuffer>((uint32_t)ExportID::RaycastDepth);
        registry.exportColumn<render::RaycastOutputArchetype,
            render::RGBOutputBuffer>((uint32_t)ExportID::RaycastRGB);
    }
}

static void setupRenderTasks(TaskGraphBuilder &builder,
                             Span<const TaskGraphNodeID> deps,
                             bool update_mats = false)
{
#if 0
    builder.addToGraph<ParallelForNode<
        Engine, printTransforms, Position, Rotation, Scale, ObjectID>>(deps);
#endif
    RenderingSystem::setupTasks(builder, deps, update_mats);
}

#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraphNodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

static void setupInitTasks(TaskGraphBuilder &builder,
                           const Sim::Config &cfg)
{
#ifdef MADRONA_GPU_MODE
    TaskGraphNodeID sort_sys;

    if (cfg.useDebugCamEntity) {
        sort_sys = queueSortByWorld<DebugCameraEntity>(builder, {});
    } else {
        sort_sys = queueSortByWorld<CameraEntity>(builder, {});
    }

    sort_sys = queueSortByWorld<RenderEntity>(builder, {sort_sys});
#else
    (void)builder;
    (void)cfg;
#endif
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    TaskGraphBuilder &init_builder = taskgraph_mgr.init(TaskGraphID::Init);
    setupInitTasks(init_builder, cfg);

    TaskGraphBuilder &render_init_builder =
        taskgraph_mgr.init(TaskGraphID::RenderInit);
    setupRenderTasks(render_init_builder, {}, true);

    TaskGraphBuilder &render_tasks_builder =
        taskgraph_mgr.init(TaskGraphID::Render);
    setupRenderTasks(render_tasks_builder, {}, false);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    RenderingSystem::init(ctx, cfg.renderBridge);

    for (CountT geom_idx = 0; geom_idx < (CountT)cfg.numGeoms; geom_idx++) {

        Entity instance;
        if (cfg.geomDataIDs[geom_idx] == -1) {
            instance = ctx.makeEntity<RenderEntity>();
            render::RenderingSystem::disableEntityRenderable(ctx, instance);
            ctx.get<ObjectID>(instance) = ObjectID {0};
        }
        else {
            instance = ctx.makeRenderableEntity<RenderEntity>();
            ctx.get<ObjectID>(instance) = ObjectID { cfg.geomDataIDs[geom_idx] };
        }

        ctx.get<Position>(instance) = Vector3::zero();
        ctx.get<Rotation>(instance) = Quat { 1, 0, 0, 0 };
        ctx.get<Scale>(instance) = Diag3x3 { 
            cfg.geomSizes[geom_idx].x,
            cfg.geomSizes[geom_idx].y,
            cfg.geomSizes[geom_idx].z };
        ctx.get<MaterialOverride>(instance) = MaterialOverride { -1 };
        ctx.get<ColorOverride>(instance) = ColorOverride { 0 };
    }

    for (CountT cam_idx = 0; cam_idx < (CountT)cfg.numCams; cam_idx++) {
        Entity cam;
        if (cfg.useDebugCamEntity) {
            cam = ctx.makeRenderableEntity<DebugCameraEntity>();

            ctx.get<Scale>(cam) = Diag3x3 { 0.1, 0.1, 0.1 };
            ctx.get<ObjectID>(cam).idx = cfg.numGeoms;
        } else {
            cam = ctx.makeEntity<CameraEntity>();
        }
        ctx.get<Position>(cam) = Vector3::zero();
        ctx.get<Rotation>(cam) = Quat { 1, 0, 0, 0 };
        render::RenderingSystem::attachEntityToView(
            ctx, cam, 60.f, 0.001f, Vector3::zero());
    }

    Entity light = render::RenderingSystem::makeLight(ctx);
    render::RenderingSystem::configureLight(ctx, light, render::LightDesc {
        .type = render::LightDesc::Type::Directional,
        .castShadow = false,
        .position = Vector3::zero(),
        .direction = Vector3 { 0, 0, -1 },
        .cutoff = 0.78539816339f,
        .active = true
    });
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
