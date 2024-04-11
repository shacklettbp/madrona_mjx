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
    registry.registerArchetype<CameraEntity>();

    registry.exportColumn<RenderEntity, Position>(
        (uint32_t)ExportID::InstancePositions);
    registry.exportColumn<RenderEntity, Rotation>(
        (uint32_t)ExportID::InstanceRotations);

    registry.exportColumn<CameraEntity, Position>(
        (uint32_t)ExportID::CameraPositions);
    registry.exportColumn<CameraEntity, Rotation>(
        (uint32_t)ExportID::CameraRotations);
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

static void setupInitTasks(TaskGraphBuilder &builder)
{
#ifdef MADRONA_GPU_MODE
    auto sort_sys = queueSortByWorld<RenderEntity>(builder, {});
    sort_sys = queueSortByWorld<CameraEntity>(builder, {sort_sys});
#else
    (void)builder;
#endif
}

static void setupRenderTasks(TaskGraphBuilder &builder)
{
    RenderingSystem::setupTasks(builder, {});
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &)
{
    TaskGraphBuilder &init_builder = taskgraph_mgr.init(TaskGraphID::Init);
    setupInitTasks(init_builder);

    TaskGraphBuilder &render_tasks_builder =
        taskgraph_mgr.init(TaskGraphID::Render);
    setupRenderTasks(render_tasks_builder);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    RenderingSystem::init(ctx, cfg.renderBridge);

    Entity cam = ctx.makeEntity<CameraEntity>();
    ctx.get<Position>(cam) = Vector3 { 0, -3, 0 };
    ctx.get<Rotation>(cam) = Quat { 1, 0, 0, 0 };
    render::RenderingSystem::attachEntityToView(
        ctx, cam, 60.f, 0.001f, Vector3::zero());

    for (CountT mesh_idx = 0; mesh_idx < (CountT)cfg.numMeshes; mesh_idx++) {
        Entity instance = ctx.makeRenderableEntity<RenderEntity>();
        ctx.get<Position>(instance) = Vector3::zero();
        ctx.get<Rotation>(instance) = Quat { 1, 0, 0, 0 };
        ctx.get<Scale>(instance) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(instance) = ObjectID { (int32_t)mesh_idx };
    }
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
