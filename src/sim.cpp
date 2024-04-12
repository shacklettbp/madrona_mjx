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

static void setupRenderTasks(TaskGraphBuilder &builder,
                             Span<const TaskGraphNodeID> deps)
{
    RenderingSystem::setupTasks(builder, deps);
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
    auto sort_sys = queueSortByWorld<CameraEntity>(builder, {});
    sort_sys = queueSortByWorld<RenderEntity>(builder, {sort_sys});

    setupRenderTasks(builder, {sort_sys});
#else
    setupRenderTasks(builder, {});
#endif
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &)
{
    TaskGraphBuilder &init_builder = taskgraph_mgr.init(TaskGraphID::Init);
    setupInitTasks(init_builder);

    TaskGraphBuilder &render_tasks_builder =
        taskgraph_mgr.init(TaskGraphID::Render);
    setupRenderTasks(render_tasks_builder, {});
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    RenderingSystem::init(ctx, cfg.renderBridge);

    for (CountT geom_idx = 0; geom_idx < (CountT)cfg.numGeoms; geom_idx++) {
        Entity instance = ctx.makeRenderableEntity<RenderEntity>();
        ctx.get<Position>(instance) = Vector3::zero();
        ctx.get<Rotation>(instance) = Quat { 1, 0, 0, 0 };

        Diag3x3 scale;
        int32_t render_obj_idx;
        switch ((MJXGeomType)cfg.geomTypes[geom_idx]) {
        case MJXGeomType::Plane: {
            // FIXME
            float plane_scale = cfg.geomSizes[geom_idx].z;
            scale.d0 = plane_scale;
            scale.d1 = plane_scale;
            scale.d2 = plane_scale;
            render_obj_idx = 0;
        } break;
        case MJXGeomType::Sphere: {
            float sphere_scale = cfg.geomSizes[geom_idx].x;
            scale.d0 = sphere_scale;
            scale.d1 = sphere_scale;
            scale.d2 = sphere_scale;
            render_obj_idx = 1;
        } break;
        case MJXGeomType::Mesh: {
            scale = Diag3x3 { 1, 1, 1 };
            render_obj_idx = 2 + cfg.geomDataIDs[geom_idx];
        } break;
        default: {
            assert(false);
        } break;
        }
        ctx.get<Scale>(instance) = scale;
        ctx.get<ObjectID>(instance) = ObjectID { render_obj_idx };
    }

    for (CountT cam_idx = 0; cam_idx < (CountT)cfg.numCams; cam_idx++) {
        Entity cam = ctx.makeEntity<CameraEntity>();
        ctx.get<Position>(cam) = Vector3::zero();
        ctx.get<Rotation>(cam) = Quat { 1, 0, 0, 0 };
        render::RenderingSystem::attachEntityToView(
            ctx, cam, 60.f, 0.001f, Vector3::zero());
    }
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
