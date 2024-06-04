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
}

#if 0
inline void printTransforms(Engine &ctx,
                            Position &pos,
                            Rotation &rot,
                            Scale &scale,
                            ObjectID &obj_id)
{
    printf("(%f %f %f) (%f %f %f %f) (%f %f %f) %d\n",
        pos.x, pos.y, pos.z,
        rot.w, rot.x, rot.y, rot.z,
        scale.d0, scale.d1, scale.d2,
        obj_id.idx);
}
#endif

static void setupRenderTasks(TaskGraphBuilder &builder,
                             Span<const TaskGraphNodeID> deps)
{
#if 0
    builder.addToGraph<ParallelForNode<
        Engine, printTransforms, Position, Rotation, Scale, ObjectID>>(deps);
#endif
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

    setupRenderTasks(builder, {sort_sys});
#else
    (void)cfg;
    setupRenderTasks(builder, {});
#endif
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    TaskGraphBuilder &init_builder = taskgraph_mgr.init(TaskGraphID::Init);
    setupInitTasks(init_builder, cfg);

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
            scale = { 0, 0, 0 }; // Hack
            render_obj_idx = (int32_t)RenderPrimObjectIDs::Plane;
        } break;
        case MJXGeomType::Sphere: {
            float sphere_scale = cfg.geomSizes[geom_idx].x;
            scale.d0 = sphere_scale;
            scale.d1 = sphere_scale;
            scale.d2 = sphere_scale;
            scale = { 0, 0, 0 }; // Hack
            render_obj_idx = (int32_t)RenderPrimObjectIDs::Sphere;
        } break;
        case MJXGeomType::Capsule: {
            Vector3 geom_size = cfg.geomSizes[geom_idx];
            scale.d0 = geom_size.x;
            scale.d1 = geom_size.y;
            scale.d2 = geom_size.z;
            scale = { 0, 0, 0 }; // Hack
            render_obj_idx = (int32_t)RenderPrimObjectIDs::Sphere;
        } break;
        case MJXGeomType::Box: {
            Vector3 geom_size = cfg.geomSizes[geom_idx];
            scale.d0 = geom_size.x;
            scale.d1 = geom_size.y;
            scale.d2 = geom_size.z;
            render_obj_idx = (int32_t)RenderPrimObjectIDs::Box;
        } break;
        case MJXGeomType::Mesh: {
            scale = Diag3x3 { 1, 1, 1 };
            render_obj_idx = (int32_t)RenderPrimObjectIDs::NumPrims +
                cfg.geomDataIDs[geom_idx];
        } break;
        case MJXGeomType::Heightfield:
        case MJXGeomType::Ellipsoid:
        case MJXGeomType::Cylinder:
            assert(false);
            break;
        default: {
            assert(false);
        } break;
        }
        ctx.get<Scale>(instance) = scale;
        ctx.get<ObjectID>(instance) = ObjectID { render_obj_idx };
    }

    for (CountT cam_idx = 0; cam_idx < (CountT)cfg.numCams; cam_idx++) {
        Entity cam;
        if (cfg.useDebugCamEntity) {
            cam = ctx.makeRenderableEntity<DebugCameraEntity>();

            ctx.get<Scale>(cam) = Diag3x3 { 0.1, 0.1, 0.1 };
            ctx.get<ObjectID>(cam).idx =
                (int32_t)RenderPrimObjectIDs::DebugCam;
        } else {
            cam = ctx.makeEntity<CameraEntity>();
        }
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
