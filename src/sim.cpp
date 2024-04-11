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

    registry.registerComponent<Action>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    registry.registerComponent<JointForce>();

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<EpisodeState>();

    registry.registerArchetype<Joint>();
    registry.registerArchetype<RigidBody>();
    registry.registerArchetype<RenderObject>();
    registry.registerArchetype<Agent>();

    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::EpisodeReset);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::EpisodeDone);

    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::AgentAction);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::AgentReward);

    registry.exportColumn<RigidBody, Position>(
        (uint32_t)ExportID::RigidBodyPositions);
    registry.exportColumn<RigidBody, Rotation>(
        (uint32_t)ExportID::RigidBodyRotations);
    registry.exportColumn<Joint, JointForce>(
        (uint32_t)ExportID::JointForces);
}

inline void actionSystem(Engine &ctx,
                         Action action)
{
    JointForce &joint_force = ctx.get<JointForce>(ctx.data().joint);
    if (action.move == 1) {
        joint_force = { 1.f };
    } else {
        joint_force = { -1.f };
    }
}

inline void checkEpisodeFinishedSystem(Engine &ctx,
                                       Done &done)
{
    // Termination conditions taken from
    // https://www.gymlibrary.dev/environments/classic_control/cart_pole/

    {
        Entity pole = ctx.data().pole;
        Quat q = ctx.get<Rotation>(pole);

        float sinp = sqrtf(1.f + 2.f * (q.w * q.y - q.x * q.z));
        float cosp = sqrtf(1.f - 2.f * (q.w * q.y - q.x * q.z));
        float angle = 2.f * atan2f(sinp, cosp) - math::pi / 2.f;

        if (fabsf(angle) > math::toRadians(12)) {
            done = { 1 };
            return;
        }
    }

    {
        Entity cart = ctx.data().cart;
        Vector3 cart_position = ctx.get<Position>(cart);

        if (fabsf(cart_position.x) > 2.4f) {
            done = { 1 };
            return;
        }
    }

    {
        EpisodeState &episode_state = ctx.singleton<EpisodeState>();
        episode_state.curStep += 1;

        if (episode_state.curStep == ctx.data().maxStepsPerEpisode) {
            done = { 1 };
            return;
        }
    }

    done = { 0 };
}
    
static inline void initState(Engine &ctx)
{
    Entity cart = ctx.data().cart;
    ctx.get<Position>(cart) = Vector3::zero();
    ctx.get<Rotation>(cart) = Quat { 1, 0, 0, 0 };

    Entity pole = ctx.data().pole;
    ctx.get<Position>(pole) = Vector3::zero();
    ctx.get<Rotation>(pole) = Quat { 1, 0, 0, 0 };

    ctx.singleton<EpisodeState>() = {
        .curStep = 0,
    };
}

// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
//
// If a reset is needed, cleanup the existing world and generate a new one.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t force_reset = reset.reset;
    int32_t done = ctx.get<Done>(ctx.data().agent).isDone;

    if (force_reset == 1 || done == 1) {
        //initState(ctx);
    }
}

// Computes reward for each agent and keeps track of the max distance achieved
// so far through the challenge. Continuous reward is provided for any new
// distance achieved.
inline void rewardSystem(Engine &,
                         Reward &reward)
{
    reward = { 1.f };
}

// Helper function for sorting nodes in the taskgraph.
// Sorting is only supported / required on the GPU backend,
// since the CPU backend currently keeps separate tables for each world.
// This will likely change in the future with sorting required for both
// environments
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraphNodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

static void setupResetAndObservationsTasks(TaskGraphBuilder &builder,
                                           Span<const TaskGraphNodeID> deps)
{
    // Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>(deps);

    RenderingSystem::setupTasks(builder, {reset_sys});
}

static void setupInitTasks(TaskGraphBuilder &builder)
{
#ifdef MADRONA_GPU_MODE
    auto sort = queueSortByWorld<RigidBody>(
        builder, {});
    sort = queueSortByWorld<RenderObject>(
        builder, {sort});
    sort = queueSortByWorld<Joint>(
        builder, {sort});
    sort = queueSortByWorld<Agent>(
        builder, {sort});

    // This isn't necessary since we don't delete any entities in this
    // example but leaving it around so I don't forget about it
    // when this gets more complex.
    auto recycle = builder.addToGraph<RecycleEntitiesNode>({sort});
#endif

    setupResetAndObservationsTasks(builder, {
#ifdef MADRONA_GPU_MODE
        sort, recycle,
#endif
    });
}

static void setupProcessActionsTasks(TaskGraphBuilder &builder)
{
    builder.addToGraph<ParallelForNode<Engine,
        actionSystem,
            Action
        >>({});
}

static void setupPostPhysicsTasks(TaskGraphBuilder &builder)
{
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        checkEpisodeFinishedSystem,
            Done
    >>({});

    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Reward
        >>({});

    setupResetAndObservationsTasks(builder, {done_sys, reward_sys});
}

// Build the task graph
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &)
{
    TaskGraphBuilder &init_builder = taskgraph_mgr.init(TaskGraphID::Init);
    setupInitTasks(init_builder);

    TaskGraphBuilder &process_actions_builder =
        taskgraph_mgr.init(TaskGraphID::ProcessActions);
    setupProcessActionsTasks(process_actions_builder);

    TaskGraphBuilder &post_physics_builder =
        taskgraph_mgr.init(TaskGraphID::PostPhysics);
    setupPostPhysicsTasks(post_physics_builder);
}

Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    maxStepsPerEpisode = cfg.maxStepsPerEpisode;

    RenderingSystem::init(ctx, cfg.renderBridge);

    agent = ctx.makeEntity<Agent>();
    ctx.get<Position>(agent) = Vector3 { 0, -3, 0 };
    ctx.get<Rotation>(agent) = Quat { 1, 0, 0, 0 };
    ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
    ctx.get<Action>(agent).move = 0;

    render::RenderingSystem::attachEntityToView(
        ctx, agent, 60.f, 0.001f, Vector3::zero());

    cart = ctx.makeRenderableEntity<RigidBody>();
    ctx.get<Scale>(cart) = Diag3x3 { 1, 1, 1 };
    ctx.get<ObjectID>(cart) = ObjectID { (int32_t)SimObject::Cart };

    pole = ctx.makeRenderableEntity<RigidBody>();
    ctx.get<Scale>(pole) = Diag3x3 { 1, 1, 1 };
    ctx.get<ObjectID>(pole) = ObjectID { (int32_t)SimObject::Pole };

    joint = ctx.makeEntity<Joint>();
    ctx.get<JointForce>(joint) = { 0 };

    Entity backdrop = ctx.makeRenderableEntity<RenderObject>();
    ctx.get<Position>(backdrop) = Vector3 { 0, 10, 0 };
    ctx.get<Rotation>(backdrop) = Quat::angleAxis(math::pi / 2.f, math::right);
    ctx.get<Scale>(backdrop) = Diag3x3 { 1, 1, 1 };
    ctx.get<ObjectID>(backdrop) = ObjectID { (int32_t)SimObject::Backdrop };

    initState(ctx);
}

// This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
