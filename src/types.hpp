#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/rand.hpp>
#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>

namespace madMJX {

// Include several madrona types into the simulator namespace for convenience
using madrona::Entity;
using madrona::RandKey;
using madrona::CountT;
using madrona::base::Position;
using madrona::base::Rotation;
using madrona::base::Scale;
using madrona::base::ObjectID;
using madrona::math::Vector3;
using madrona::math::Quat;

// WorldReset is a per-world singleton component that causes the current
// episode to be terminated and the world regenerated
// (Singleton components like WorldReset can be accessed via Context::singleton
// (eg ctx.singleton<WorldReset>().reset = 1)
struct WorldReset {
    int32_t reset;
};

// Discrete action component.
// repeated here for clarity
struct Action {
    int32_t move; // [0, 1]
};

// Per-agent reward
// Exported as an [N, 1] float tensor to training code
struct Reward {
    float v;
};

// Per-agent component that indicates that the agent's episode is finished
// This is exported per-agent for simplicity in the training code
struct Done {
    // Currently bool components are not supported due to
    // padding issues, so Done is an int32_t
    int32_t isDone;
};

struct JointForce {
    float force;
};

struct EpisodeState {
    uint32_t curStep;
};

/* ECS Archetypes */
struct Joint : public madrona::Archetype<
    JointForce
> {};

struct RigidBody : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,

    // All entities with the Renderable component will be drawn by the
    // viewer and batch renderer
    madrona::render::Renderable
> {};

struct RenderObject : public madrona::Archetype<
    Position,
    Rotation,
    Scale,
    ObjectID,

    // All entities with the Renderable component will be drawn by the
    // viewer and batch renderer
    madrona::render::Renderable
> {};

struct Agent : public madrona::Archetype<
    Position, // Note position / rotation is for the camera
    Rotation,
    Scale,
    Action,
    Reward,
    Done,
    madrona::render::RenderCamera
> {};

}
