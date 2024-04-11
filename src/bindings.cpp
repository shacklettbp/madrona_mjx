#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

using namespace madrona;

namespace nb = nanobind;

namespace madMJX {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(madrona_mjx_sim, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor and madrona::py::PyExecMode.
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<VisualizerGPUHandles>(m, "VisualizerGPUHandles");

    nb::class_<Manager>(m, "SimManager")
        .def("__init__", [](Manager *self,
                            madrona::py::PyExecMode exec_mode,
                            int64_t gpu_id,
                            int64_t num_worlds,
                            int64_t max_episode_len,
                            bool enable_batch_renderer,
                            int64_t batch_render_view_width,
                            int64_t batch_render_view_height,
                            VisualizerGPUHandles *viz_gpu_hdls) {
            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .maxEpisodeLength = (uint32_t)max_episode_len,
                .enableBatchRenderer = enable_batch_renderer,
                .batchRenderViewWidth = (uint32_t)batch_render_view_width,
                .batchRenderViewHeight = (uint32_t)batch_render_view_height,
            }, viz_gpu_hdls != nullptr ? *viz_gpu_hdls :
                Optional<VisualizerGPUHandles>::none());
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("num_worlds"),
           nb::arg("max_episode_length"),
           nb::arg("enable_batch_renderer") = false,
           nb::arg("batch_render_view_width") = 64,
           nb::arg("batch_render_view_height") = 64,
           nb::arg("visualizer_gpu_handles") = nb::none(),
           nb::keep_alive<1, 9>())
        .def("init", &Manager::init)
        .def("process_actions", &Manager::processActions)
        .def("post_physics", &Manager::postPhysics)
        .def("process_actions_async", [](Manager &mgr, int64_t strm) {
            mgr.processActionsAsync((cudaStream_t)strm);
        })
        .def("post_physics_async", [](Manager &mgr, int64_t strm) {
            mgr.postPhysicsAsync((cudaStream_t)strm);
        })
        .def("reset_tensor", &Manager::resetTensor)
        .def("done_tensor", &Manager::doneTensor)
        .def("action_tensor", &Manager::actionTensor)
        .def("reward_tensor", &Manager::rewardTensor)
        .def("rigid_body_positions_tensor", &Manager::rigidBodyPositionsTensor)
        .def("rigid_body_rotations_tensor", &Manager::rigidBodyRotationsTensor)
        .def("joint_forces_tensor", &Manager::jointForcesTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
    ;
}

}
