#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>
#include <madrona/py/bindings.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>

using namespace madrona;
using namespace madrona::viz;

namespace nb = nanobind;

namespace madMJX {

struct VisualizerGPUState {
    WindowManager wm;
    WindowHandle window;
    render::GPUHandle renderGPU;

    inline VisualizerGPUState(int64_t window_width,
                              int64_t window_height,
                              int gpu_id)
        : wm(),
          window(wm.makeWindow("MadMJX", window_width, window_height)),
          renderGPU(wm.initGPU(gpu_id, { window.get() }))
    {}

    inline VisualizerGPUHandles getGPUHandles()
    {
        return VisualizerGPUHandles {
            .renderAPI = wm.gpuAPIManager().backend(),
            .renderDev = renderGPU.device(),
        };
    }
};

struct Visualizer {
    Viewer viewer;

    inline Visualizer(VisualizerGPUState &gpu_state, Manager &mgr)
        : viewer(mgr.getRenderManager(), gpu_state.window.get(), {
            .numWorlds = mgr.numWorlds(),
            .simTickRate = 30,
            .cameraMoveSpeed = 10.f,
            .cameraPosition = { 0, -3, 0 },
            .cameraRotation = { 1, 0, 0, 0 },
        })
    {}

    template <typename Fn>
    inline void loop(Manager &mgr, Fn &&sim_cb)
    {
        // Main loop for the viewer
        viewer.loop(
        [&mgr](CountT world_idx, const Viewer::UserInput &input)
        {
            using Key = Viewer::KeyboardKey;
            if (input.keyHit(Key::R)) {
                mgr.triggerReset(world_idx);
            }
        },
        [&mgr](CountT world_idx, CountT,
               const Viewer::UserInput &input)
        {
            using Key = Viewer::KeyboardKey;

            int32_t move = 0;
            if (input.keyPressed(Key::Space)) {
                move = 1;
            }

            mgr.setAction(world_idx, move);
        }, [&]() {
            sim_cb();
        }, []() {});
    }
};

NB_MODULE(madrona_mjx_viz, m) {
    nb::class_<VisualizerGPUState>(m, "VisualizerGPUState")
        .def("__init__", [](VisualizerGPUState *self,
                            int64_t window_width,
                            int64_t window_height,
                            int gpu_id) {
            new (self) VisualizerGPUState(window_width, window_height, gpu_id);
        }, nb::arg("window_width"),
           nb::arg("window_height"),
           nb::arg("gpu_id") = 0)
        .def("get_gpu_handles", &VisualizerGPUState::getGPUHandles, 
             nb::keep_alive<0, 1>())
    ;

    nb::class_<Visualizer>(m, "Visualizer")
        .def("__init__", [](Visualizer *self,
                            VisualizerGPUState *viz_gpu_state,
                            Manager *mgr) {
            new (self) Visualizer(*viz_gpu_state, *mgr);
        }, nb::arg("visualizer_gpu_state"),
           nb::arg("manager"),
           nb::keep_alive<1, 2>(),
           nb::keep_alive<1, 3>())
        .def("loop", [](Visualizer *self, Manager *mgr, nb::callable cb) {
            self->loop(*mgr, [&]() {
                cb();
            });
        })
    ;
}

}
