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
                            nb::ndarray<const float, nb::shape<-1, 3>,
                                nb::device::cpu> geo_vertices,
                            nb::ndarray<const int32_t, nb::shape<-1, 3>,
                                nb::device::cpu> geo_faces,
                            nb::ndarray<const int32_t, nb::shape<-1>,
                                nb::device::cpu> geo_mesh_vertex_offsets,
                            nb::ndarray<const int32_t, nb::shape<-1>,
                                nb::device::cpu> geo_mesh_face_offsets,
                            int64_t num_worlds,
                            int64_t batch_render_view_width,
                            int64_t batch_render_view_height,
                            VisualizerGPUHandles *viz_gpu_hdls)
        {
            MJXModelGeometry geo {
                .vertices = (math::Vector3 *)geo_vertices.data(),
                .indices = (uint32_t *)geo_faces.data(),
                .vertexOffsets = (uint32_t *)geo_mesh_vertex_offsets.data(),
                .triOffsets = (uint32_t *)geo_mesh_face_offsets.data(),
                .numVertices = (uint32_t)geo_vertices.shape(0),
                .numTris = (uint32_t)geo_faces.shape(0),
                .numMeshes = (uint32_t)geo_mesh_vertex_offsets.shape(0),
            };

            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .batchRenderViewWidth = (uint32_t)batch_render_view_width,
                .batchRenderViewHeight = (uint32_t)batch_render_view_height,
            }, geo, viz_gpu_hdls != nullptr ? *viz_gpu_hdls :
                Optional<VisualizerGPUHandles>::none());
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("geo_vertices"),
           nb::arg("geo_faces"),
           nb::arg("geo_mesh_vertex_offsets"),
           nb::arg("geo_mesh_face_offsets"),
           nb::arg("num_worlds"),
           nb::arg("batch_render_view_width"),
           nb::arg("batch_render_view_height"),
           nb::arg("visualizer_gpu_handles") = nb::none(),
           nb::keep_alive<1, 9>())
        .def("init", &Manager::init)
        .def("render", &Manager::render)
        .def("render_async", [](Manager &mgr, int64_t strm) {
            mgr.renderAsync((cudaStream_t)strm);
        })
        .def("instance_positions_tensor", &Manager::instancePositionsTensor)
        .def("instance_rotations_tensor", &Manager::instanceRotationsTensor)
        .def("camera_positions_tensor", &Manager::cameraPositionsTensor)
        .def("camera_rotations_tensor", &Manager::cameraRotationsTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
    ;
}

}
