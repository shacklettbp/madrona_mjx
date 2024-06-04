#include "mgr.hpp"

#include <madrona/macros.hpp>
#include <madrona/py/bindings.hpp>

using namespace madrona;

namespace nb = nanobind;

namespace madMJX {

// This file creates the python bindings used by the learning code.
// Refer to the nanobind documentation for more details on these functions.
NB_MODULE(_madrona_mjx_batch_renderer, m) {
    // Each simulator has a madrona submodule that includes base types
    // like madrona::py::Tensor and madrona::py::PyExecMode.
    madrona::py::setupMadronaSubmodule(m);

    nb::class_<VisualizerGPUHandles>(m, "VisualizerGPUHandles");

    nb::class_<Manager>(m, "MadronaBatchRenderer")
        .def("__init__", [](
            Manager *self,
            madrona::py::PyExecMode exec_mode,
            int64_t gpu_id,
            nb::ndarray<const float, nb::shape<-1, 3>,
                nb::device::cpu> mesh_vertices,
            nb::ndarray<const int32_t, nb::shape<-1, 3>,
                nb::device::cpu> mesh_faces,
            nb::ndarray<const int32_t, nb::shape<-1>,
                nb::device::cpu> mesh_vertex_offsets,
            nb::ndarray<const int32_t, nb::shape<-1>,
                nb::device::cpu> mesh_face_offsets,
            nb::ndarray<const int32_t, nb::shape<-1>,
                nb::device::cpu> geom_types,
            nb::ndarray<const int32_t, nb::shape<-1>,
                nb::device::cpu> geom_data_ids,
            nb::ndarray<const float, nb::shape<-1, 3>,
                nb::device::cpu> geom_sizes,
            int64_t num_cams,
            int64_t num_worlds,
            int64_t batch_render_view_width,
            int64_t batch_render_view_height,
            bool add_cam_debug_geo,
            VisualizerGPUHandles *viz_gpu_hdls)
        {
            MJXModelGeometry mesh_geo {
                .vertices = (math::Vector3 *)mesh_vertices.data(),
                .indices = (uint32_t *)mesh_faces.data(),
                .vertexOffsets = (uint32_t *)mesh_vertex_offsets.data(),
                .triOffsets = (uint32_t *)mesh_face_offsets.data(),
                .numVertices = (uint32_t)mesh_vertices.shape(0),
                .numTris = (uint32_t)mesh_faces.shape(0),
                .numMeshes = (uint32_t)mesh_vertex_offsets.shape(0),
            };

            MJXModel mjx_model {
                .meshGeo = mesh_geo,
                .geomTypes = (int32_t *)geom_types.data(),
                .geomDataIDs = (int32_t *)geom_data_ids.data(),
                .geomSizes = (math::Vector3 *)geom_sizes.data(),
                .numGeoms = (uint32_t)geom_types.shape(0),
                .numCams = (uint32_t)num_cams,
            };

            new (self) Manager(Manager::Config {
                .execMode = exec_mode,
                .gpuID = (int)gpu_id,
                .numWorlds = (uint32_t)num_worlds,
                .batchRenderViewWidth = (uint32_t)batch_render_view_width,
                .batchRenderViewHeight = (uint32_t)batch_render_view_height,
                .addCamDebugGeometry = add_cam_debug_geo,
            }, mjx_model, viz_gpu_hdls != nullptr ? *viz_gpu_hdls :
                Optional<VisualizerGPUHandles>::none());
        }, nb::arg("exec_mode"),
           nb::arg("gpu_id"),
           nb::arg("mesh_vertices"),
           nb::arg("mesh_faces"),
           nb::arg("mesh_vertex_offsets"),
           nb::arg("mesh_face_offsets"),
           nb::arg("geom_types"),
           nb::arg("geom_data_ids"),
           nb::arg("geom_sizes"),
           nb::arg("num_cams"),
           nb::arg("num_worlds"),
           nb::arg("batch_render_view_width"),
           nb::arg("batch_render_view_height"),
           nb::arg("add_cam_debug_geo") = false,
           nb::arg("visualizer_gpu_handles") = nb::none(),
           nb::keep_alive<1, 16>())
        .def("init", [](Manager &mgr,
                        nb::ndarray<const float, nb::shape<-1, -1, 3>> geom_pos,
                        nb::ndarray<const float, nb::shape<-1, -1, 4>> geom_rot,
                        nb::ndarray<const float, nb::shape<-1, -1, 3>> cam_pos,
                        nb::ndarray<const float, nb::shape<-1, -1, 4>> cam_rot)

        {
            mgr.init((math::Vector3 *)geom_pos.data(),
                     (math::Quat *)geom_rot.data(),
                     (math::Vector3 *)cam_pos.data(),
                     (math::Quat *)cam_rot.data());
        })
        .def("render", [](Manager &mgr,
                          nb::ndarray<const float, nb::shape<-1, -1, 3>> geom_pos,
                          nb::ndarray<const float, nb::shape<-1, -1, 4>> geom_rot,
                          nb::ndarray<const float, nb::shape<-1, -1, 3>> cam_pos,
                          nb::ndarray<const float, nb::shape<-1, -1, 4>> cam_rot)

        {
            mgr.render((math::Vector3 *)geom_pos.data(),
                       (math::Quat *)geom_rot.data(),
                       (math::Vector3 *)cam_pos.data(),
                       (math::Quat *)cam_rot.data());
        })
        .def("instance_positions_tensor", &Manager::instancePositionsTensor)
        .def("instance_rotations_tensor", &Manager::instanceRotationsTensor)
        .def("camera_positions_tensor", &Manager::cameraPositionsTensor)
        .def("camera_rotations_tensor", &Manager::cameraRotationsTensor)
        .def("rgb_tensor", &Manager::rgbTensor)
        .def("depth_tensor", &Manager::depthTensor)
        .def("xla_entries", [](Manager &mgr)
        {
            Manager *mgr_ptr = &mgr;
            auto sim_encode = nb::bytes((char *)&mgr_ptr, sizeof(char *));

            void (*gpu_init_fn)(cudaStream_t, void **, const char *, size_t) = [](
                cudaStream_t strm, void **buffers, const char *opaque, size_t)
            {
                Manager *mgr = *(Manager **)opaque;

                // The first buffer entry is a token JAX uses for ordering,
                // skip over it
                mgr->gpuStreamInit(strm, buffers + 1);
            };

            void (*gpu_render_fn)(cudaStream_t, void **, const char *, size_t) = [](
                cudaStream_t strm, void **buffers, const char *opaque, size_t)
            {
                Manager *mgr = *(Manager **)opaque;

                // The first buffer entry is a token JAX uses for ordering,
                // skip over it
                mgr->gpuStreamRender(strm, buffers + 1);
            };

            return nb::make_tuple(
                sim_encode,
                nb::capsule(std::bit_cast<void *>(gpu_init_fn), 
                            "xla._CUSTOM_CALL_TARGET"),
                nb::capsule(std::bit_cast<void *>(gpu_render_fn), 
                            "xla._CUSTOM_CALL_TARGET"));
        })
    ;
}

}
