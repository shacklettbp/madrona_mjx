set(SIMULATOR_SRCS
    types.hpp sim.hpp sim.inl sim.cpp
)

add_library(madmjx_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(madmjx_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(madmjx_mgr SHARED
    mgr.hpp mgr.cpp
)

target_link_libraries(madmjx_mgr 
    PRIVATE
        madrona_python_utils
        madmjx_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            madmjx_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(madmjx_mgr PRIVATE
        madmjx_gpu_srcs
        madrona_mw_gpu
        madrona_cuda
    )
endif ()

target_compile_definitions(madmjx_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

madrona_python_module(_madrona_mjx_batch_renderer
    bindings.cpp
)

target_link_libraries(_madrona_mjx_batch_renderer PRIVATE
    madmjx_mgr
)

if (TARGET madrona_viz)
    madrona_python_module(_madrona_mjx_visualizer
        viewer.cpp
    )

    target_link_libraries(_madrona_mjx_visualizer PRIVATE 
        madrona_viz madmjx_mgr
    )

    if (TARGET madrona_cuda)
        target_link_libraries(_madrona_mjx_visualizer PRIVATE
            madrona_cuda
        )
    endif()
endif()
