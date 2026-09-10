ddx_use_googlebenchmark()

add_executable(benchmarks benchmark.cpp)
target_link_libraries(benchmarks PRIVATE benchmark::benchmark ddx)
ddx_target_flags(benchmarks)

if (DDX_BUILD_JIT)
    add_executable(benchmarks_jit jit/benchmark_jit.cpp)
    target_link_libraries(benchmarks_jit PRIVATE benchmark::benchmark ddx::jit)
    ddx_target_flags(benchmarks_jit)
    ddx_runtime_deps(benchmarks_jit)

    # Their own main(), not Google Benchmark loops.
    add_executable(benchmarks_jit_compile jit/benchmark_jit_compile.cpp)
    target_link_libraries(benchmarks_jit_compile PRIVATE ddx::jit)
    ddx_target_flags(benchmarks_jit_compile)
    ddx_runtime_deps(benchmarks_jit_compile)

    add_executable(benchmarks_hvp jit/benchmark_hvp.cpp)
    target_link_libraries(benchmarks_hvp PRIVATE ddx::jit)
    ddx_target_flags(benchmarks_hvp)
    ddx_runtime_deps(benchmarks_hvp)
endif ()

if (DDX_OPENCL)
    add_executable(benchmarks_cl cl/benchmark_cl.cpp)
    # tests/rt/energy_models.hpp, the model every arm computes.
    target_include_directories(benchmarks_cl PRIVATE "${PROJECT_SOURCE_DIR}/tests")
    target_link_libraries(benchmarks_cl PRIVATE benchmark::benchmark ddx::cl)
    ddx_target_flags(benchmarks_cl)
    ddx_runtime_deps(benchmarks_cl)
endif ()

add_custom_target(benchmark_json
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/benchmark-results"
        COMMAND $<TARGET_FILE:benchmarks>
        --benchmark_min_time=0.05s
        --benchmark_out="${CMAKE_BINARY_DIR}/benchmark-results/benchmarks.json"
        --benchmark_out_format=json
        DEPENDS benchmarks
        USES_TERMINAL
        COMMENT "Running benchmarks and exporting JSON results")
