# A consumer of ddx as a separate project, driven by ../CMakeLists.txt; it
# fails when find_package(ddx) does not give a consumer what it needs.
cmake_minimum_required(VERSION 3.26)
project(ddx_package_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(DDX_CONSUMER_COMPONENTS "" CACHE STRING "Components to request of ddx")

if (DDX_CONSUMER_COMPONENTS)
    find_package(ddx REQUIRED COMPONENTS ${DDX_CONSUMER_COMPONENTS})
else ()
    find_package(ddx REQUIRED)
endif ()

message(STATUS "ddx ${ddx_VERSION} from ${ddx_DIR}")
message(STATUS "ddx built with: ${ddx_CODEGEN_FLAGS} (FP contraction: ${ddx_FP_CONTRACT})")

add_executable(consumer consumer.cpp)
target_link_libraries(consumer PRIVATE ddx::ddx)
target_compile_options(consumer PRIVATE ${ddx_CODEGEN_FLAGS})

# Off what find_package reported, not cache variables the parent never passes.
target_link_libraries(consumer PRIVATE ddx::rt)
target_compile_definitions(consumer PRIVATE DDX_CONSUMER_RT)
if (ddx_jit_FOUND)
    target_link_libraries(consumer PRIVATE ddx::jit)
endif ()

# Every header the package ships, against the Boost it ships.
get_target_property(DDX_INCLUDE_DIRS ddx::util INTERFACE_INCLUDE_DIRECTORIES)
list(GET DDX_INCLUDE_DIRS 0 DDX_INCLUDE_DIR)
file(GLOB_RECURSE DDX_HEADERS RELATIVE "${DDX_INCLUDE_DIR}" "${DDX_INCLUDE_DIR}/*.hpp")
list(FILTER DDX_HEADERS EXCLUDE REGEX "^boost/")
if (NOT DDX_HEADERS)
    message(FATAL_ERROR "no ddx headers under ${DDX_INCLUDE_DIR}")
endif ()
list(SORT DDX_HEADERS)
set(DDX_ALL_HEADERS_CPP "")
foreach (header IN LISTS DDX_HEADERS)
    string(APPEND DDX_ALL_HEADERS_CPP "#include \"${header}\"\n")
endforeach ()
string(APPEND DDX_ALL_HEADERS_CPP "int main() { return 0; }\n")
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/all_headers.cpp"
        CONTENT "${DDX_ALL_HEADERS_CPP}")

add_executable(all_headers "${CMAKE_CURRENT_BINARY_DIR}/all_headers.cpp")
target_link_libraries(all_headers PRIVATE ddx::ddx ddx::rt)
target_compile_options(all_headers PRIVATE ${ddx_CODEGEN_FLAGS})
if (ddx_jit_FOUND)
    target_link_libraries(all_headers PRIVATE ddx::jit)
endif ()

enable_testing()
add_test(NAME consumer_runs COMMAND consumer)
add_test(NAME all_headers_runs COMMAND all_headers)
