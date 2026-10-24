cmake_minimum_required(VERSION 3.26)
project(test_package LANGUAGES CXX)

find_package(ddx CONFIG REQUIRED COMPONENTS rt)

add_executable(test_package test_package.cpp)
target_link_libraries(test_package PRIVATE ddx::rt)
target_compile_features(test_package PRIVATE cxx_std_23)
# The flags libddx was built with: -fno-exceptions, the ISA level.
target_compile_options(test_package PRIVATE ${ddx_CODEGEN_FLAGS})
