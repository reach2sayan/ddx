# The only LLVM-aware code. MSVC needs an archive, which
# WINDOWS_EXPORT_ALL_SYMBOLS skips; elsewhere --exclude-libs would hide one.
if (MSVC)
    add_library(ddx_jit_objs STATIC codegen.cpp compiler.cpp)
else ()
    add_library(ddx_jit_objs OBJECT codegen.cpp compiler.cpp)
endif ()

set_target_properties(ddx_jit_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

target_compile_definitions(ddx_jit_objs PRIVATE DDX_BUILDING ${LLVM_DEFINITIONS_LIST})
target_include_directories(ddx_jit_objs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(ddx_jit_objs SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_link_libraries(ddx_jit_objs PRIVATE ddx Boost::headers ${DDX_LLVM_LIBS})
ddx_target_flags(ddx_jit_objs)

# Match LLVM's RTTI setting, or vtables from LLVM headers reference typeinfo
# that LLVM never built.
if (NOT LLVM_ENABLE_RTTI)
    target_compile_options(ddx_jit_objs PRIVATE
            $<IF:$<CXX_COMPILER_ID:MSVC>,/GR-,-fno-rtti>)
endif ()

if (MSVC)
    target_link_libraries(ddx_rt PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,ddx_jit_objs>)
else ()
    target_sources(ddx_rt PRIVATE $<TARGET_OBJECTS:ddx_jit_objs>)
endif ()
add_library(ddx::jit ALIAS ddx_rt)

# PUBLIC because consumers read these. The JIT defaults follow libddx's own
# build: its optimisation level (never 0) and its FP contraction.
string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type)
set(jit_opt 2)
if (build_type STREQUAL "DEBUG")
    set(jit_opt 1)
elseif (build_type STREQUAL "RELEASE")
    set(jit_opt 3)
endif ()
target_compile_definitions(ddx_rt PUBLIC
        DDX_HAS_JIT
        DDX_JIT_DEFAULT_OPT=${jit_opt}
        DDX_JIT_DEFAULT_CONTRACT=$<BOOL:${DDX_FP_FLAGS}>)

# LLVM is linked in statically and hidden; tests/check_exports.cmake checks it.
target_link_libraries(ddx_rt PRIVATE ${DDX_LLVM_LIBS})
if (NOT MSVC)
    # --exclude-libs hides LLVM's symbols; the map hides ddx symbols that
    # mention LLVM types.
    set(hide_llvm "${CMAKE_CURRENT_SOURCE_DIR}/hide_llvm.map")
    target_link_options(ddx_rt PRIVATE -Wl,--exclude-libs,ALL -Wl,--gc-sections
            "-Wl,--version-script=${hide_llvm}")
    set_property(TARGET ddx_rt APPEND PROPERTY LINK_DEPENDS "${hide_llvm}")
endif ()
