# The grammar, the one libddx object built with exceptions: Boost.Parser's
# parse() is a try/catch.
add_library(ddx_text_objs OBJECT parse.cpp)

set_target_properties(ddx_text_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

target_compile_definitions(ddx_text_objs PRIVATE DDX_BUILDING BOOST_PARSER_DISABLE_TRACE)
target_link_libraries(ddx_text_objs PRIVATE ddx Boost::headers)
ddx_target_flags(ddx_text_objs EXCEPTIONS)

# C4702 from the disabled trace code; /external:W0 does not reach codegen warnings.
if (MSVC)
    target_compile_options(ddx_text_objs PRIVATE /wd4702)
endif ()

target_sources(ddx_rt PRIVATE $<TARGET_OBJECTS:ddx_text_objs>)
