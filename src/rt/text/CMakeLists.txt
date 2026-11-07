# The text grammar is the only libddx object built with exceptions, because
# Boost.Parser's parse() uses try/catch.
add_library(ddx_text_objs OBJECT parse.cpp)

set_target_properties(ddx_text_objs PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)

target_compile_definitions(ddx_text_objs PRIVATE DDX_BUILDING BOOST_PARSER_DISABLE_TRACE)
target_link_libraries(ddx_text_objs PRIVATE ddx Boost::headers)
ddx_target_flags(ddx_text_objs EXCEPTIONS)

# Silences C4702 in the disabled trace code, which /external:W0 cannot reach.
if (MSVC)
    target_compile_options(ddx_text_objs PRIVATE /wd4702)
endif ()

target_sources(ddx_rt PRIVATE $<TARGET_OBJECTS:ddx_text_objs>)
