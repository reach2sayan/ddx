target_sources(ddx_md INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/md.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/layouts.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/tensor.hpp>
        # Vendored kokkos/mdspan, one LOCAL PATCH; scripts/mdspan_vendor_diff.py checks it.
        $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/third_party/mdspan.hpp>
)
