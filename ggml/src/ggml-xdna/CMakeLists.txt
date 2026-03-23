# Intentionally skip find_package(xrt) — the shipped xrt-config.cmake
# references libxilinxopencl.a which is an FPGA-only library not present
# on NPU-only installs (e.g. CachyOS xrt + xrt-plugin-amdxdna).
# We detect the NPU-relevant libraries directly instead.

find_path(XRT_INCLUDE_DIR
    NAMES xrt/xrt_device.h
    HINTS /usr/include /usr/local/include)

find_library(XRT_CORE_LIB
    NAMES xrt_core
    HINTS /usr/lib /usr/local/lib)

find_library(XRT_COREUTIL_LIB
    NAMES xrt_coreutil
    HINTS /usr/lib /usr/local/lib)

find_library(XRT_DRIVER_XDNA_LIB
    NAMES xrt_driver_xdna
    HINTS /usr/lib /usr/local/lib)
# xrt-plugin-amdxdna ships only the versioned .so.2 without an unversioned
# symlink on some distros (e.g. CachyOS).  Fall back to the versioned name.
if (NOT XRT_DRIVER_XDNA_LIB)
    find_file(XRT_DRIVER_XDNA_LIB
        NAMES libxrt_driver_xdna.so.2
        HINTS /usr/lib /usr/local/lib)
endif()

find_library(XRT_CPP_LIB
    NAMES xrt++
    HINTS /usr/lib /usr/local/lib)

if (XRT_INCLUDE_DIR AND XRT_CORE_LIB AND XRT_COREUTIL_LIB AND XRT_CPP_LIB AND XRT_DRIVER_XDNA_LIB)
    set(XRT_LIBRARIES
        ${XRT_CPP_LIB}
        ${XRT_CORE_LIB}
        ${XRT_COREUTIL_LIB}
        ${XRT_DRIVER_XDNA_LIB})

    message(STATUS "XRT found — building ggml-xdna backend")
    message(STATUS "  Includes : ${XRT_INCLUDE_DIR}")
    message(STATUS "  Libraries: ${XRT_LIBRARIES}")

    ggml_add_backend_library(ggml-xdna ggml-xdna.cpp)

    target_include_directories(ggml-xdna SYSTEM PRIVATE ${XRT_INCLUDE_DIR})
    target_link_libraries(ggml-xdna PRIVATE ${XRT_LIBRARIES})
    target_compile_features(ggml-xdna PRIVATE cxx_std_17)
    target_compile_options(ggml-xdna PRIVATE -Wall -Wextra)

    if (GGML_BUILD_TESTS OR GGML_XDNA)
        add_executable(test-xdna-correctness tests/test-xdna-correctness.cpp)
        target_compile_features(test-xdna-correctness PRIVATE cxx_std_17)
        target_compile_options(test-xdna-correctness PRIVATE -Wall -Wextra)
        target_include_directories(test-xdna-correctness PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME test-xdna-correctness COMMAND test-xdna-correctness)
    endif ()
else()
    message(STATUS "ggml-xdna: XRT not found — skipping backend. "
        "To install: sudo pacman -S cachyos-extra-znver4/xrt cachyos-extra-znver4/xrt-plugin-amdxdna")
    return()
endif()
