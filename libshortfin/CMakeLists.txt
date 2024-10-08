# Copyright 2024 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information. SPDX-License-Identifier:
# Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.28)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(
    FATAL_ERROR
      "Do not build in-source. Please remove CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source."
  )
endif()

project(
  "libshortfin"
  VERSION 0.9
  LANGUAGES C CXX)

set(SOVERSION 1)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 20)
# https://discourse.cmake.org/t/cmake-3-28-cmake-cxx-compiler-clang-scan-deps-notfound-not-found/9244/3
set(CMAKE_CXX_SCAN_FOR_MODULES 0)

# Problems with linking libfmt without PIC.
# Turn on PIC on non windows targets.
if(NOT WIN32)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# build options
option(SHORTFIN_BUILD_PYTHON_BINDINGS "Builds Python Bindings" OFF)
option(SHORTFIN_BUILD_TESTS "Builds C++ tests" ON)
option(SHORTFIN_BUNDLE_DEPS "Download dependencies instead of using system libraries" OFF)

set(SHORTFIN_IREE_SOURCE_DIR "" CACHE FILEPATH "Path to IREE source")

# Enabling ASAN. Note that this will work best if building in a completely
# bundled fashion and with an ASAN rigged CPython. Otherwise, various LD_PRELOAD
# hacks are needed. This is merely a develope convenience: people are more
# than welcome to set flags themselves.
option(SHORTFIN_ENABLE_ASAN "Enable ASAN" OFF)
if(SHORTFIN_ENABLE_ASAN)
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
endif()

option(SHORTFIN_SYSTEMS_AMDGPU "Builds for AMD GPU systems" ON)
message(STATUS "libshortfin supported systems:")
if(SHORTFIN_SYSTEMS_AMDGPU)
  message(STATUS "  - AMD GPU")
  add_compile_definitions("SHORTFIN_HAVE_AMDGPU")
endif()
message(STATUS "  - Host")

include(FetchContent)

# Includes.
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake/
)
include(shortfin_library)

# Dependencies.

if(SHORTFIN_BUNDLE_DEPS)
  ## fmt
  FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281 # 10.2.1 (sync with spdlog)
    GIT_SHALLOW TRUE
  )

  ## spdlog
  # We build fmt from source instead, because we also use fmt.
  set(SPDLOG_FMT_EXTERNAL ON)
  FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG        2d4acf8cc321d7783d8f2e22e17a794c6d0e9450 # v1.14.1
    GIT_SHALLOW TRUE
  )

  ## xtl: required for xtensor
  FetchContent_Declare(
    xtl
    GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
    GIT_TAG        a7c1c5444dfc57f76620391af4c94785ff82c8d6 # v0.7.7
    GIT_SHALLOW TRUE
  )

  ## xtensor
  FetchContent_Declare(
    xtensor
    GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
    GIT_TAG        3634f2ded19e0cf38208c8b86cea9e1d7c8e397d # v0.25.0
    GIT_SHALLOW TRUE
  )

  FetchContent_MakeAvailable(fmt spdlog xtl xtensor)
else()
  find_package(spdlog)
  find_package(xtensor)
endif()

## iree runtime

if (NOT SHORTFIN_IREE_SOURCE_DIR AND SHORTFIN_BUNDLE_DEPS)
  FetchContent_Declare(
    iree
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG candidate-20240821.992
    # TODO: We shouldn't have to pull googletest when we are not building tests.
    #       This needs to be fixed with IREE.
    GIT_SUBMODULES "third_party/benchmark third_party/cpuinfo third_party/flatcc third_party/hip-build-deps third_party/googletest"
    GIT_SHALLOW TRUE
  )
  FetchContent_GetProperties(iree)
  if(NOT iree_POPULATED)
    FetchContent_Populate(iree)
  endif()
  set(SHORTFIN_IREE_SOURCE_DIR ${iree_SOURCE_DIR})
endif()

if(SHORTFIN_IREE_SOURCE_DIR)
  set(IREE_BUILD_COMPILER OFF)
  set(IREE_BUILD_TESTS OFF)
  set(IREE_BUILD_SAMPLES OFF)
  # Disable missing submodules error because we are only building the runtime.
  set(IREE_ERROR_ON_MISSING_SUBMODULES OFF)
  # Only enable local_sync/local_task/hip drivers for now.
  set(IREE_HAL_DRIVER_DEFAULTS OFF)
  set(IREE_HAL_DRIVER_LOCAL_SYNC ON)
  set(IREE_HAL_DRIVER_LOCAL_TASK ON)
  if(SHORTFIN_SYSTEMS_AMDGPU)
    set(IREE_HAL_DRIVER_HIP ON)
  endif()
  add_subdirectory(${SHORTFIN_IREE_SOURCE_DIR} shortfin_iree SYSTEM EXCLUDE_FROM_ALL)
else()
  # Try to find iree using find_package
  find_package(IREERuntime)
endif()

# tests

if(SHORTFIN_BUILD_TESTS)
  if (NOT SHORTFIN_IREE_SOURCE_DIR)
    # For now we use gtest shipped alongside with IREE.
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()
  include(GoogleTest)
  enable_testing()
endif()


add_subdirectory(src)

if(SHORTFIN_BUILD_PYTHON_BINDINGS)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  add_subdirectory(bindings/python)
  set(SHORTFIN_PYTHON_CPP_PREBUILT "TRUE") # See setup.py.
  configure_file(setup.py setup.py @ONLY)
  configure_file(pyproject.toml pyproject.toml COPYONLY)
endif()
