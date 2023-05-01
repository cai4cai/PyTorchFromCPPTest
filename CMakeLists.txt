cmake_minimum_required(VERSION 3.25)
project(PyTorchFromCPPTest)

# libtorch doesn't seem happy with c++20
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Warnings compiler flags
if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Optionally enable clang-tidy
find_program(CLANG_TIDY_EXE NAMES clang-tidy PATHS /opt/homebrew/opt/llvm/bin/)
if(NOT CLANG_TIDY_EXE)
  message(STATUS "clang-tidy not found. Skipping corresponding checks.")
else()
  set(CMAKE_CXX_CLANG_TIDY 
    ${CLANG_TIDY_EXE};
    -header-filter=.*;
    -checks=-*,portability-*,bugprone-*,readability-,clang-analyzer-*,performance-*,cppcoreguidelines-*;
    )
  message(STATUS "Found clang-tidy: ${CLANG_TIDY_EXE}.")
endif()

# fetch libtorch
include(FetchContent)

# Get libtorch binaries. For simplicity, cpu-only version are used here
if(MSVC)
  set(libtorch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-2.0.0%2Bcpu.zip")
  set(libtorch_MD5 "558517f2feba033f57233d3e512ba8af")
elseif(LINUX)
  set(libtorch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.0%2Bcpu.zip")
  set(libtorch_MD5 "8f9fa09c72c05cffef7135711dcac92a")
elseif(APPLE)
  if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64")
    # Official libtorch doesn't support arm64 on mac
    set(libtorch_URL "https://github.com/mlverse/libtorch-mac-m1/releases/download/LibTorch/libtorch-v2.0.0.zip")
    set(libtorch_MD5 "eb3d8b8567a12d22fcd1e144afa1f791")
  else()
    set(libtorch_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.0.0.zip")
    set(libtorch_MD5 "b1a87518e95f24452af71759b64f2e1f")
  endif()
endif()
message(STATUS "Using libtorch from: ${libtorch_URL}")

FetchContent_Declare(
  libtorch
  URL        ${libtorch_URL}
  URL_HASH   MD5=${libtorch_MD5}
  SYSTEM
  FIND_PACKAGE_ARGS NAMES Torch
)

FetchContent_MakeAvailable(libtorch)
find_package(Torch REQUIRED PATHS ${libtorch_SOURCE_DIR}/share/cmake/Torch)
add_compile_options(${TORCH_CXX_FLAGS})

find_package(Python REQUIRED COMPONENTS Interpreter Development)

add_executable(pytorchfromcpptest
  pytorchfromcpptest.cpp)

if(APPLE AND (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "arm64"))
  # Hack to workaround https://github.com/mlverse/libtorch-mac-m1/issues/4 
  target_link_libraries(pytorchfromcpptest
    PRIVATE "${TORCH_INSTALL_PREFIX}/lib/libomp.dylib")

  add_custom_command(TARGET pytorchfromcpptest 
    POST_BUILD COMMAND 
    ${CMAKE_INSTALL_NAME_TOOL} -change /opt/homebrew/opt/libomp/lib/libomp.dylib @rpath/libomp.dylib 
    $<TARGET_FILE:pytorchfromcpptest>)
endif()

target_link_libraries(pytorchfromcpptest
  PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(pytorchfromcpptest
  PRIVATE Python::Python)

# Give access to python files through a hardcoded definition
target_compile_definitions(pytorchfromcpptest PRIVATE CUSTOM_SYS_PATH="${PyTorchFromCPPTest_SOURCE_DIR}")

# Add to ctest
enable_testing()
add_test(NAME pytorchfromcpptest COMMAND pytorchfromcpptest)