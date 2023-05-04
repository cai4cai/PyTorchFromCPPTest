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

# Get python
find_package(Python REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Found python: ${Python_EXECUTABLE}")

# Find where pytorch lives
file(WRITE ${PyTorchFromCPPTest_BINARY_DIR}/findtorchcmake.py
  "import torch; print(torch.utils.cmake_prefix_path)")
execute_process(COMMAND ${Python_EXECUTABLE} ${PyTorchFromCPPTest_BINARY_DIR}/findtorchcmake.py
                OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
		OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE FOUND_TORCH_CMAKE_PREFIX_PATH)
message(STATUS "Found torch.utils.cmake_prefix_path: ${TORCH_CMAKE_PREFIX_PATH}")

# Find torch for cmake purposes
find_package(Torch REQUIRED PATHS "${TORCH_CMAKE_PREFIX_PATH}/Torch")

# Add torch specific compile flags
add_compile_options(${TORCH_CXX_FLAGS})

# Specify our executable
add_executable(pytorchfromcpptest
  pytorchfromcpptest.cpp)

target_link_libraries(pytorchfromcpptest
  PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(pytorchfromcpptest
  PRIVATE Python::Python)

find_library(TORCHPY_LIB NAMES torch_python HINTS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(pytorchfromcpptest PRIVATE "${TORCHPY_LIB}")

# Give access to python files through a hardcoded definition
cmake_path(NATIVE_PATH PyTorchFromCPPTest_SOURCE_DIR NORMALIZE NATIVE_CUSTOM_MODULE_SYS_PATH)
if(WIN32)
  string(REPLACE "\\" "\\\\" NATIVE_CUSTOM_MODULE_SYS_PATH ${NATIVE_CUSTOM_MODULE_SYS_PATH})
endif(WIN32)
target_compile_definitions(pytorchfromcpptest PRIVATE CUSTOM_MODULE_SYS_PATH="${NATIVE_CUSTOM_MODULE_SYS_PATH}")
set(torchpath "${TORCH_CMAKE_PREFIX_PATH}/../../..")
cmake_path(NATIVE_PATH torchpath NORMALIZE NATIVE_CUSTOM_TORCH_SYS_PATH)
if(WIN32)
  string(REPLACE "\\" "\\\\" NATIVE_CUSTOM_TORCH_SYS_PATH ${NATIVE_CUSTOM_TORCH_SYS_PATH})
endif(WIN32)
target_compile_definitions(pytorchfromcpptest PRIVATE CUSTOM_TORCH_SYS_PATH="${NATIVE_CUSTOM_TORCH_SYS_PATH}")

# Add to ctest
enable_testing()
add_test(NAME pytorchfromcpptest COMMAND pytorchfromcpptest)