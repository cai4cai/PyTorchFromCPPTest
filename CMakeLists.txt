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
    -checks=-*,portability-*,bugprone-*,readability-,clang-analyzer-*,performance-*,cppcoreguidelines-*,-cppcoreguidelines-avoid-magic-numbers;
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
endif()
target_compile_definitions(pytorchfromcpptest PRIVATE CUSTOM_MODULE_SYS_PATH="${NATIVE_CUSTOM_MODULE_SYS_PATH}")
set(torchpath "${TORCH_CMAKE_PREFIX_PATH}/../../..")
cmake_path(NATIVE_PATH torchpath NORMALIZE NATIVE_CUSTOM_TORCH_SYS_PATH)
if(WIN32)
  string(REPLACE "\\" "\\\\" NATIVE_CUSTOM_TORCH_SYS_PATH ${NATIVE_CUSTOM_TORCH_SYS_PATH})
endif()
target_compile_definitions(pytorchfromcpptest PRIVATE CUSTOM_TORCH_SYS_PATH="${NATIVE_CUSTOM_TORCH_SYS_PATH}")

# Specify another executable
add_executable(pytorchfromcpptestwiththreadedloop
  pytorchfromcpptestwiththreadedloop.cpp)
target_link_libraries(pytorchfromcpptestwiththreadedloop
  PRIVATE "${TORCH_LIBRARIES}")
target_link_libraries(pytorchfromcpptestwiththreadedloop
  PRIVATE Python::Python)
target_link_libraries(pytorchfromcpptestwiththreadedloop
  PRIVATE "${TORCHPY_LIB}")
target_compile_definitions(pytorchfromcpptestwiththreadedloop PRIVATE CUSTOM_MODULE_SYS_PATH="${NATIVE_CUSTOM_MODULE_SYS_PATH}")
target_compile_definitions(pytorchfromcpptestwiththreadedloop PRIVATE CUSTOM_TORCH_SYS_PATH="${NATIVE_CUSTOM_TORCH_SYS_PATH}")

# Some systems needs to link explicitely with pthread
find_package(Threads REQUIRED)
target_link_libraries(pytorchfromcpptestwiththreadedloop
  PRIVATE Threads::Threads)

# Add to ctest
enable_testing()

# There is no rpath on windows -> use cmake hack
# to help ctest find the dlls
function(getpathsforctest path_list target)
  get_target_property(libs ${target} LINK_LIBRARIES)
  set(lib_paths "")
  foreach(lib ${libs})
    set(lib_file "")
    if (TARGET ${lib})
      get_target_property(lib_file ${lib} LOCATION)
    elseif(EXISTS ${lib})
      set(lib_file ${lib})
    endif()
    if(EXISTS ${lib_file})
      get_filename_component(lib_path ${lib_file} DIRECTORY)
      cmake_path(NATIVE_PATH lib_path NORMALIZE lib_path)
      list(APPEND lib_paths ${lib_path})
    endif()
  endforeach()
  list(REMOVE_DUPLICATES lib_paths)

  if(WIN32)
    list(JOIN lib_paths "\\\;" lib_paths)
  else()
    list(JOIN lib_paths ":" lib_paths)
  endif()
  
  set(${path_list} ${lib_paths} PARENT_SCOPE)
endfunction()

add_test(NAME pytorchfromcpptest COMMAND pytorchfromcpptest)
if(WIN32)
  getpathsforctest(test_paths pytorchfromcpptest)
  message("pytorchfromcpp test_paths: ${test_paths}")
  set_property(TEST pytorchfromcpptest PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${test_paths}")
endif()

add_test(NAME pytorchfromcpptestwiththreadedloop COMMAND pytorchfromcpptestwiththreadedloop)
if(WIN32)
  getpathsforctest(test_paths pytorchfromcpptestwiththreadedloop)
  message("pytorchfromcpptestwiththreadedloop test_paths: ${test_paths}")
  set_property(TEST pytorchfromcpptestwiththreadedloop PROPERTY ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${test_paths}")
endif()
