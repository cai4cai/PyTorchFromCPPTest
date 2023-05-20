// Copyright 2023 Tom Vercauteren. All rights reserved.
//
// This software is licensed under the Apache 2 License.
// See the LICENSE file for details.

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>

namespace py = pybind11;

torch::Device getbesttorchdevice() {
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available. Running on GPU." << std::endl;
    device = torch::kCUDA;
  }
#if TORCH_VERSION_MAJOR >= 2
  // See https://github.com/pytorch/pytorch/issues/96425
#if TORCH_VERSION_MAJOR >= 2 && TORCH_VERSION_MINOR > 0
  if (torch::mps::is_available()) {
#else
  if (at::hasMPS()) {
#endif
    std::cout << "MPS is available. Running on GPU." << std::endl;
    device = torch::kMPS;
  }
#endif
  return device;
}

py::module setupandloadpymodule() {
  // Add source dir to python path
  py::module sys = py::module::import("sys");
  sys.attr("path").attr("insert")(1, CUSTOM_MODULE_SYS_PATH);

  // Add torch from libtorch dir to python path
  sys.attr("path").attr("insert")(1, CUSTOM_TORCH_SYS_PATH);

  // Load custom python module
  py::module pycustomtorchmodule = py::module::import("pycustomtorchmodule");
  std::cout << "Custom python module loaded from " << CUSTOM_MODULE_SYS_PATH
            << std::endl;

  // py::module pytorchmodule = py::module::import("torch");
  // std::cout << "Python torch module loaded from " << CUSTOM_TORCH_SYS_PATH
  // << std::endl;*/

  return pycustomtorchmodule;
}

int hybridcall() {
  std::cout << "Starting test from c++" << std::endl;
  std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;

  torch::Device device = getbesttorchdevice();

  torch::Tensor tensor = torch::rand({2, 3}, device);
  std::cout << "Random 2x3 tensor (c++ side):" << std::endl
            << tensor << std::endl;

  py::scoped_interpreter guard{};

  // As per
  // https://github.com/pybind/pybind11/discussions/4673#discussioncomment-5939343
  // the python interpreter has to be alive to properly catch exceptions
  // stemming from python See also
  // https://pybind11.readthedocs.io/en/stable/reference.html#_CPPv4NK17error_already_set4whatEv
  try {
    py::module pycustomtorchmodule = setupandloadpymodule();

    // Run Python op
    py::function pyop = pycustomtorchmodule.attr("simpleop");
    torch::Tensor pyretval = pyop(tensor).cast<torch::Tensor>();
    std::cout << "Python return value " << std::endl << pyretval << std::endl;
  } catch (const py::error_already_set& e) {
    std::cout << "Rethrowing py::error_already_set exception" << std::endl;
    throw std::runtime_error(e.what());
  }

  py::gil_scoped_release no_gil;

  return EXIT_SUCCESS;
}

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
  try {
    return hybridcall();
  } catch (const std::exception& e) {
    // standard exceptions
    std::cout << "Caught std::exception in main: " << e.what() << std::endl;
    return EXIT_FAILURE;
  } catch (...) {
    // everything else
    std::cout << "Caught unknown exception in main" << std::endl;
    return EXIT_FAILURE;
  }
}
