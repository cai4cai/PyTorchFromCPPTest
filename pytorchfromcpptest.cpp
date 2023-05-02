// Copyright 2023 Tom Vercauteren. All rights reserved.
//
// This software is licensed under the Apache 2 License.
// See the LICENSE file for details.

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <iostream>

namespace py = pybind11;

int hybridcall() {
  std::cout << "Starting test from c++" << std::endl;
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << "Random 2x3 tensor (c++ side):" << std::endl
            << tensor << std::endl;

  py::scoped_interpreter guard{};

  // Add source dir to python path
  py::module sys = py::module::import("sys");
  sys.attr("path").attr("insert")(1, CUSTOM_MODULE_SYS_PATH);

  // Add torch from libtorch dir to python path
  sys.attr("path").attr("insert")(1, CUSTOM_TORCH_SYS_PATH);

  // Load custom python module
  py::module pycustomtorchmodule = py::module::import("pycustomtorchmodule");
  std::cout << "Custom python module loaded from " << CUSTOM_MODULE_SYS_PATH << std::endl;

  py::module pytorchmodule = py::module::import("torch");
  std::cout << "Python torch module loaded from " << CUSTOM_TORCH_SYS_PATH << std::endl;

  // Run Python op
  py::function pyop = pycustomtorchmodule.attr("simpleop");
  int testinput = 21;
  int pyretval = pyop(testinput).cast<py::int_>();
  // The line below currently makes thes test segfault
  //int pyretval = pyop(tensor).cast<py::int_>();
  //auto pyretval = pyop(tensor);
  std::cout << "Python return value " << pyretval << std::endl;

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
