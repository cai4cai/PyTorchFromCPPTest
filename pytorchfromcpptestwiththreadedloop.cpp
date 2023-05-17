// Copyright 2023 Tom Vercauteren. All rights reserved.
//
// This software is licensed under the Apache 2 License.
// See the LICENSE file for details.

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <chrono>
#include <future>
#include <iostream>
#include <thread>

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
  // using namespace std::chrono_literals;
  using std::chrono_literals::operator""ms;

  std::cout << "Starting test from c++" << std::endl;
  std::cout << "PyTorch version: " << TORCH_VERSION << std::endl;

  torch::Device device = getbesttorchdevice();

  torch::Tensor globaltensor = torch::arange(0, 3, device);
  std::cout << "Example globaltensor (c++ side):" << std::endl
            << globaltensor << std::endl;

  py::scoped_interpreter guard{};

  py::module pycustomtorchmodule = setupandloadpymodule();

  // Add variables to the custom module
  pycustomtorchmodule.attr("globalval") = globaltensor;

  // Prepare thread. Don't use a thread pool to avoid messing up with th GIL
  py::gil_scoped_release no_gil;
  std::future<void> pyfuture;

  // Define the function that will call python in the thread
  // Silence the warning about using a reference for teh tensor as this
  // leds to crashs and tensors use shallow copy anyway
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  auto wrappedop = [](py::module& custommodule, torch::Tensor inputtensor) {
    py::gil_scoped_acquire gil;
    py::function pyop = custommodule.attr("opwithglobal");
    torch::Tensor pyretval = pyop(inputtensor).cast<torch::Tensor>();

    // Simulate tie consuming task
    std::this_thread::sleep_for(50ms);

    std::cout << "Python return value (in c++) " << std::endl
              << pyretval << std::endl;
  };

  // Simulate a fast loop running on c++
  for (int i = 0; i < 100; ++i) {
    // Let's assume we create a tensor
    torch::Tensor localtensor = torch::arange(i, i + 3, device);
    // std::cout << "Example localtensor (c++ side):" << std::endl
    //           << localtensor << std::endl;

    // Simulate fast (but not immediate) processing
    std::this_thread::sleep_for(5ms);

    // Use wait_for() with zero milliseconds to check thread status.
    bool threadready = ((!pyfuture.valid()) ||
                        (pyfuture.wait_for(0ms) == std::future_status::ready));
    if (threadready) {
      // clone the tensor
      torch::Tensor localtensorclone = torch::clone(localtensor);

      // Run Python op
      std::cout << "Launching thread at iter " << i << " with cloned tensor "
                << std::endl
                << localtensorclone << std::endl;
      pyfuture = std::async(std::launch::async, wrappedop,
                            std::ref(pycustomtorchmodule), localtensorclone);
    } else {
      std::cout << "Thread is busy at iter " << i << std::endl;
    }
  }

  pyfuture.get();

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
