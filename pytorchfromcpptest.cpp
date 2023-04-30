#include <torch/torch.h>
#include <iostream>

int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[]) {
  try {
    std::cout<<"Starting test."<<std::endl;
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "Random 2x3 tensor:" << std::endl << tensor << std::endl;
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
