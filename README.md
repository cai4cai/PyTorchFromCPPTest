# PyTorchFromCPPTest
A simple example repository illustrating how to call PyTorch (python) from c++.

Since we use the python side of PyTorch in addition to the c++ side, libtorch is not sufficient. Thankfully, PyTorch (for python) includes libtorch. Our CMakeLists.txt thus looks for the PyTorch python packages and gets the include directories and libraries from there. The pybind11 library from PyTorch is used to create a python interpreter and call an example python script. A tensor is passed from c++ to python and a new tensor is then provided back from python to c++.

A slightly more elaborate example shows the potential usefulness of this construct in a scenario for a c++ program runs a fast loop and at each iteration a threaded call to a slower python function is attempted. To avoid tampering with the GIL, the slow python call is skipped if the runner thread is bussy.

This example is tested though GitHub Actions on linux, mac and windows albeit using only CPU runners. A non-automated test with CUDA can be done through [Google colab](https://colab.research.google.com/github/cai4cai/PyTorchFromCPPTest/blob/main/pytorchfromcpptest_colabcudatest.ipynb).
A local test also confirmed this runs on Apple MPS.

## Resources
- https://docs.nersc.gov/performance/case-studies/CPP_2_Py/
