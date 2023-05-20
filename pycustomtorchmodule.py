import torch  # noqa: F401


def simpleop(inval):
    print("Start python op")
    print(inval)
    return 2 * inval


def opwithglobal(inval):
    print("Start python op")
    print(globalval)  # noqa: F821
    print(inval)
    return globalval + inval  # noqa: F821
