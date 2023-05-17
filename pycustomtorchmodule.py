import torch


def simpleop(inval):
    print("Start python op")
    print(inval)
    return 2 * inval


def opwithglobal(inval):
    print("Start python op")
    print(globalval)
    print(inval)
    return globalval + inval
