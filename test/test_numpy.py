import numpy as np


def test_sort():
    arr = np.array([1, 2, 3, 4])
    s_arr = np.argsort(-arr)
    print(arr[s_arr])
