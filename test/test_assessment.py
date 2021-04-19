import numpy as np

from donut.assessment import get_fp


def test_get_F_score():
    pass

def test_get_fp():
    catch_index = np.array([1, 2, 3, 4])
    real_index= np.array([  3, 4])
    fp_index, fp_num = get_fp(catch_index, real_index)
    print(fp_index,fp_num)