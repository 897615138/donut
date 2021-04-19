
import numpy as np

from donut.util.time_util import get_constant_timestamp


def test_constant_time():
    time = [146962380, 146962410, 146962740, 146967720, 146967750, 1469677800,
            146967810, 146967840, 146967870, 146967900, 146976330, 1469763600,
            146976390, 146976420, 146976540, 146984850, 146984910, 1469849400,
            146985030, 146985150, 146985180, 146993460, 146993550, 1469936100,
            146993640, 146993670, 146993700, 146993730, 146993850, 1470014700,
            147001710, 147001770, 147001830, 147001890, 147001920, 1470020400,
            147002070, 147002100, 147002130, 147002160, 147002190, 1470022500,
            147002280, 147002340, 147002370, 147002400, 147002430, 1470024600,
            147002490, 147002610, 147002940, 147002970, 147003000, 1470031800,
            147003210, 147003300, 147003330, 147003420, 1470036900]

    interval_num, interval_str = get_constant_timestamp(time, 30)
    print(interval_num, interval_str)


def test_merge():
    train_timestamp_sorted = np.asarray([1, 2])
    test_timestamp_sorted = np.asarray([3, 4, 5])
    merge_set = set(train_timestamp_sorted).intersection(set(test_timestamp_sorted))
    print(len(merge_set))


def test_union():
    train_timestamp_sorted = np.asarray([1, 2, 3])
    test_timestamp_sorted = np.asarray([3, 4, 5])
    union_list = list(set(train_timestamp_sorted).union(set(test_timestamp_sorted)))
    print(union_list)


def test_sqrt():
    print(np.sqrt(3))


def test_sub():
    zero_num = 1
    train_timestamp_sorted = np.asarray([1, 2, 3])
    print(train_timestamp_sorted[zero_num:np.size(train_timestamp_sorted)])


def test_sort():
    lis = []
    catch = {"score": 1, "num": 2, "index": 3, "f": 4}
    lis.append(catch)
    catch = {"score": 4, "num": 3, "index": 3, "f": 1}
    lis.append(catch)
    catch = {"score": 2, "num": 3, "index": 3, "f": 3}
    lis.append(catch)
    lis = sorted(lis, key=lambda dict_catch: (dict_catch['f'], dict_catch['score']))
    print(lis)


def test_in():
    arr=[1,2,3,4]
    print(1 in arr )