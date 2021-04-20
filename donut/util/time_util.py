import time

import numpy as np


class TimeUse:
    def __init__(self, use, name):
        self.use = use
        self.name = name

    def __str__(self):
        return self.name + "【用时：{}】".format(self.use)

    def __eq__(self, other):
        return self.use == other.use

    def __lt__(self, other):
        return self.use < other.use

    def __gt__(self, other):
        return self.use > other.use

    def __cmp__(self, other):
        if self.use < other.use:
            return -1
        elif self.use == other.use:
            return 0
        else:
            return 1


class TimeCounter(object):
    def __init__(self):
        self._start = time.time()
        self._end = time.time()

    def start(self):
        self._start = time.time()

    def end(self):
        self._end = time.time()

    def get(self):
        return self._end - self._start

    def get_s(self):
        return str(self.get())


def format_time(atime):
    """
    格式化时间
    Args:
        atime: 时间戳

    Returns:
        格式化后的时间

    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(atime))


def get_constant_timestamp(timestamps, step):
    """
    获得连续的时间戳区间字符串
    Args:
        timestamps: 时间戳
        step: 时间戳步长
    Returns:
        时间戳区间字符串

    """
    if np.size(timestamps) == 0:
        return 0, None
    else:
        timestamps = np.sort(timestamps)
        # print(timestamps)
        has_dot = False
        has_head = False
        time_str = "其中时间戳分布为\n\t"
        last = 0
        interval_num = 0
        start = 0
        count = 0
        for i, t in enumerate(timestamps):
            if not has_head:
                time_str = time_str + str(t)
                has_head = True
                last = t
                start = t
            else:
                if int(t) == (int(last) + int(step)):
                    if not has_dot:
                        time_str = time_str + "..."
                        has_dot = True
                        interval_num = interval_num + 1
                    last = t
                else:
                    if has_dot and last != start:
                        time_str = time_str + str(last) + "、"
                    else:
                        time_str = time_str + "、"
                    count = count + 1
                    last = t
                    has_head = False
                    has_dot = False
        return interval_num, time_str
