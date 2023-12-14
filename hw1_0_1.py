# -*- coding: utf-8 -*-

import time
import numpy as np
from tqdm import tqdm


def match_timestamps_fastest_version(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    """
    Timestamp matching function. It returns such array `matching` of length len(timestamps1),
    that for each index i of timestamps1 the output element matching[i] contains
    the index j of timestamps2, so that the difference between
    timestamps2[j] and timestamps1[i] is minimal.
    Example:
        timestamps1 = [0, 0.091, 0.5]
        timestamps2 = [0.001, 0.09, 0.12, 0.6]
        => matching = [0, 1, 3]
    """
    matching = []
    t2_point = 0

    for t1_point in timestamps1:

        while t2_point < len(timestamps2):
            if np.abs(t1_point - timestamps2[t2_point]) <= np.abs(t1_point - timestamps2[t2_point + 1]):
                matching.append(t2_point)
                break
            else:
                t2_point += 1

    return np.array(matching)


def make_timestamps(fps: int, st_ts: float, fn_ts: float) -> np.ndarray:
    """
    Create array of timestamps. This array is discretized with fps,
    but not evenly.
    Timestamps are assumed sorted nad unique.
    Parameters:
    - fps: int
        Average frame per second
    - st_ts: float
        First timestamp in the sequence
    - fn_ts: float
        Last timestamp in the sequence
    Returns:
        np.ndarray: synthetic timestamps
    """
    # generate uniform timestamps
    timestamps = np.linspace(st_ts, fn_ts, int((fn_ts - st_ts) * fps))
    # add an fps noise
    timestamps += np.random.randn(len(timestamps))
    timestamps = np.unique(np.sort(timestamps))
    return timestamps


def binary_search_closest(array, target):
    """
    Find the index of the closest value to the target in a sorted array using binary search.
    """
    low, high = 0, len(array) - 1
    best_index = low
    while low <= high:
        mid = (low + high) // 2
        if array[mid] < target:
            low = mid + 1
        elif array[mid] > target:
            high = mid - 1
        else:
            return mid
        # Update the best_index if this element is closer to the target
        if abs(array[mid] - target) < abs(array[best_index] - target):
            best_index = mid
    return best_index


def match_timestamps_binary_search(timestamps1: np.ndarray, timestamps2: np.ndarray) -> np.ndarray:
    """
    Optimized timestamp matching function using binary search.
    """
    # Ensure timestamps2 is sorted for binary search
    timestamps2_sorted = np.sort(timestamps2)
    return np.array([binary_search_closest(timestamps2_sorted, ts) for ts in timestamps1])


def main():
    """
    Setup:
        Say we have two cameras, each filming the same scene. We make
        a prediction based on this scene (e.g. detect a human pose).
        To improve the robustness of the detection algorithm,
        we average the predictions from both cameras at each moment.
        The camera data is a pair (frame, timestamp), where the timestamp
        represents the moment when the frame was captured by the camera.
        The structure of the prediction does not matter here.

    Problem:
        For each frame of camera1, we need to find the index of the
        corresponding frame received by camera2. The frame i from camera2
        corresponds to the frame j from camera1, if
        abs(timestamps[i] - timestamps[j]) is minimal for all i.

    Estimation criteria:
        - The solution has to be optimal algorithmically. If the
    best solution turns out to have the O(n^3) complexity [just an example],
    the solution with O(n^3 * logn) will have -1 point,
    the solution O(n^4) will have -2 points and so on.
    Make sure your algorithm cannot be optimized!
        - The solution has to be optimal python-wise.
    If it can be optimized ~x5 times by rewriting the algorithm in Python,
    this will be a -1 point. x20 times optimization will give -2 points, and so on.
    You may use any library, even write your own
    one in C++.
        - All corner cases must be handled correctly. A wrong solution
    will have -3 points.
        - Top 3 solutions get 10 points. The measurement will be done in a single thread.
        - The base score is 9.
        - Shipping the solution in a Docker container results in +1 point.
    Such solution must contain a Dockerfile, which later will be built
    via `docker build ...`, and the hw1.py script will be called from this container.
    Try making this container as small as possible in Mb!
        - Parallel implementation adds +1 point, provided it is effective
    (cannot be optimized x5 times)
        - Maximal score is 10 points, minimal score is 5 points.
        - The deadline is November 21 23:59. Failing the deadline will
    result in -2 points, and each additional week will result in -1 point.
        - The solution can be improved/fixed after the deadline provided that the initial
    version is submitted on time.

    Optimize the solution to work with ~2-3 hours of data.
    Good luck!
    """
    numb_iter = 100
    time_fast = 0
    time_binary = 0

    for iter in tqdm(range(numb_iter)):
        # generate timestamps for the first camera
        timestamps1 = make_timestamps(
            30, time.time() - 100, time.time() + 3600 * 2)
        # generate timestamps for the second camera
        timestamps2 = make_timestamps(
            60, time.time() + 200, time.time() + 3600 * 2.5)

        start = time.time()
        matching = match_timestamps_fastest_version(timestamps1, timestamps2)
        time_fast += time.time() - start

    print('average time of fastest approach:', time_fast / numb_iter)

    for iter in tqdm(range(numb_iter)):
        # generate timestamps for the first camera
        timestamps1 = make_timestamps(
            30, time.time() - 100, time.time() + 3600 * 2)
        # generate timestamps for the second camera
        timestamps2 = make_timestamps(
            60, time.time() + 200, time.time() + 3600 * 2.5)

        start = time.time()
        matching = match_timestamps_binary_search(timestamps1, timestamps2)
        time_binary += time.time() - start

    print('average time of binary search approach:', time_binary / numb_iter)


if __name__ == '__main__':
    main()
