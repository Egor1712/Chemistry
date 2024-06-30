import numpy as np


def sort_by_i(i, first, second):
    if first[i] > second[i]:
        return second, first
    return first, second


def close_curve(curve):
    curve = list(curve)
    first, last = curve[0], curve[-1]

    if first[0] != last[0]:
        curve.append([last[0], 0])
        curve.append([first[0], 0])

    curve.append(first)
    return curve


# Draws a phase along a contour from a file
def draw_phase(file, size, x_max, y_max):
    width = size[0]
    height = size[1]
    phase = np.zeros(size, dtype=int)
    curves = [[]]

    with open(file) as file:
        for line in file.readlines()[6:-1]:
            point = line.strip().split()[:2]
            if not point:
                curves.append([])
            else:
                curves[-1].append(list(map(float, point)))

    for curve in curves:
        if not curve:
            continue
        curve = close_curve(curve)
        for i in range(height):
            y = (i + 0.5) / size[1] * y_max
            intersections = []
            for j in range(len(curve) - 1):
                first = curve[j]
                second = curve[j + 1]
                if first[1] > second[1]:
                    first, second = second, first
                if first[1] <= y <= second[1]:
                    k = (first[0] - second[0]) / (first[1] - second[1])
                    intersections.append(first[0] + k * (y - first[1]))

            intersections.sort()

            for j in range(len(intersections) // 2):
                x_1 = int(intersections[j * 2] / x_max * width)
                x_2 = int(intersections[j * 2 + 1] / x_max * width)
                if x_1 > x_2:
                    x_1, x_2 = x_2, x_1

                for x in range(x_1, min(x_2 + 1, width)):
                    phase[x][i] = 1

    return phase
