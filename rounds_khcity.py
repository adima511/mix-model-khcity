import random as rnd
import numpy as np
import time
from math import acos, pi

from shapely.geometry.point import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union
from matplotlib import pyplot as plt
from descartes import PolygonPatch

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

from scipy.optimize import minimize


def generate_begin_point(cover_area, n):
    boundaries = cover_area.bounds
    coords = []
    while len(coords) < n * 2:
        point = Point([round(rnd.triangular(boundaries[0], boundaries[2]), 2),
                       round(rnd.triangular(boundaries[1], boundaries[3]), 2)])
        if cover_area.contains(point):
            coords.extend([point.x, point.y])
    return coords


def create_round(center, radius):
    return Point(center).buffer(radius)


def rounds_init(coords, radiuses):
    n = len(radiuses)
    rounds = []
    for i in range(n):
        rounds.append(create_round((coords[i * 2], coords[i * 2 + 1]),
                                   radiuses[i]))
    return rounds


def show_plot(x, radiuses, cover_area):
    rounds = rounds_init(x, radiuses)

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(radiuses)):
         patch = PolygonPatch(rounds[i], alpha=0.5, zorder=2)
         ax.add_patch(patch)
    patch = PolygonPatch(cover_area, fc='r', ec='r', alpha=0.3, zorder=3)
    ax.add_patch(patch)
    ax.plot(x[::2], x[1::2], 'go', ms=3)
    ax.set_aspect('equal', adjustable='box')
    plt.axis([cover_area.bounds[0], cover_area.bounds[2], cover_area.bounds[1], cover_area.bounds[3]])
    plt.grid(linestyle='--')
    plt.show()


def pair_rounds_intersection(x1, y1, x2, y2, r1, r2):
    d = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    if d <= abs(r1 - r2):
        return 3.141592653589793 * min(r1, r2) ** 2
    elif d >= r1 + r2:
        return 0
    else:
        d1 = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
        d2 = d - d1
        return r1 ** 2 * acos(d1 / r1) - d1 * (r1 ** 2 - d1 ** 2) ** 0.5 + \
               r2 ** 2 * acos(d2 / r2) - d2 * (r2 ** 2 - d2 ** 2) ** 0.5



def calc_intersection(x, cover_area, radiuses):
    n = len(x) // 2
    rounds = rounds_init(x, radiuses)
    overlap_area_map = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    # область
    overlap_area_map[0][0] = cover_area.area
    for i in range(n):
        overlap_area_map[0][i + 1] = overlap_area_map[i + 1][0] = cover_area.intersection(rounds[i]).area

    # круги
    for i in range(n):
        overlap_area_map[i + 1][i + 1] = rounds[i].area

    for i in range(n - 1):
        for j in range(i + 1, n):
            overlap_area_map[i + 1][j + 1] = \
                overlap_area_map[j + 1][i + 1] = pair_rounds_intersection(x[i * 2], x[i * 2 + 1],
                                                                          x[j * 2], x[j * 2 + 1],
                                                                          radiuses[i], radiuses[j])

    return overlap_area_map


def fun_intersection(x, cover_area, radiuses, alpha):
    n = len(x) // 2

    sum_area = 0
    sum_objs_area = 0
    sum_objs_intern_area = 0
    M_matr = calc_intersection(x, cover_area, radiuses)
    for i in range(1, n + 1):
        sum_objs_area += M_matr[i][i]
        sum_objs_intern_area += M_matr[i][0]

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            sum_area += M_matr[i][j]

    return sum_area + alpha * (sum_objs_area - sum_objs_intern_area)


def calc_jac_intersection(x, cover_area, radiuses, alpha, dx=0.0001):
    n = len(x) // 2
    overlap_area_map = calc_intersection(x, cover_area, radiuses)
    rounds = rounds_init(x, radiuses)
    jac = [0 for _ in range(n * 2)]

    for i in range(n):
        for j in range(2):
            diff = 0
            x[i * 2 + j] += dx
            diff_round = create_round((x[i * 2], x[i * 2 + 1]), radiuses[i])
            if overlap_area_map[i + 1][0] < overlap_area_map[i + 1][i + 1]:
                diff += -alpha * (diff_round.intersection(cover_area).area - overlap_area_map[i + 1][0])
            for k in range(n):
                if overlap_area_map[i + 1][k + 1] > 0 and k != i:
                    diff_area = pair_rounds_intersection(x[i * 2], x[i * 2 + 1], x[k * 2], x[k * 2 + 1],
                                                         radiuses[i], radiuses[k])
                    diff += diff_area - overlap_area_map[i + 1][k + 1]
            x[i * 2 + j] -= dx
            jac[i * 2 + j] = diff / dx
    return jac


def fun_cover(x, cover_area, radiuses):
    rounds = rounds_init(x, radiuses)
    return -cover_area.intersection(unary_union(rounds)).area


def calc_jac_cover(x, cover_area, radiuses, dx=0.001):
    n = len(x) // 2
    overlap_area_map = calc_intersection(x, cover_area, radiuses)
    rounds = rounds_init(x, radiuses)
    jac = [0 for _ in range(n * 2)]

    for i in range(n):
        for j in range(2):
            diff = 0
            x[i * 2 + j] += dx
            diff_round = create_round((x[i * 2], x[i * 2 + 1]), radiuses[i])
            if overlap_area_map[i + 1][0] < overlap_area_map[i + 1][i + 1]:
                diff += (diff_round.area - diff_round.intersection(cover_area).area) - \
                                  (rounds[i].area - overlap_area_map[i + 1][0])
            for k in range(n):
                if overlap_area_map[i + 1][k + 1] > 0 and k != i:
                    diff_area = pair_rounds_intersection(x[i * 2], x[i * 2 + 1], x[k * 2], x[k * 2 + 1],
                                                         radiuses[i], radiuses[k])
                    diff += diff_area - overlap_area_map[i + 1][k + 1]
            x[i * 2 + j] -= dx
            jac[i * 2 + j] = diff / dx
    return jac


cover_area_coords = [(226, 313), (239, 294), (228,289), (223,268), (219,269), (217,242), (224,242), (224,230),
                     (242, 232), (236, 199), (254, 202), (258, 192), (264, 192), (267, 182), (272, 179), (269, 176),
                     (270, 170), (283, 169), (276, 160), (274, 139), (260, 130), (267, 125), (258, 112), (265, 115),
                     (275, 114), (281, 108), (293, 109), (297, 115), (303, 111), (291, 96), (285, 93), (273, 97),
                     (264, 64), (269, 62), (284, 83), (281, 88), (290, 88), (307, 72), (316, 57), (330, 44), (355, 34),
                     (366, 23), (374, 20), (385,22), (389, 29), (381, 50), (397, 109), (405, 107), (409, 103),
                     (409, 93), (412, 92), (413, 84), (420, 82), (417, 74), (427, 71), (437, 87), (438, 95), (447, 93),
                     (455, 88), (460, 90), (442, 101), (446, 107), (458, 101), (470, 102), (475, 88), (479, 88),
                     (479, 97), (488, 89), (491, 91), (479, 104), (485, 107), (493, 96), (496, 97), (488, 110),
                     (508, 124), (502, 134), (499, 134), (505, 142), (503, 145), (513, 153), (489, 175), (500, 182),
                     (490, 190), (502, 201), (509, 197), (520, 216), (523, 230), (505, 234), (509, 251), (560, 279),
                     (556, 294), (590, 315), (594, 323), (589, 334), (578, 333), (573, 327), (558, 334), (567, 350),
                     (557, 356), (554, 350), (541, 357), (538, 355), (547, 335), (509, 318), (506, 321), (496, 314),
                     (501, 306), (462, 294), (392, 308), (366, 342), (341, 345), (335, 352), (326, 352), (317, 333),
                     (306, 321), (290, 319), (284, 291), (260, 298), (256, 314), (236, 321)]


for i in range(len(cover_area_coords)):
    cover_area_coords[i] = (cover_area_coords[i][0] - 400, 150 - cover_area_coords[i][1])

cover_area = Polygon(cover_area_coords)

alpha = 1
n = 30
object_area = cover_area.area / n
radi = (object_area / pi) ** 0.5
radiuses = []
for i in range(-n // 2, n // 2):
    radiuses.append(round(radi + i / 1.5, 1))

print(f'radiuses: {radiuses}')


best_function_value = float("inf")
best_point = []

time_begin = time.time()
for i in range(10):

    coords = generate_begin_point(cover_area, n)
    result_intersection = minimize(fun_intersection, np.array(coords), args=(cover_area, radiuses, alpha),
                                   method='bfgs', jac=calc_jac_intersection, options={'maxiter': 1000, 'gtol': 1e-1})
    show_plot(result_intersection['x'], radiuses, cover_area)
    if fun_cover(result_intersection['x'], cover_area, radiuses) < best_function_value:
        best_function_value = fun_cover(result_intersection['x'], cover_area, radiuses)
        best_point = result_intersection["x"]
        print(f'intersection point: {best_point}')
        print(f'interseciton cover value: {fun_cover(best_point, cover_area, radiuses)}')

print(f'time for multistart: {time.time() - time_begin} sec')
print(f'best intersection point: {best_point}')
print(f'best interseciton cover value: {fun_cover(best_point, cover_area, radiuses)}')


# show_plot(coords, radiuses, cover_area)
show_plot(best_point, radiuses, cover_area)

time_begin = time.time()
result_cover = minimize(fun_cover, np.array(best_point), args=(cover_area, radiuses), method='bfgs',
                        jac=calc_jac_cover, options={'maxiter': 1000, 'gtol': 1e-1})
print(f'time for better: {time.time() - time_begin} sec')

print(f'final result: {result_cover["x"]}')
print(f'final cover value: {fun_cover(result_cover["x"], cover_area, radiuses)}')
show_plot(result_cover['x'], radiuses, cover_area)
