import random as rnd
import numpy as np
import time

from shapely.geometry.point import Point
from shapely.geometry import Polygon
from shapely import affinity
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
    while len(coords) < n * 3:
        point = Point([round(rnd.triangular(boundaries[0], boundaries[2]), 2),
                       round(rnd.triangular(boundaries[1], boundaries[3]), 2)])
        if cover_area.contains(point):
            coords.extend([point.x, point.y, round(rnd.randrange(-9, 9), 2)])
    return coords


def create_ellipse(center, semiaxes, angle):
    circ = Point(center).buffer(1)
    ellipse = affinity.scale(circ, semiaxes[0], semiaxes[1])
    ellipse_ready = affinity.rotate(ellipse, angle * 10)
    return ellipse_ready


def ellipses_init(coords, semiaxes):
    n = len(semiaxes) // 2
    ellipses = []
    for i in range(n):
        ellipses.append(create_ellipse((coords[i * 3], coords[i * 3 + 1]),
                                       (semiaxes[i * 2], semiaxes[i * 2 + 1]),
                                       coords[i * 3 + 2]))
    return ellipses


def show_plot(x, semiaxes, cover_area):
    ellipses = ellipses_init(x, semiaxes)

    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(ellipses)):
         patch = PolygonPatch(ellipses[i], alpha=0.5, zorder=2)
         ax.add_patch(patch)
    patch = PolygonPatch(cover_area, fc='r', ec='r', alpha=0.3, zorder=3)
    ax.add_patch(patch)
    ax.plot(x[::3], x[1::3], 'go', ms=3)
    ax.set_aspect('equal', adjustable='box')
    plt.axis([cover_area.bounds[0], cover_area.bounds[2], cover_area.bounds[1], cover_area.bounds[3]])
    plt.grid(linestyle='--')
    plt.show()


def calc_intersection(x, cover_area, semiaxes):
    n = len(x) // 3
    ellipses = ellipses_init(x, semiaxes)
    overlap_area_map = [[0 for _ in range(n + 1)] for _ in range(n + 1)]

    # область
    overlap_area_map[0][0] = cover_area.area
    for i in range(n):
        overlap_area_map[0][i + 1] = overlap_area_map[i + 1][0] = cover_area.intersection(ellipses[i]).area

    # эллипсы
    for i in range(n):
        overlap_area_map[i + 1][i + 1] = ellipses[i].area

    for i in range(n - 1):
        for j in range(i + 1, n):
            overlap_area_map[i + 1][j + 1] = overlap_area_map[j + 1][i + 1] = ellipses[i].intersection(ellipses[j]).area

    return overlap_area_map


def fun_intersection(x, cover_area, semiaxes, alpha):
    n = len(x) // 3

    sum_area = 0
    sum_objs_area = 0
    sum_objs_intern_area = 0
    M_matr = calc_intersection(x, cover_area, semiaxes)
    for i in range(1, n + 1):
        sum_objs_area += M_matr[i][i]
        sum_objs_intern_area += M_matr[i][0]

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            sum_area += M_matr[i][j]

    return sum_area + alpha * (sum_objs_area - sum_objs_intern_area)


def calc_jac_intersection(x, cover_area, semiaxes, alpha, dx=0.0001):
    n = len(x) // 3
    overlap_area_map = calc_intersection(x, cover_area, semiaxes)
    ellipses = ellipses_init(x, semiaxes)
    jac = [0 for _ in range(n * 3)]

    for i in range(n):
        for j in range(3):
            diff = 0
            x[i * 3 + j] += dx
            diff_ellipse = create_ellipse((x[i * 3], x[i * 3 + 1]),
                                          (semiaxes[i * 2], semiaxes[i * 2 + 1]), x[i * 3 + 2])
            if overlap_area_map[i + 1][0] < overlap_area_map[i + 1][i + 1]:
                diff += -alpha * (diff_ellipse.intersection(cover_area).area - overlap_area_map[i + 1][0])
            for k in range(n):
                if overlap_area_map[i + 1][k + 1] > 0 and k != i:
                    diff += diff_ellipse.intersection(ellipses[k]).area - overlap_area_map[i + 1][k + 1]
            x[i * 3 + j] -= dx
            jac[i * 3 + j] = diff / dx
    return jac


def fun_cover(x, cover_area, semiaxes):
    ellipses = ellipses_init(x, semiaxes)
    return -cover_area.intersection(unary_union(ellipses)).area


def calc_jac_cover(x, cover_area, semiaxes, dx=0.0001):
    n = len(x) // 3
    overlap_area_map = calc_intersection(x, cover_area, semiaxes)
    ellipses = ellipses_init(x, semiaxes)
    jac = [0 for _ in range(n * 3)]

    for i in range(n):
        for j in range(3):
            diff = 0
            x[i * 3 + j] += dx
            diff_ellipse = create_ellipse((x[i * 3], x[i * 3 + 1]),
                                          (semiaxes[i * 2], semiaxes[i * 2 + 1]), x[i * 3 + 2])
            if overlap_area_map[i + 1][0] < overlap_area_map[i + 1][i + 1]:
                diff += (diff_ellipse.area - diff_ellipse.intersection(cover_area).area) - \
                                  (ellipses[i].area - overlap_area_map[i + 1][0])
            for k in range(n):
                if overlap_area_map[i + 1][k + 1] > 0 and k != i:
                    diff += diff_ellipse.intersection(ellipses[k]).area - overlap_area_map[i + 1][k + 1]
            x[i * 3 + j] -= dx
            jac[i * 3 + j] = diff / dx
    return jac


cover_area_coords = [(226, 313), (239, 294), (228, 289), (223, 268), (219, 269), (217, 242), (224, 242), (224, 230),
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

n = 30
alpha = 1

semiaxes = [20, 35, 20, 27, 18, 22, 37, 18, 33, 30, 25, 31, 25, 33, 23, 37, 21, 30, 29, 32, 30, 24, 29, 20, 35, 20,
            18, 20, 37, 16, 31, 18, 28, 34, 29, 22, 27, 34, 22, 36, 16, 35, 37, 18, 24, 37, 24, 17, 19, 34, 35, 21,
            25, 26, 36, 26, 17, 36, 24, 36]


ell_sum_area = 0

print(f'semiaxes: {semiaxes}')


best_function_value = float("inf")
best_point = []

time_begin = time.time()

for i in range(10):

    coords = generate_begin_point(cover_area, n)
    result_intersection = minimize(fun_intersection, np.array(coords), args=(cover_area, semiaxes, alpha),
                                   method='bfgs', jac=calc_jac_intersection, options={'maxiter': 1000, 'gtol': 1e-1})

    show_plot(result_intersection['x'], semiaxes, cover_area)
    print(result_intersection["fun"])
    if fun_cover(result_intersection['x'], cover_area, semiaxes) < best_function_value:
        best_function_value = fun_cover(result_intersection['x'], cover_area, semiaxes)
        best_point = result_intersection["x"]
        print(f'intersection point: {best_point}')
        print(f'interseciton cover value: {fun_cover(best_point, cover_area, semiaxes)}')

print(f'time for multistart: {time.time() - time_begin} sec')
print(f'best intersection point: {best_point}')
print(f'best interseciton cover value: {fun_cover(best_point, cover_area, semiaxes)}')

show_plot(best_point, semiaxes, cover_area)

time_begin = time.time()
result_cover = minimize(fun_cover, np.array(best_point), args=(cover_area, semiaxes), method='bfgs',
                        jac=calc_jac_cover, options={'maxiter': 1000, 'gtol': 1e-1})
print(f'time for better: {time.time() - time_begin} sec')

print(f'final result: {result_cover["x"]}')
print(f'final cover value: {fun_cover(result_cover["x"], cover_area, semiaxes)}')
show_plot(result_cover['x'], semiaxes, cover_area)
