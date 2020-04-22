import random
import numpy as np
import sklearn.linear_model as lm
from scipy.stats import f, t
from prettytable import PrettyTable
import math


# Обчислення рівняння регрестї
def regression(x, b):
    y = sum([x[i] * b[i] for i in range(len(x))])
    return y


# Повертає масив з дисперсіями по рядках
def dispersion(y, y_aver, n, m):
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
        res.append(round(s, 3))
    return res


# Повертає матрицю з натуралізованими значеннями факторів, матрицю значень експерементів,
# матрицю з нормалізованими значеннями факторів
def planing_matrix(n, m, interaction, quadratic_terms):
    x_normalized = [[1, -1, -1, -1],
                    [1, -1, 1, 1],
                    [1, 1, -1, 1],
                    [1, 1, 1, -1],
                    [1, -1, -1, 1],
                    [1, -1, 1, -1],
                    [1, 1, -1, -1],
                    [1, 1, 1, 1]]

    # Створює матрицю n на m заповнену нулями
    y = np.zeros(shape=(n, m))
    # Заповнює матрицю планування випадковим чином
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(y_min, y_max)
    # print(y)
    # З yрахуванням ефекту взаємодії
    if interaction:
        for x in x_normalized:
            x.append(x[1] * x[2])
            x.append(x[1] * x[3])
            x.append(x[2] * x[3])
            x.append(x[1] * x[2] * x[3])

    if quadratic_terms and interaction:
        for row in x_normalized:
            for i in range(0, 3):
                row.append(1)

        l = 1.215
        for i in range(0, 3):
            row1 = [1]
            row2 = [1]
            for n in range(0, i):
                row1.append(0)
                row2.append(0)
            row1.append(-l)
            row2.append(l)
            for _ in range(0, 6):
                row1.append(0)
                row2.append(0)
            row1.append(round(l*l, 3))
            row2.append(round(l*l, 3))
            temp = 2 - i
            for _ in range(0, temp):
                row1.append(0)
                row2.append(0)
            x_normalized.append(row1)
            x_normalized.append(row2)
        row15 = []
        for _ in range(0, 11):
            row15.append(0)
        x_normalized.append(row15)


    # print(x_normalized)

    # Визначає кількість рядків матриці планування
    x_normalized = np.array(x_normalized[:len(y)])
    # Створює матрицю заповнену одиницями
    x = np.ones(shape=(len(x_normalized), len(x_normalized[0])))

    # Заповнює матрицю з натуралізованими значеннями факторів
    for i in range(len(x_normalized)):
        for j in range(1, 4):
            if x_normalized[i][j] == -1:
                x[i][j] = x_range[j - 1][0]
            else:
                x[i][j] = x_range[j - 1][1]
    # print(x)
    if quadratic_terms and interaction:
        x[8] = [1,-l * delta_x(0) + x_nul(0), x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[9] = [1, l * delta_x(0) + x_nul(0), x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[10] = [1, x_nul(0), -l * delta_x(1) + x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[11] = [1, x_nul(0),  l * delta_x(1) + x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[12] = [1, x_nul(0), x_nul(1), -l * delta_x(2) + x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[13] = [1, x_nul(0), x_nul(1),  l * delta_x(2) + x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[14] = [1, x_nul(0), x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]

        for i in range(8, 15):
            for j in range(0, 11):
                x[i][j] = round(x[i][j], 3)

    # З yрахуванням ефекту взаємодії
    if interaction:
        for i in range(len(x)):
            x[i][4] = round(x[i][1] * x[i][2], 3)
            x[i][5] = round(x[i][1] * x[i][3], 3)
            x[i][6] = round(x[i][2] * x[i][3], 3)
            x[i][7] = round(x[i][1] * x[i][3] * x[i][2], 3)
            x[i][8] = round(x[i][1] * x[i][1], 3)
            x[i][9] = round(x[i][2] * x[i][2], 3)
            x[i][10] = round(x[i][3] * x[i][3], 3)


    if interaction:
        # З yрахуванням ефекту взаємодії
        print(f'\nМатриця планування для n = {n}, m = {m}')

        print('\nЗ кодованими значеннями факторів:')
        caption = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3","X1^2", "X2^2", "X3^2", "Y1", "Y2", "Y3"]
        rows_kod = np.concatenate((x, y), axis=1)
        print_table(caption, rows_kod)

        print('\nЗ нормованими значеннями факторів:\n')
        rows_norm = np.concatenate((x_normalized, y), axis=1)
        print_table(caption, rows_norm)
    else:
        print('\nМатриця планування:')
        caption = ["X0", "X1", "X2", "X3", "Y1", "Y2", "Y3"]
        rows = np.concatenate((x, y), axis=1)
        print_table(caption, rows)

    return x, y, x_normalized


def x_nul(n):
    return (x_range[n][0] + x_range[n][1]) / 2


def delta_x(n):
    return x_nul(n) - x_range[n][0]


def print_table(caption, values):
    table = PrettyTable()
    table.field_names = caption

    for row in values:
        table.add_row(row)
    print(table)


def find_coef(X, Y, norm=False):
    skm = lm.LinearRegression(fit_intercept=False)
    skm.fit(X, Y)
    B = skm.coef_

    if norm == 1:
        print('\nКоефіцієнти рівняння регресії з нормованими X:')
    else:
        print('\nКоефіцієнти рівняння регресії:')
    B = [round(i, 3) for i in B]
    print(B)
    return B


def s_kv(y, y_aver, n, m):
    """
    Функція для знаходження квадратної дисперсії
    """
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j])**2 for j in range(m)]) / m
        res.append(s)
    return res


def kriteriy_fishera(y, y_aver, y_new, n, m, d):
    """
    Функція для знаходження критерія фішера
    """
    S_kv_ad = (m / (n - d)) * sum([(y_new[i] - y_aver[i])**2 for i in range(len(y))])
    S_kv_b = s_kv(y, y_aver, n, m)
    S_kv_b_aver = sum(S_kv_b) / n

    return S_kv_ad / S_kv_b_aver

def check(n, m, interaction, quadratic_terms):

    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05

    x, y, x_norm = planing_matrix(n, m, interaction, quadratic_terms)

    if interaction:
        y_average = [round(sum(i) / len(i), 3) for i in y]
        B = find_coef(x_norm, y_average, norm=interaction)
    else:
        y_average = [round(sum(i) / len(i), 2) for i in y]
        B = find_coef(x, y_average, norm=interaction)

    print('\nСереднє значення y:', y_average)

    dispersion_arr = dispersion(y, y_average, n, m)

    # Табличне значення критерія Кохрена
    temp_cohren = f.ppf(q=(1 - q / f1), dfn=f2, dfd=(f1 - 1) * f2)
    cohren_cr_table = temp_cohren / (temp_cohren + f1 - 1)
    # Розрахункове значення критерія Кохрена
    Gp = max(dispersion_arr) / sum(dispersion_arr)

    print('\nПеревірка за критерієм Кохрена:\n')
    print(f'Розрахункове значення: Gp = {Gp}'
          f'\nТабличне значення: Gt = {cohren_cr_table}')
    if Gp < cohren_cr_table:
        print(f'З ймовірністю {1-q} дисперсії однорідні.')
    else:
        print("Необхідно збільшити ксть дослідів")
        m += 1
        check(n, m, interaction)

    # Табличне значення критерія Стюдента
    qq = (1 + 0.95) / 2
    student_cr_table = t.ppf(df=f3, q=qq)
    # Розрахункове значення критерія Стюдента

    Dispersion_B = sum(dispersion_arr) / n
    Dispersion_beta = Dispersion_B / (m * n)
    S_beta = math.sqrt(abs(Dispersion_beta))

    student_t = []
    for i in range(len(B)):
        student_t.append(round(abs(B[i]) / S_beta, 3))

    print('\nТабличне значення критерій Стьюдента:\n', student_cr_table)
    print('Розрахункове значення критерій Стьюдента:\n', student_t)
    res_student_t = [temp for temp in student_t if temp > student_cr_table]
    final_coefficients = [B[i] for i in range(len(student_t)) if student_t[i] in res_student_t]
    print('\nКоефіцієнти {} статистично незначущі.'.format(
        [round(i, 3) for i in B if i not in final_coefficients]))

    y_new = []
    if interaction:
        for j in range(n):
            y_new.append(regression([x[j][i] for i in range(len(student_t)) if student_t[i] in res_student_t], final_coefficients))
    else:
        for j in range(n):
            y_new.append(regression([x[j][student_t.index(i)] for i in student_t if i in res_student_t], final_coefficients))

    print(f'\nЗначення рівння регресії з коефіцієнтами {final_coefficients}: ')
    for i in range(len(y_new)):
        y_new[i] = round(y_new[i], 3)
    print(y_new)
    d = len(res_student_t)

    if d >= n:
        print('\nF4 <= 0')
        print('')
        return
    f4 = n - d

    # Розрахункове значення критерія Фішера
    Fp = kriteriy_fishera(y, y_average, y_new, n, m, d)
    # Табличне значення критерія Фішера
    Ft = f.ppf(dfn=f4, dfd=f3, q=1 - 0.05)

    print('\nПеревірка адекватності за критерієм Фішера:\n')
    print('Розрахункове значення критерія Фішера: Fp =', Fp)
    print('Табличне значення критерія Фішера: Ft =', Ft)
    if Fp < Ft:
        print('Математична модель адекватна експериментальним даним')
        return True
    else:
        print('Математична модель не адекватна експериментальним даним')
        return False


def main(n, m):
    if not check(n, m, False, False):
        if not check(15, m, True, True):
            main(n, m)


if __name__ == '__main__':
    # Значення за варіантом
    x_range = ((-7, 4), (-6, 10), (-8, 1))

    y_max = 200 + int(sum([x[1] for x in x_range]) / 3)
    y_min = 200 + int(sum([x[0] for x in x_range]) / 3)

    main(8, 3)
