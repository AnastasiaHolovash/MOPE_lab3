import random
import numpy as np
import sklearn.linear_model as lm
from scipy.stats import f, t
from numpy.linalg import solve


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
def planing_matrix(n, m, interaction):
    x_normalized = [[1, -1, -1, -1],
                            [1, -1, 1, 1],
                            [1, 1, -1, 1],
                            [1, 1, 1, -1],
                            [1, -1, -1, 1],
                            [1, -1, 1, -1],
                            [1, 1, -1, -1],
                            [1, 1, 1, 1]]

    # Створює матрицю n на m заповнену нулями
    y = np.zeros(shape=(n, m), dtype=np.int64)
    # Заповнює матрицю планування випадковим чином
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(y_min, y_max)

    # З yрахуванням ефекту взаємодії
    if interaction:
        for x in x_normalized:
            x.append(x[1] * x[2])
            x.append(x[1] * x[3])
            x.append(x[2] * x[3])
            x.append(x[1] * x[2] * x[3])

    # Визначає кількість рядків матриці планування
    x_normalized = np.array(x_normalized[:len(y)])
    # Створює матрицю заповнену одиницями
    x = np.ones(shape=(len(x_normalized), len(x_normalized[0])), dtype=np.int64)

    # Заповнює матрицю з натуралізованими значеннями факторів
    for i in range(len(x_normalized)):
        for j in range(1, 4):
            if x_normalized[i][j] == -1:
                x[i][j] = x_range[j - 1][0]
            else:
                x[i][j] = x_range[j - 1][1]

    # З yрахуванням ефекту взаємодії
    if interaction:
        for i in range(len(x)):
            x[i][4] = x[i][1] * x[i][2]
            x[i][5] = x[i][1] * x[i][3]
            x[i][6] = x[i][2] * x[i][3]
            x[i][7] = x[i][1] * x[i][3] * x[i][2]

    if interaction:
        # З yрахуванням ефекту взаємодії
        print(f'\nМатриця планування для n = {n}, m = {m}:')
        print('\nЗ кодованими значеннями факторів:')
        print('\n     X0    X1    X2    X3  X1X2  X1X3  X2X3 X1X2X3   Y1    Y2     Y3')
        print(np.concatenate((x, y), axis=1))
        print('\nНормовані значення факторів:\n')
        print(x_normalized)
    else:
        print('\nМатриця планування:')
        print('\n   X0  X1  X2  X3  Y1  Y2   Y3  ')
        print(np.concatenate((x, y), axis=1))

    return x, y, x_normalized


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


def bs(x, y, y_aver, n):
    res = [sum(1 * y for y in y_aver) / n]
    for i in range(7):  # 8 - ксть факторів
        b = sum(j[0] * j[1] for j in zip(x[:, i], y_aver)) / n
        res.append(b)
    return res


def kriteriy_studenta2(x, y, y_aver, n, m):
    S_kv = dispersion(y, y_aver, n, m)
    s_kv_aver = sum(S_kv) / n

    # статиcтична оцінка дисперсії
    s_Bs = (s_kv_aver / n / m) ** 0.5
    Bs = bs(x, y, y_aver, n)
    ts = [round(abs(B) / s_Bs, 3) for B in Bs]

    return ts


def kriteriy_studenta(x, y_average, n, m, dispersion):
    dispersion_average = sum(dispersion) / n
    s_beta_s = (dispersion_average / n / m) ** 0.5

    beta = [sum(1 * y for y in y_average) / n]
    for i in range(3):
        b = sum(j[0] * j[1] for j in zip(x[:,i], y_average)) / n
        beta.append(b)

    t = [round(abs(b) / s_beta_s, 3) for b in beta]

    return t


def kriteriy_fishera(y, y_average, y_new, n, m, d, dispersion):
    S_ad = m / (n - d) * sum([(y_new[i] - y_average[i])**2 for i in range(len(y))])
    dispersion_average = sum(dispersion) / n

    return S_ad / dispersion_average


def check(n, m, interaction):

    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05

    x, y, x_norm = planing_matrix(n, m, interaction)

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
    if interaction:
        student_t = kriteriy_studenta2(x_norm[:, 1:], y, y_average, n, m)
    else:
        student_t = kriteriy_studenta(x_norm[:, 1:], y_average, n, m, dispersion_arr)

    print('\nТабличне значення критерій Стьюдента:\n', student_cr_table)
    print('Розрахункове значення критерій Стьюдента:\n', student_t)
    res_student_t = [temp for temp in student_t if temp > student_cr_table]
    final_coefficients = [B[i] for i in range(len(student_t)) if student_t[i] in res_student_t]
    print('\nКоефіцієнти {} статистично незначущі.'.format(
        [round(i, 3) for i in B if i not in final_coefficients]))

    y_new = []
    if interaction:
        for j in range(n):
            y_new.append(regression([x_norm[j][i] for i in range(len(student_t)) if student_t[i] in res_student_t], final_coefficients))
    else:
        for j in range(n):
            y_new.append(regression([x[j][student_t.index(i)] for i in student_t if i in res_student_t], final_coefficients))

    print(f'\nЗначення значення рівння регресії з коефіцієнтами {final_coefficients}: ')
    print(y_new)

    d = len(res_student_t)
    if d >= n:
        print('\nF4 <= 0')
        print('')
        return
    f4 = n - d
    # Розрахункове значення критерія Фішера
    Fp = kriteriy_fishera(y, y_average, y_new, n, m, d, dispersion_arr)
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
    if not check(n, m, False):
        if not check(n, m, True):
            main(n, m)


if __name__ == '__main__':
    # Значення за варіантом
    x_range = ((15, 45), (15, 50), (15, 30))

    y_max = 200 + int(sum([x[1] for x in x_range]) / 3)
    y_min = 200 + int(sum([x[0] for x in x_range]) / 3)

    main(8, 3)