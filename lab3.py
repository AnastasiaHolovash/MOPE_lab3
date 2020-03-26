import numpy as np
import random
from numpy.linalg import solve
from scipy.stats import f, t
import time


# Обчислює max та min значення у
def find_y():
    x_max_average = (x1_max + x2_max + x3_max) / 3
    x_min_average = (x1_min + x2_min + x3_min) / 3

    return 200 + int(x_min_average), 200 + int(x_max_average)


# Обчислення рівняння регрестї
def regression(x, b):
    y = sum([x[i]*b[i] for i in range(len(x))])
    return y


# Повертає матрицю з натуралізованими значеннями факторів, матрицю значень експерементів,
# матрицю з нормалізованими значеннями факторів
def planning_matrix(n, m, x_range):
    x_normalized = np.array([[1, -1, -1, -1],
                             [1, -1, 1, 1],
                             [1, 1, -1, 1],
                             [1, 1, 1, -1],
                             [1, -1, -1, 1],
                             [1, -1, 1, -1],
                             [1, 1, -1, -1],
                             [1, 1, 1, 1]])
    # Створює матрицю n на m заповнену нулями
    y = np.zeros(shape=(n,m))
    # Заповнює матрицю планування випадковим чином
    for i in range(n):
        for j in range(m):
            y[i][j] = random.randint(find_y()[0],find_y()[1])

    # Визначає кількість рядків матриці планування
    x_normalized = x_normalized[:len(y)]

    # Створює матрицю заповнену одиницями
    x = np.ones(shape=(len(x_normalized), len(x_normalized[0])))
    # Заповнює матрицю з натуралізованими значеннями факторів
    for i in range(len(x_normalized)):
        for j in range(1, len(x_normalized[i])):
            if x_normalized[i][j] == -1:
                x[i][j] = x_range[j-1][0]
            else:
                x[i][j] = x_range[j-1][1]

    print('\nМатриця планування:' )
    print('\n    X0  X1   X2   X3   Y1   Y2   Y3   Y4')
    print(np.concatenate((x, y), axis=1))

    return x, y, x_normalized

# Повертає масив рередніх значень функції відгуку за рядками та
# масив коефіцієнтів рівняння регресії
def regression_equation(x, y, n):
    y_average = [round(sum(i) / len(i), 2) for i in y]

    mx1 = sum(x[:, 1]) / n
    mx2 = sum(x[:, 2]) / n
    mx3 = sum(x[:, 3]) / n

    my = sum(y_average) / n

    a1 = sum([y_average[i] * x[i][1] for i in range(len(x))]) / n
    a2 = sum([y_average[i] * x[i][2] for i in range(len(x))]) / n
    a3 = sum([y_average[i] * x[i][3] for i in range(len(x))]) / n

    a12 = sum([x[i][1] * x[i][2] for i in range(len(x))]) / n
    a13 = sum([x[i][1] * x[i][3] for i in range(len(x))]) / n
    a23 = sum([x[i][2] * x[i][3] for i in range(len(x))]) / n

    a11 = sum([i ** 2 for i in x[:, 1]]) / n
    a22 = sum([i ** 2 for i in x[:, 2]]) / n
    a33 = sum([i ** 2 for i in x[:, 3]]) / n

    X = [[1, mx1, mx2, mx3], [mx1, a11, a12, a13], [mx2, a12, a22, a23], [mx3, a13, a23, a33]]
    Y = [my, a1, a2, a3]
    B = [round(i, 2) for i in solve(X, Y)]

    print('\nРівняння регресії:')
    print(f'y = {B[0]} + {B[1]}*x1 + {B[2]}*x2 + {B[3]}*x3')

    return y_average, B


# Повертає масив з дисперсіями по рядках
def dispersion(y, y_aver, n, m):
    result = []
    for i in range(n):
        summ = sum([(y_aver[i] - y[i][j])**2 for j in range(m)]) / m
        result.append(summ)
    return result


def kriteriy_studenta(x, y_average, n, m, dispersion):
    dispersion_average = sum(dispersion) / n
    s_beta_s = (dispersion_average / n / m) ** 0.5

    beta = [sum(1 * y for y in y_average) / n]
    for i in range(n-1):
        b = sum(j[0] * j[1] for j in zip(x[:,i], y_average)) / n
        beta.append(b)

    t = [abs(b) / s_beta_s for b in beta]

    return t


def kriteriy_fishera(y, y_average, y_new, n, m, d, dispersion):
    S_ad = m / (n - d) * sum([(y_new[i] - y_average[i])**2 for i in range(len(y))])
    dispersion_average = sum(dispersion) / n

    return S_ad / dispersion_average


def main(n, m):
    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05

    x_range = [(x1_min, x1_max), (x2_min, x2_max), (x3_min, x3_max)]

    x, y, x_norm = planning_matrix(n, m, x_range)

    y_average, B = regression_equation(x, y, n)

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
        main(n, m)

    # Табличне значення критерія Стюдента
    qq = (1 + 0.95) / 2
    student_cr_table = t.ppf(df=f3, q=qq)
    # Розрахункове значення критерія Стюдента
    student_t = kriteriy_studenta(x_norm[:,1:], y_average, n, m, dispersion_arr)

    print('\nТабличне значення критерій Стьюдента:\n', student_cr_table)
    print('Розрахункове значення критерій Стьюдента:\n', student_t)
    res_student_t = [temp for temp in student_t if temp > student_cr_table]
    final_coefficients = [B[student_t.index(i)] for i in student_t if i in res_student_t]
    print('Коефіцієнти {} статистично незначущі.'.
          format([i for i in B if i not in final_coefficients]))

    y_new = []
    for j in range(n):
        y_new.append(regression([x[j][student_t.index(i)] for i in student_t if i in res_student_t], final_coefficients))

    print(f'\nОтримаємо значення рівння регресії для {m} дослідів: ')
    print(y_new)

    d = len(res_student_t)
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
    else:
        print('Математична модель не адекватна експериментальним даним')


if __name__ == '__main__':

    # Значення за варіатом
    x1_min = -30
    x1_max = 20
    x2_min = 25
    x2_max = 45
    x3_min = 25
    x3_max = 30

    start_time = time.time()
    main(4, 4)
    print(f'\nЧас виконання програми: {(time.time() - start_time)} секунд')
