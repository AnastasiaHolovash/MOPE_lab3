import random
import numpy as np
import sklearn.linear_model as lm
from scipy.stats import f, t
from prettytable import PrettyTable
import math
import time


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


def myVariantY(x1, x2, x3, interaction):
    result = 0
    if interaction:
        result = 10.0 + 6.7 * x1 + 6.4 * x2 + 8.0 * x3 +(1.2 * x1 * x2) + (0.7 * x1 * x3) + (1.1 * x2 * x3) + (2.8 * x1 * x2 * x3)
    else:
        result = 10.0 + 6.7 * x1 + 6.4 * x2 + 8.0 * x3 + (6.2 * x1 * x1) + (0.2 * x2 * x2) + (7.6 * x3 * x3) + (1.2 * x1 * x2) + (0.7 * x1 * x3) + (1.1 * x2 * x3) + (2.8 * x1 * x2 * x3)
    
    return round(result, 3)


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

    # print(y)
    # З yрахуванням ефекту взаємодії
    if interaction or quadratic_terms:
        for x in x_normalized:
            x.append(x[1] * x[2])
            x.append(x[1] * x[3])
            x.append(x[2] * x[3])
            x.append(x[1] * x[2] * x[3])

    l = 1.73
    
    if quadratic_terms:
        for row in x_normalized:
            for i in range(0, 3):
                row.append(1)

        for i in range(0, 3):
            row1 = [1]
            row2 = [1]
            for _ in range(0, i):
                row1.append(0)
                row2.append(0)
            row1.append(-l)
            row2.append(l)
            for _ in range(0, 6):
                row1.append(0)
                row2.append(0)
            row1.append(round(l * l, 3))
            row2.append(round(l * l, 3))
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


    # Визначає кількість рядків матриці планування
    x_normalized = np.array(x_normalized[:14])
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
    if quadratic_terms:
        x[8] = [1, -l * delta_x(0) + x_nul(0), x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[9] = [1, l * delta_x(0) + x_nul(0), x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[10] = [1, x_nul(0), -l * delta_x(1) + x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[11] = [1, x_nul(0), l * delta_x(1) + x_nul(1), x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[12] = [1, x_nul(0), x_nul(1), -l * delta_x(2) + x_nul(2), 1, 1, 1, 1, 1, 1, 1]
        x[13] = [1, x_nul(0), x_nul(1), l * delta_x(2) + x_nul(2), 1, 1, 1, 1, 1, 1, 1]

        for i in range(8, 14):
            for j in range(0, 11):
                x[i][j] = round(x[i][j], 3)

    # З yрахуванням ефекту взаємодії
    if interaction or quadratic_terms:
        for i in range(len(x)):
            x[i][4] = round(x[i][1] * x[i][2], 3)
            x[i][5] = round(x[i][1] * x[i][3], 3)
            x[i][6] = round(x[i][2] * x[i][3], 3)
            x[i][7] = round(x[i][1] * x[i][3] * x[i][2], 3)
    if quadratic_terms:
        for i in range(len(x)):
            x[i][8] = round(x[i][1] * x[i][1], 3)
            x[i][9] = round(x[i][2] * x[i][2], 3)
            x[i][10] = round(x[i][3] * x[i][3], 3)

    # Створює матрицю n на m заповнену нулями
    y = np.zeros(shape=(n, m))
    # Заповнює матрицю планування випадковим чином
    for i in range(n):
        for j in range(m):
            x1 = x[i][1]
            x2 = x[i][2]
            x3 = x[i][3]
            y[i][j] = myVariantY(x1, x2, x3, interaction) + random.randrange(0, 10) - 5

    if interaction:
        # З yрахуванням ефекту взаємодії
        print(f'\nМатриця планування для n = {n}, m = {m}')

        print('\nЗ кодованими значеннями факторів:')
        caption = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "Y1", "Y2", "Y3"]
        rows_kod = np.concatenate((x, y), axis=1)
        print_table(caption, rows_kod)

        print('\nЗ нормованими значеннями факторів:\n')
        rows_norm = np.concatenate((x_normalized, y), axis=1)
        print_table(caption, rows_norm)
    else:
        print('\nМатриця планування:')
        caption = ["X0", "X1", "X2", "X3", "X1X2", "X1X3", "X2X3", "X1X2X3", "X1^2", "X2^2", "X3^2", "Y1", "Y2", "Y3"]
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


# Функція для знаходження квадратної дисперсії
def s_kv(y, y_aver, n, m):
    res = []
    for i in range(n):
        s = sum([(y_aver[i] - y[i][j]) ** 2 for j in range(m)]) / m
        res.append(s)
    return res


# Функція для знаходження критерія фішера
def kriteriy_fishera(y, y_aver, y_new, n, m, d):

    S_kv_ad = (m / (n - d)) * sum([(y_new[i] - y_aver[i]) ** 2 for i in range(len(y))])
    S_kv_b = s_kv(y, y_aver, n, m)
    S_kv_b_aver = sum(S_kv_b) / n

    return S_kv_ad / S_kv_b_aver


def check(n, m, interaction, quadratic_terms, iterationNumber):
    if iterationNumber == maxIterationNumber:
        print("{} ітерацій виконано. Модель не адекватна.".format(iterationNumber))
        return True

    f1 = m - 1
    f2 = n
    f3 = f1 * f2
    q = 0.05

    x, y, x_norm = planing_matrix(n, m, interaction, quadratic_terms)

    y_average = [round(sum(i) / len(i), 3) for i in y]
    
    B = np.linalg.lstsq(x, y_average, rcond=None)[0]

    print('\nСереднє значення y:', y_average)

    dispersion_arr = dispersion(y, y_average, n, m)

    y_perevirka = []
    list_bi = B
    x_nat = x
    for i in range(n):
        if interaction:
            y_perevirka.append(
                list_bi[0] + list_bi[1] * x_nat[i][1] + list_bi[2] * x_nat[i][2] + list_bi[3] * x_nat[i][3] + list_bi[4] * x_nat[i][4] + list_bi[5] * x_nat[i][5] + list_bi[6] * x_nat[i][6] + list_bi[7] * x_nat[i][7])
        else:
            y_perevirka.append(
                list_bi[0] + list_bi[1] * x_nat[i][1] + list_bi[2] * x_nat[i][2] + list_bi[3] * x_nat[i][3] + list_bi[4] * x_nat[i][4] + list_bi[5] * x_nat[i][5] + list_bi[6] * x_nat[i][6] + list_bi[7] * x_nat[i][7] + list_bi[8] *
                x_nat[i][8] + list_bi[9] * x_nat[i][9] + list_bi[10] * x_nat[i][10])

    print(
        "\nПеревірка \n   (підставимо значення факторів з матриці планування і порівняємо результат з середніми значеннями функції відгуку за строками):")
    for i in range(len(y_perevirka)):
        print(" y{} (перевірка) = {} ≈ {} ".format((i + 1), y_perevirka[i], y_average[i]))
    print("\nОскільки значення приблизно однакові, то коефіціенти знайдені правильно")

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
        check(n, m, interaction, quadratic_terms, iterationNumber)

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
            y_new.append(round(regression([x[j][i] for i in range(len(student_t)) if student_t[i] in res_student_t],
                                    final_coefficients), 3))
    else:
        for j in range(n):
            y_new.append(round(regression([x[j][i] for i in range(len(student_t)) if student_t[i] in res_student_t],
            final_coefficients), 3))

    print("\nПеревірка при підстановці в спрощене рівняння регресії:")
    differ = []
    for i in range(len(y_new)):
        differ.append(round(abs(y_new[i] - y_average[i]), 3))

    table = PrettyTable()
    table.add_column("y", y_new)
    table.add_column("y(середнє)", y_average)
    table.add_column("Різниця", differ)
    print(table)

    for i in range(len(final_coefficients)):
        final_coefficients[i] = round(final_coefficients[i], 3)

    print(f'\nКоефіцієнти рівння регресії: {final_coefficients}')
    for i in range(len(y_new)):
        y_new[i] = round(y_new[i], 3)

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


def main(n, m, iterationNumber):
    iterationNumber += 1

    # Спробуємо еффект взаємодії
    if not check(n, m, True, False, iterationNumber):
        
        # Якщо модель не адекватна, переходимо до еффекту квадратних членів
        if not check(14, m, False, True, iterationNumber):
            main(n, m, iterationNumber)


if __name__ == '__main__':
    # Значення за варіантом
    x_range = ((-30, 20), (25, 45), (25, 30))

    # Максимальна кількість ітерацій
    maxIterationNumber = 20

    # Початок виконання
    main(8, 3, 0)
