# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

# Словарь цветов для масок пересечений контуров
overlap_colors = {
    'afm_co': '#ff9900',  # цвет для пересечения AFM и CO
    'afm_sc': 'green',    # цвет для пересечения AFM и SC
    'afm_fl': 'magenta',  # цвет для пересечения AFM и FL
    'co_sc': '#00ff00',   # цвет для пересечения CO и SC
    'co_fl': '#856959',   # цвет для пересечения CO и FL
    'sc_fl': '#00004d',   # цвет для пересечения SC и FL
    'co_sc_fl': '#303136' # цвет для пересечения CO, SC и FL
}

# Задаем минимальный уровень для контуров
zerolevel = 0.1

# Ограничения значений по осям
x_min, x_max = 0, 0.8
y_min, y_max = 0, 0.22

def check_levels(z, zerolevel):
    '''
    Генерация уровней значений, распределяет значения для непустого массива.
    Аргументы:
             z: массив контура
             zerolevel: константа для минимального уровня контуров

    Возвращает:
             Массив или None, если массив пустой
    '''
    if z is None:
        return None
    min_val = np.nanmin(z)
    max_val = np.nanmax(z)
    if min_val == max_val or np.isclose(min_val, max_val):
        return None
    if zerolevel < min_val:
        zerolevel = min_val
    levels = np.linspace(zerolevel, max_val, 10)
    if np.all(np.isclose(levels[:-1], levels[1:])):
        return None
    return levels

def plot_filled_contour(n1, T1, z, levels, color, label):
    '''
    Создает заливку контуров на графике.

    Аргументы:
             n1, T1: значения по осям OX и OY
             z: массив контура
             levels: уровни контура
             color: цвет заливки
             label: метка
    '''
    if z is not None and levels is not None:
        plt.contourf(n1, T1, z, levels=levels, colors=color)
        plt.contour(n1, T1, z, levels=[levels[0]], colors=color, linewidths=1, label=label)

def create_overlap_mask(*arrays, threshold=zerolevel):
    '''
    Создает маску для пересечения нескольких массивов.

    Аргументы:
             *arrays: набор массивов
             threshold: пороговое значение

    Возвращает:
            Маска пересечения массивов
    '''
    mask = np.ones(arrays[0].shape, dtype=bool)
    for arr in arrays:
        if arr is not None:
            mask = np.logical_and(mask, arr >= threshold)
    return mask

def plot_overlap(n1, T1, mask, color, label):
    '''
    Отображает маску на графике заданным цветом.

    Аргументы:
            n1, T1: значения по осям OX и OY
            mask: маска пересечения массивов
            color: цвет маски
            label: метка
    '''
    plt.contourf(n1, T1, np.where(mask, 0, np.nan), levels=[-1, 1], colors=color)
    plt.contour(n1, T1, np.where(mask, 0, np.nan), levels=[-1, 1], colors=color, linewidths=1, label=label)


def create_and_save_png_by_csv(filename, result_filename):
    # Загрузка данных из .csv файла и их обработка
    csv1 = pd.read_csv(
        filename,
        header=None
    )
    csv1.columns = ['0', 'Температура', 'Концентрация примеси', '3',
                    '4', 'CO', '6', '7', '8',
                    'AFM', '10', 'FL', 'SC', '13']
    csv1_1 = csv1.drop(columns=['0', '3', '4', '6', '7', '8', '10', '13'])

    # Определяем набор данных
    n = csv1_1.loc[:, 'Концентрация примеси']  # Ось OX
    T = csv1_1.loc[:, 'Температура']           # Ось OY
    AFM = csv1_1.loc[:, 'AFM']                 # Данные для AFM
    CO = csv1_1.loc[:, 'CO']                   # Данные для CO
    SC = csv1_1.loc[:, 'SC']                   # Данные для SC
    FL = csv1_1.loc[:, 'FL']                   # Данные для FL

    # Сетка значений для n и T
    n1 = np.linspace(x_min, x_max, 300)
    T1 = np.linspace(y_min, y_max, 300)
    n1, T1 = np.meshgrid(n1, T1)

    # Интерполяция данных на сетку
    AFM_1 = griddata((n, T), AFM, (n1, T1), method='cubic')
    CO_1 = griddata((n, T), CO, (n1, T1), method='cubic')
    SC_1 = griddata((n, T), SC, (n1, T1), method='cubic')
    FL_1 = griddata((n, T), FL, (n1, T1), method='cubic')

    # Обработка NaN значений, замена на zerolevel
    AFM_1 = np.nan_to_num(AFM_1, nan=zerolevel) if AFM_1 is not None else None
    CO_1 = np.nan_to_num(CO_1, nan=zerolevel) if CO_1 is not None else None
    SC_1 = np.nan_to_num(SC_1, nan=zerolevel) if SC_1 is not None else None
    FL_1 = np.nan_to_num(FL_1, nan=zerolevel) if FL_1 is not None else None

    # Использование функции check_levels() для определения уровней контуров
    levels_afm = check_levels(AFM_1, zerolevel)
    levels_co = check_levels(CO_1, zerolevel)
    levels_sc = check_levels(SC_1, zerolevel)
    levels_fl = check_levels(FL_1, zerolevel)

    # Применение Гаусс-фильтра для сглаживания данных
    AFM_1 = gaussian_filter(AFM_1, sigma=7) if levels_afm is not None else None
    CO_1 = gaussian_filter(CO_1, sigma=7) if levels_co is not None else None
    SC_1 = gaussian_filter(SC_1, sigma=7) if levels_sc is not None else None
    FL_1 = gaussian_filter(FL_1, sigma=7) if levels_fl is not None else None

    # Созданем фигуру, устанавливая размер полотна с длиной и шириной 5 дюймов
    plt.figure(figsize=(5, 5))

    # Проверка и создание масок пересечений областей при помощи функции create_overlap_mask()
    mask_afm_co = create_overlap_mask(AFM_1, CO_1) if AFM_1 is not None and CO_1 is not None else None
    mask_afm_sc = create_overlap_mask(AFM_1, SC_1) if AFM_1 is not None and SC_1 is not None else None
    mask_afm_fl = create_overlap_mask(AFM_1, FL_1) if AFM_1 is not None and FL_1 is not None else None
    mask_co_sc = create_overlap_mask(CO_1, SC_1) if CO_1 is not None and SC_1 is not None else None
    mask_co_fl = create_overlap_mask(CO_1, FL_1) if CO_1 is not None and FL_1 is not None else None
    mask_sc_fl = create_overlap_mask(SC_1, FL_1) if SC_1 is not None and FL_1 is not None else None

    # Проверка наличия маски и отрисовка при помощи функции plot_overlap()
    if mask_afm_co is not None:
        plot_overlap(n1, T1, mask_afm_co, overlap_colors['afm_co'], 'AFM & CO')

    if mask_afm_sc is not None:
        plot_overlap(n1, T1, mask_afm_sc, overlap_colors['afm_sc'], 'AFM & SC')

    if mask_afm_fl is not None:
        plot_overlap(n1, T1, mask_afm_fl, overlap_colors['afm_fl'], 'AFM & FL')

    if mask_co_sc is not None:
        plot_overlap(n1, T1, mask_co_sc, overlap_colors['co_sc'], 'CO & SC')

    if mask_co_fl is not None:
        plot_overlap(n1, T1, mask_co_fl, overlap_colors['co_fl'], 'CO & FL')

    if mask_sc_fl is not None:
        plot_overlap(n1, T1, mask_sc_fl, overlap_colors['sc_fl'], 'SC & FL')

    # Обработка единственного случая пересечения трех контуров CO, SC и FL
    if mask_co_sc is not None and mask_co_fl is not None and mask_sc_fl is not None:
        mask_co_sc_fl = np.logical_and(mask_co_sc, np.logical_and(mask_co_fl, mask_sc_fl))
        plot_overlap(n1, T1, mask_co_sc_fl, overlap_colors['co_sc_fl'], 'CO, SC & FL')

    # Блок для создания новых версий массивов контуров (для точности отрисовки)
    AFM_1_no_overlap = np.where(np.logical_or(
                                    np.logical_or(mask_afm_co if mask_afm_co is not None else np.zeros_like(n1, dtype=bool),
                                                  mask_afm_sc if mask_afm_sc is not None else np.zeros_like(n1, dtype=bool)),
                                    mask_afm_fl if mask_afm_fl is not None else np.zeros_like(n1, dtype=bool)),
                                np.nan, AFM_1) if AFM_1 is not None else None

    CO_1_no_overlap = np.where(np.logical_or(
                                    np.logical_or(mask_afm_co if mask_afm_co is not None else np.zeros_like(n1, dtype=bool),
                                                  mask_co_sc if mask_co_sc is not None else np.zeros_like(n1, dtype=bool)),
                                    mask_co_fl if mask_co_fl is not None else np.zeros_like(n1, dtype=bool)),
                                np.nan, CO_1) if CO_1 is not None else None

    SC_1_no_overlap = np.where(np.logical_or(
                                    np.logical_or(mask_afm_sc if mask_afm_sc is not None else np.zeros_like(n1, dtype=bool),
                                                  mask_co_sc if mask_co_sc is not None else np.zeros_like(n1, dtype=bool)),
                                    mask_sc_fl if mask_sc_fl is not None else np.zeros_like(n1, dtype=bool)),
                                np.nan, SC_1) if SC_1 is not None else None

    FL_1_no_overlap = np.where(np.logical_or(
                                    np.logical_or(mask_afm_fl if mask_afm_fl is not None else np.zeros_like(n1, dtype=bool),
                                                  mask_co_fl if mask_co_fl is not None else np.zeros_like(n1, dtype=bool)),
                                    mask_sc_fl if mask_sc_fl is not None else np.zeros_like(n1, dtype=bool)),
                                np.nan, FL_1) if FL_1 is not None else None

    # Создание контуров при помощи функции plot_filled_contour()
    plot_filled_contour(n1, T1, AFM_1_no_overlap, levels_afm, 'red', 'AFM')
    plot_filled_contour(n1, T1, CO_1_no_overlap, levels_co, 'yellow', 'CO')
    plot_filled_contour(n1, T1, SC_1_no_overlap, levels_sc, 'blue', 'SC')
    plot_filled_contour(n1, T1, FL_1_no_overlap, levels_fl, 'purple', 'FL')

    # Настройка осей и заголовков
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('n')
    plt.ylabel('T')

    plt.savefig(result_filename)
