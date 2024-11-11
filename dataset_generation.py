from os import listdir
from os.path import isfile, join
import re
import os
from multiprocessing import Pool
from image_generation import create_and_save_png_by_csv

def process_file(names):
    create_and_save_png_by_csv(names[0], names[1])

if __name__ == '__main__':
    pool_size = 3
    directory = r'' #place here directory with your csv
    result_directory = os.path.join(directory, 'result')
    regex = re.compile('.*.csv')


    if not os.path.exists(result_directory):
        os.mkdir(result_directory)
    files = [f for f in listdir(directory) if isfile(join(directory, f)) and regex.match(f)]
    filenames = map(lambda f: [os.path.join(directory, f), os.path.join(result_directory, f.replace(".csv", '.png'))], files)

    with Pool(pool_size) as pool:
        results = pool.map(process_file, filenames)
