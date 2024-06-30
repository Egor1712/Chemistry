import os

# path to the folder with outlines
base_path = 'c/v-variation/'
# path where the outlines will be saved
result_path = 'Chemistry/Data2_V'

for file_name in os.listdir(base_path):
    # some filter
    if 'result' not in file_name or 'contour' not in file_name:
        continue
    print('move', file_name)
    output_file_name = file_name.replace('.csv', '_')
    with open(os.path.join(base_path, file_name), 'r') as file:
        with open(os.path.join(result_path, output_file_name), 'w') as output_file:
            output_file.writelines(file.readlines())
