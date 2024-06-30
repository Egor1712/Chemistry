# Chemistry

## Подготовка данных
Сырые данные для обучения приходят в виде больший csv-шек с кучей столбцов. Далее нам нужен gnuplot чтобы из этих csv-шек получить фазовые диограммы (подробности у физиков. Я на виртуалку с ubuntu устанавливал). В папке "gnuplot" лежат немного модифицированные файлы постоения диограмм. Помимо графика, они ещё сохраняют 4 файла с контурами каждой фазы. Эти 2 файла кидаем в папку с данными и запускаем код из раздела **Создание датасета**. В нём нужно будет прописать пити до этих папок. Это будет обучающим/тестовым набором, путь до которого нужно будет указать.

В текущем коде данные нужно поместить в /Data, там же лежат 2 файла скрипта, которые запускаються в разделе **Создание датасета**

## Обучение
Обучение производится в соответствии с комментариями в файле fit_models.ipynb.

## Requirements
Pillow, Numpy, Pytorch

## Пример
Вот ссылка на папку на диске, откуда и запускаеться colab (https://drive.google.com/drive/folders/1VowVHybmaFSP81kaiG8GHuYvxkLkHngp?usp=drive_link)
