import tkinter as tk

import numpy as np
from PIL import Image, ImageTk
from tkinter import ttk, HORIZONTAL


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.canvas_size = (640, 640)
        self.phases_size = (64, 64)
        self.height, self.width = self.canvas_size
        self.phase_count = 4
        self.phase_colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255)]
        self.phases = np.zeros((4, *self.phases_size), dtype=int)
        self.imageTk = None
        self.radius = 5

        self.canvas = tk.Canvas(width=self.width, height=self.height, bg='white')
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.pack()

        control = tk.Frame()
        control.pack()

        tool_settings = tk.Frame(control)
        tool_settings.pack(side='left')

        tools = tk.Frame(tool_settings)
        tools.pack(side='left')
        self.tool = tk.StringVar(value='pen')
        button = ttk.Radiobutton(tools, text='pen', value='pen', variable=self.tool)
        button.pack()
        button = ttk.Radiobutton(tools, text='eraser', value='eraser', variable=self.tool)
        button.pack()

        radius_settings = tk.Frame(tools)
        radius_settings.pack()
        self.radius_label = ttk.Label(radius_settings, text=f'size: {self.radius}')
        self.radius_label.pack()
        size_bar = ttk.Scale(radius_settings, orient=HORIZONTAL, length=200, from_=1, to=10, command=self.change_radius,
                             value=self.radius)
        size_bar.pack()

        phases = tk.Frame(tool_settings)
        phases.pack(side='left')
        self.phase = tk.IntVar(value=0)
        button = ttk.Radiobutton(phases, text='AFM red', value=0, variable=self.phase, command=self.update_canvas)
        button.pack()
        button = ttk.Radiobutton(phases, text='CO blue', value=1, variable=self.phase, command=self.update_canvas)
        button.pack()
        button = ttk.Radiobutton(phases, text='FL pink', value=2, variable=self.phase, command=self.update_canvas)
        button.pack()
        button = ttk.Radiobutton(phases, text='SC green', value=3, variable=self.phase, command=self.update_canvas)
        button.pack()

        file_settings = tk.Frame(control)
        file_settings.pack(side='left')
        label = tk.Label(file_settings, text='file name')
        label.pack()
        self.file_name = tk.Entry(file_settings, width=80)
        self.file_name.pack()
        save_button = tk.Button(file_settings, text='save file', command=self.phases_to_csv_file)
        save_button.pack()
        save_button = tk.Button(file_settings, text='load file', command=self.csv_file_to_phases)
        save_button.pack()
        save_button = tk.Button(file_settings, text='clear', command=self.clear)
        save_button.pack()

    def change_radius(self, new_val):
        self.radius = int(float(new_val))
        self.radius_label.config(text=f'size: {self.radius}')

    def update_canvas(self):
        img = self.phases_to_image()
        self.imageTk = ImageTk.PhotoImage(image=Image.fromarray(img))
        self.canvas.create_image(int(self.width / 2), int(self.height / 2), image=self.imageTk)

    def draw(self, event):
        x_center = int(event.x / self.canvas_size[1] * self.phases_size[1])
        y_center = int(event.y / self.canvas_size[0] * self.phases_size[0])
        radius = self.radius
        for x in range(-radius, radius):
            for y in range(-radius, radius):
                if (self.phases_size[0] <= y + y_center or y + y_center < 0 or
                        self.phases_size[1] <= x + x_center or x + x_center < 0):
                    continue
                if x ** 2 + y ** 2 < radius ** 2:
                    if self.tool.get() == 'pen':
                        self.phases[self.phase.get(), y + y_center, x + x_center] = 1
                    elif self.tool.get() == 'eraser':
                        self.phases[self.phase.get(), y + y_center, x + x_center] = 0
        self.update_canvas()

    def phases_to_image(self):
        repeat = 10
        img = np.ones((*self.canvas_size, 3), dtype=np.uint8()) * 255
        phases = self.phases.repeat(repeat, axis=1).repeat(repeat, axis=2)[..., np.newaxis]
        for i in range(self.phase_count):
            if i == self.phase.get():
                layer = phases[i] * np.array(self.phase_colors[i]) * 0.6
                img -= layer.astype(np.uint8())
            else:
                layer = phases[i] * np.array(self.phase_colors[i]) * 0.2
                img -= layer.astype(np.uint8())
        return img

    def phases_to_csv_file(self):
        with open(self.file_name.get(), 'w') as file:
            for x in range(self.phases.shape[2]):
                for y in range(self.phases.shape[1]):
                    line = [self.phases.shape[1] - y - 1, x]
                    for i in range(self.phase_count):
                        line.append(self.phases[i, y, x])
                    file.write(','.join(map(str, line)) + '\n')

    def csv_file_to_phases(self):
        with open(self.file_name.get(), 'r') as file:
            for line in file.readlines():
                y, x, *phases = map(int, line.split(','))
                for i, p in enumerate(phases):
                    self.phases[i, self.phases.shape[1] - y - 1, x] = p
        self.update_canvas()

    def clear(self):
        self.phases = np.zeros((4, *self.phases_size), dtype=int)
        self.update_canvas()


def main():
    root = App()
    root.mainloop()


if __name__ == '__main__':
    main()
