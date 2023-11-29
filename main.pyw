from tkinter import *
from tkinter import filedialog, messagebox, colorchooser
TkLabel = Label
from tkinter.ttk import *

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
from matplotlib.animation import FuncAnimation

from numba import cuda
from datetime import datetime
from math import *
from mpmath import mpf, mp
from scipy.ndimage import gaussian_filter, zoom
from typing import *
from threading import Thread
from skimage.transform import rescale

import matplotlib.pyplot as plt
import numpy as np
import nvidia_smi, tkinter, tempfile, os
import pickle, re
import moviepy.video.io.ImageSequenceClip
import time, cv2

initial_iteration_count = 80
inset_color = np.array([0, 0, 0])

iteration_coefficient = 0.96
blur_sigma = 0.0
brightness = 5
spectrum_offset = 0

zoom_coefficient = 0.9
spss_factor = 4
interpolation_method = "bicubic"

show_coordinates = True
show_zoom = True
show_iteration_count = False

last_computation = time.time_ns()

g1, g2 = 10, 10
b1, b2 = 20, 20

lod_res = 600

s = time.time()
spectrum = []
for i in range(6):
    match i:
        case 0:
            for i in range(256):
                spectrum.append(np.array([255, i, 0]))
        case 1:
            for i in range(256):
                spectrum.append(np.array([255 - i, 255, 0]))
        case 2:
            for i in range(256):
                spectrum.append(np.array([0, 255, i]))
        case 3:
            for i in range(256):
                spectrum.append(np.array([0, 255 - i, 255]))
        case 4:
            for i in range(256):
                spectrum.append(np.array([i, 0, 255]))
        case 5:
            for i in range(256):
                spectrum.append(np.array([255, 0, 255 - i]))
spectrum = np.array(spectrum)
spectrum_gpu = cuda.to_device(spectrum)

initial_spectrum = []
for i in range(256):
    initial_spectrum.append(np.array([i, 0, 0]))
initial_spectrum = np.array(initial_spectrum)
initial_spectrum_gpu = cuda.to_device(initial_spectrum)

subpixels = [0.0, 0.0, 0.0]
subpixels_gpu = cuda.to_device(subpixels)

nvidia_smi.nvmlInit()

def remove_trailing_9s(s: str):
    n = 0
    for i, c in enumerate(s[::-1]):
        if c in ('9', '0'):
            n += 1
        elif n > 5:
            return float(s[:len(s)-i-1])
    return s

class Config(Toplevel):
    def __init__(self, master: Tk, menu: Menu, name: str, variable: Union[IntVar, DoubleVar], def_value: Union[int, float],
                 min: Union[int, float], max: Union[int, float], preset_labels: bool, default: Tuple[Union[float, int]], apply_func: Callable):
        super().__init__(master)
        self.root = master
        self.menu = menu
        self.variable, self.apply_func, self.def_value = variable, apply_func, def_value

        self._coefficient, self._blur_sigma, self._brightness, self._spectrum_offset = default
        if default[0] is not None: self._coefficient *= 10e+6
        if default[1] is not None: self._blur_sigma  *= 10e+3
        if default[2] is not None: self._brightness  *= 10e+3

        self.wm_title("Fine tune " + name)
        w = 352
        h = 300 
        x = self.master.winfo_x()
        y = self.master.winfo_y()
        self.wm_geometry("%dx%d+%d+%d" % (w, h, x + 100, y + 100))
        self.wm_resizable(height=False, width=False)

        self.var = IntVar(value=def_value)
        self.scale = Scale(self, variable=self.var, from_=min, to=max, orient=VERTICAL)
        self.scale.place(x=10, y=10, height=280)
        self.var.trace_add('write', lambda *args, **kwargs: self.update_preview())

        if preset_labels:
            Label(self, text="Very High", foreground="#6d6d6d").place(x=37, y=57)
            Label(self, text="High", foreground="#6d6d6d").place(x=37, y=106)
            Label(self, text="Medium", foreground="#6d6d6d").place(x=37, y=140)
            Label(self, text="Low", foreground="#6d6d6d").place(x=37, y=174)
            Label(self, text="Very Low", foreground="#6d6d6d").place(x=37, y=221)
        else:
            Label(self, text="Maximum", foreground="#6d6d6d").place(x=37, y=15)
            Label(self, text="Minimum", foreground="#6d6d6d").place(x=37, y=255)

        self.fig = Figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, aspect=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=120, y=30, height=215, width=215)
        Label(self, text="Preview:").place(x=118, y=5)

        self.applyButton = Button(self, text="Apply", width=16, command=self.apply)
        self.cancelButton = Button(self, text="Cancel", width=16, command=self.destroy)
        self.applyButton.place(x=119, y=250)
        self.cancelButton.place(x=230, y=250)

        self.update_preview()

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.focus_force()
        self.transient(master)
        self.mainloop()

    @property
    def coefficient(self) -> Optional[int]:
        return self._coefficient if self._coefficient is not None else self.var.get()
    @property
    def blur_sigma(self) -> Optional[float]:
        return self._blur_sigma if self._blur_sigma is not None else self.var.get()
    @property
    def brightness(self) -> Optional[float]:
        return self._brightness if self._brightness is not None else self.var.get()
    @property
    def spectrum_offset(self) -> Optional[int]:
        return self._spectrum_offset if self._spectrum_offset is not None else self.var.get()

    def apply(self):
        self.variable.set(self.var.get())
        self.applyButton.configure(state=DISABLED)
        self.apply_func()
        self.destroy()
    
    def update_preview(self):
        self.applyButton.configure(state=DISABLED if self.var.get() == self.def_value else NORMAL)
        mandelbrot_kernel[(g1, g2), (b1, b2)](self.root.zoom, np.array([float(x) for x in self.root.center]), self.coefficient / 10e+6, self.root.preview_gpu, spectrum_gpu, initial_spectrum_gpu, int(self.brightness / 10e+3), self.spectrum_offset, inset_color, self.root.smooth_coloring.get())
        self.root.preview_gpu.copy_to_host(self.root.preview)
        self.ax.clear()
        self.ax.imshow(gaussian_filter(self.root.preview, sigma=self.blur_sigma / 10e+3), extent=[-2.5, 1.5, -2, 2])
        self.canvas.draw()
    
    def on_exit(self):
        self.destroy()

class Color(Frame):
    def __init__(self, master: Toplevel, n: int, color: tuple = None, pos: int = 0, state: bool = False):
        super().__init__(master)
        self.var = BooleanVar(value=state)
        self.pos = IntVar(value=pos * 10e+6)
        self.master = master
        self.n = n
        self.color = color if color is not None else [0, 0, 0]

        self.canvas = Canvas(self, height=10, width=10)
        self.cbox = self.canvas.create_rectangle(0, 0, 10, 10, fill="#%02x%02x%02x" % tuple(self.color))
        self.scale = Scale(self, variable=self.pos, from_=0, to=10e+6, orient=HORIZONTAL, state=NORMAL)
        self.selectColor = Button(self, text="Color...", takefocus=0, command=self.pick_color)
        self.remove = Button(self, text="X", width=3, takefocus=0, command=self.delete)

        self.pos.trace_add('write', lambda *args, **kwargs: self.on_scale_update())

        self.canvas.place(x=0, y=6)
        self.scale.place(x=30, y=0, width=302)
        self.selectColor.place(x=340, y=0)
        self.remove.place(x=420, y=0)
    
    def pick_color(self):
        c = colorchooser.Chooser(self, initialcolor=tuple(self.color), parent=self, title="Pick a color").show()[0]
        if c:
            for i in range(3):
                self.color[i] = c[i]
            self.master.update_palette()
            self.canvas.itemconfig(self.cbox, fill="#%02x%02x%02x" % tuple(self.color))
    def on_scale_update(self):
        self.master.update_palette()

    def delete(self):
        for i in range(self.n, len(self.master.colors)):
            c: Color = self.master.colors[i]
            c.n = c.n - 1
            x, y = c.winfo_x(), c.winfo_y()
            c.place_configure(x=x, y=y-35)
        self.master.controls.place_configure(x=self.master.controls.winfo_x(), y=self.master.controls.winfo_y() - 35)
        self.master.colors = np.delete(self.master.colors, self.n - 1)
        self.destroy()
        self.master.update_palette()

class PaletteEditor(Toplevel):
    def __init__(self, master: Tk):
        super().__init__(master)
        self.root: MandelbrotVoyage = master

        self.wm_title("Palette Editor")
        w = 467
        h = 450
        x = self.master.winfo_x()
        y = self.master.winfo_y()
        self.wm_geometry("%dx%d+%d+%d" % (w, h, x + 100, y + 100))
        self.wm_resizable(height=False, width=False)

        self.length = IntVar(value=1792)
        self.colors = []

        self.colors.append(Color(self, 1, [255, 0, 0], 0.0, True))
        self.colors.append(Color(self, 2, [255, 255, 0], 0.2, True))
        self.colors.append(Color(self, 3, [0, 255, 0], 0.4, True))
        self.colors.append(Color(self, 4, [0, 255, 255], 0.6, True))
        self.colors.append(Color(self, 5, [0, 0, 255], 0.8, True))
        self.colors.append(Color(self, 6, [255, 0, 255], 1.0, True))

        self.interpolated_colors = self.interpolate_color(np.array([x.color for x in self.colors]), np.array([x.pos.get() / 10e+6 for x in self.colors]), self.length.get())

        self.fig = Figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, aspect=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=40, y=101, height=20, width=302)
        self.ax.imshow(self.interpolated_colors.reshape(1, -1, 3).astype(np.uint8), aspect='auto', extent=[0, 1, 0, 1])

        for c, y in zip(self.colors, range(139, 140 + len(self.colors) * 35, 35)):
            c.place(x=10, y=y, height=50, width=457)

        self.controls = Frame(self)
        self.addColor = Button(self.controls, text="Add", width=15, command=self.add_color)
        self.addColor.place(x=0, y=5)

        self.fig2 = Figure()
        self.fig2.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax2 = self.fig2.add_subplot(111, aspect=1)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().place(x=351, y=15, height=106, width=106)
        if self.root.subpixel_supersampling.get():
            self.ax2.imshow(gaussian_filter(self.root.rgb_colors, sigma=blur_sigma), extent=[-2.5, 1.5, -2, 2])
        else:
            mandelbrot_kernel[(g1, g2), (b1, b2)](self.root.zoom, np.array([float(x) for x in self.root.center]), zoom_coefficient, self.root.preview_gpu, spectrum_gpu, initial_spectrum_gpu, brightness, spectrum_offset, inset_color, self.root.smooth_coloring.get())
            self.root.preview_gpu.copy_to_host(self.root.preview)
            self.ax2.imshow(gaussian_filter(self.root.preview, sigma=blur_sigma), extent=[-2.5, 1.5, -2, 2])

        self.controls.place(x=10, y=438, height=245, width=457)

        self.protocol("WM_DELETE_WINDOW", self.on_exit)

        self.focus_force()
        self.transient(master)
        self.mainloop()

    def add_color(self):
        self.colors += None
    
    def update_palette(self):
        sorted_indices = np.argsort([x.pos.get() / 10e+6 for x in self.colors])
        self.interpolated_colors = self.interpolate_color(np.array([x.color for x in self.colors])[sorted_indices], np.array([x.pos.get() / 10e+6 for x in self.colors])[sorted_indices], self.length.get())
        self.ax.clear()
        self.ax.imshow(self.interpolated_colors.reshape(1, -1, 3).astype(np.uint8), aspect='auto', extent=[0, 1, 0, 1])
        self.canvas.draw()

    @staticmethod
    def interpolate_color(colors: list[list[int]], positions: list[float], length):
        colors = list(colors)
        positions = list(positions)
        # add black colors to the edges
        colors.insert(0, [0, 0, 0])
        colors.append([0, 0, 0])
        positions.insert(0, 0.0)
        positions.append(1.0)

        t = np.linspace(0, 1, length)
        interpolated_colors = np.zeros((length, 3))

        for i in range(len(colors) - 1):
            sc = colors[i]
            ec = colors[i + 1]
            sp = positions[i]
            ep = positions[i + 1]

            mask = (t >= sp) & (t <= ep)
            segment_t = (t[mask] - sp) / (ep - sp)

            interpolated_colors[mask] = (1 - segment_t[:, np.newaxis]) * sc + segment_t[:, np.newaxis] * ec

        return interpolated_colors

    def on_exit(self):
        self.destroy()

@cuda.jit(nopython=True)
def bicubic_resize(img, new_shape):
    height, width, _ = img.shape
    new_height, new_width = new_shape

    output = np.empty((new_height, new_width, 3), dtype=np.uint8)

    for c in range(3):  # Loop over RGB channels
        for i in range(new_height):
            for j in range(new_width):
                y = i * height / new_height
                x = j * width / new_width
                y_low = int(y)
                x_low = int(x)

                a = y - y_low
                b = x - x_low

                values = np.zeros((4, 4), dtype=img.dtype)
                for k in range(4):
                    for l in range(4):
                        y_idx = y_low - 1 + k
                        x_idx = x_low - 1 + l

                        y_idx = max(0, min(height - 1, y_idx))
                        x_idx = max(0, min(width - 1, x_idx))

                        values[k, l] = img[y_idx, x_idx, c]

                output[i, j, c] = bicubic_interpolate(values, a, b)
    return output

@cuda.jit(nopython=True)
def bicubic_interpolate(p, a, b):
    return (
        p[1, 1] + 0.5 * a * (p[1, 2] - p[1, 0] + a * (2.0 * p[1, 0] - 5.0 * p[1, 1] + 4.0 * p[1, 2] - p[1, 3] + a * (3.0 * (p[1, 1] - p[1, 2]) + p[1, 3] - p[1, 0])))
        + 0.5 * b * (p[2, 1] - p[0, 1] + a * (2.0 * p[0, 1] - 5.0 * p[1, 1] + 4.0 * p[2, 1] - p[3, 1] + a * (3.0 * (p[1, 1] - p[2, 1]) + p[3, 1] - p[0, 1])))
        + 0.5 * a * b * (p[2, 2] - p[0, 2] + a * (2.0 * p[0, 2] - 5.0 * p[1, 2] + 4.0 * p[2, 2] - p[3, 2] + a * (3.0 * (p[1, 2] - p[2, 2]) + p[3, 2] - p[0, 2])))
        + 0.5 * a * b * (p[1, 2] - p[1, 0] + a * (2.0 * p[1, 0] - 5.0 * p[1, 1] + 4.0 * p[1, 2] - p[1, 3] + a * (3.0 * (p[1, 1] - p[1, 2]) + p[1, 3] - p[1, 0])))
    )

@cuda.jit(device=True)
def mandelbrot_pixel(c, max_iters):
    z: complex = c
    for i in range(max_iters):
        if (z.real ** 2 + z.imag ** 2) >= 4:
            return i if z != c else 1
        z = z * z + c
    return 0

@cuda.jit(device=True)
def mandelbrot_pixel_normalized(c, max_iters):
    z: complex = c
    boundry = 15 if z.real ** 2 + z.imag ** 2 >= 4 else 4
    for i in range(max_iters):
        if (z.real ** 2 + z.imag ** 2) >= boundry:
            smooth_value = i + 1 - log2(log2(abs(z)))
            return smooth_value
        z = z * z + c
    return 0.0

@cuda.jit
def mandelbrot_kernel(zoom, center, coefficient, output, spectrum, initial_spectrum, brightness, spectrum_offset, inset_color, smooth_coloring):
    max_iters = initial_iteration_count / (coefficient ** (log(zoom / 4.5) / log(zoom_coefficient)))
    pixel_size = zoom / min(output.shape[0], output.shape[1])
    start_x, start_y = cuda.grid(2)
    grid_x, grid_y = cuda.gridsize(2)

    x_center, y_center = center
    x_offset = x_center - output.shape[1] / 2 * pixel_size
    y_offset = y_center - output.shape[0] / 2 * pixel_size

    for i in range(start_x, output.shape[0], grid_x):
        for j in range(start_y, output.shape[1], grid_y):
            c = complex((j * pixel_size + x_offset), (i * pixel_size + y_offset))
            p = mandelbrot_pixel_normalized(c, max_iters) if smooth_coloring else mandelbrot_pixel(c, max_iters)
            if p != 0.0:
                p += spectrum_offset

                # bilinear interpolation
                t = p * brightness
                if t < 255:
                    index1 = int(t) % len(initial_spectrum)
                    index2 = (index1 + 1) % len(initial_spectrum)
                    t = t % 1
                    color1 = initial_spectrum[index1]
                    color2 = initial_spectrum[index2]
                else:
                    t -= 255
                    index1 = int(t) % len(spectrum)
                    index2 = (index1 + 1) % len(spectrum)
                    t = t % 1
                    color1 = spectrum[index1]
                    color2 = spectrum[index2]

                for k in range(3):
                    output[i, j, k] = (1 - t) * color1[k] + t * color2[k]
            else:
                # Use initial_spectrum for pixels inside the set
                for k in range(3):
                    output[i, j, k] = float(inset_color[k])  # Convert to float

mp.dps = 200

class MandelbrotVoyage(Tk):
    def __init__(self):
        super().__init__()

        self.width, self.height = 600, 620

        self.wm_title("Mandelbrot Voyage")
        self.wm_geometry(f"{self.width}x{self.height}")
        self.configure(bg="black")
        self.wm_iconbitmap("icon.ico")

        self.fig = Figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, aspect=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=0, y=-20, height=self.height + 20, width=self.width)
        self.center = np.array([mpf("-0.4" + "0" * 180), mpf(0)])
        self.zoom = 4.5

        self.custom_coefficient = 0
        self.custom_blur_sigma = 0
        self.custom_brightness = 0
        self.custom_spectrum_offset = 0

        self.subpixel_supersampling = IntVar(value=0)
        self.smooth_coloring = IntVar(value=1)
        self.dynamic_resolution = IntVar(value=1)
        self.use_lod = IntVar(value=1)

        self.preview = np.empty((215, 215, 3), dtype=np.uint8)
        self.preview_gpu = cuda.to_device(self.preview)
        
        self.rgb_colors = np.empty((int((self.height + 20) * (spss_factor if self.subpixel_supersampling.get() else 1)), int(self.width * (spss_factor if self.subpixel_supersampling.get() else 1)), 3), dtype=np.uint8)
        self.rgb_colors_gpu = cuda.to_device(self.rgb_colors)

        self.rgb_colors_lod = np.empty((int(lod_res * self.height / self.width), lod_res, 3), dtype=np.uint8)
        self.rgb_colors_lod_gpu = cuda.to_device(self.rgb_colors_lod)

        self.ax.imshow(self.rgb_colors, extent=[-2.5, 1.5, -2, 2], interpolation=interpolation_method)

        self.load_image = None

        self.bind("<MouseWheel>", self._on_mousewheel)

        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        self.alertLabel = tkinter.Label(self, bg="black", fg="red",
            text="Higher Precision Required - Use \"Render with CPU\"")

        self.dragging = False
        self.dragging_right = False
        self.double_click = False
        self.resize = None
        self.last_x, self.last_y = None, None

        self.changeLocationUI = None

        class menuBar(Menu):
            def __init__(self, root: MandelbrotVoyage):
                super().__init__(root, tearoff=0)

                self.root = root

                class fileMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root: MandelbrotVoyage = self.master.master
                        
                        self.add_command(label="Take screenshot", accelerator="Ctrl+S", command=self.root.save_image)
                        self.add_separator()
                        self.add_command(label="Create zoom video", command=self.root.make_video)
                        self.add_separator()
                        self.add_command(label="Exit", accelerator="Alt+F4", command=lambda: os._exit(0))

                class settingsMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master

                        class iterMenu(Menu):
                            def __init__(self, master: menuBar):
                                super().__init__(master, tearoff=0)
                                self.root = self.master.master.master
                                self.iter = DoubleVar(value=9600000)

                                self.add_radiobutton(label="Very low", value=9850000, variable=self.iter, command=self.change_iteration_count)
                                self.add_radiobutton(label="Low", value=9700000, variable=self.iter, command=self.change_iteration_count)
                                self.add_radiobutton(label="Medium (default)", value=9600000, variable=self.iter, command=self.change_iteration_count)
                                self.add_radiobutton(label="High", value=9500000, variable=self.iter, command=self.change_iteration_count)
                                self.add_radiobutton(label="Very High", value=9350000, variable=self.iter, command=self.change_iteration_count)
                                self.add_separator()
                                self.add_radiobutton(label="Custom", value=self.root.custom_coefficient, variable=self.iter, command=self.change_iteration_count, state=DISABLED)
                                self.add_command(label="Fine tune", command=self.fine_tune)

                            def change_iteration_count(self):
                                global iteration_coefficient
                                iteration_coefficient = self.iter.get() / 10e+6
                                self.root.update_image()

                            def fine_tune(self):
                                def apply():
                                    global iteration_coefficient
                                    self.change_iteration_count()
                                    self.root.custom_coefficient = iteration_coefficient
                                    self.entryconfig(6, state=NORMAL, value=self.iter.get())
                                Config(self.root, self, "iteration increase factor", self.iter, iteration_coefficient * 10e+6, 0.92 * 10e+6, 9999999, True, (None, blur_sigma, brightness, spectrum_offset), apply)

                        class blurMenu(Menu):
                            def __init__(self, master: menuBar):
                                super().__init__(master, tearoff=0)
                                self.root = self.master.master.master
                                self.blur = DoubleVar(value=0)

                                self.add_radiobutton(label="Disabled", value=0, variable=self.blur, command=self.change_blur_sigma)
                                self.add_separator()
                                self.add_radiobutton(label="Very low", value=1900, variable=self.blur, command=self.change_blur_sigma)
                                self.add_radiobutton(label="Low", value=3750, variable=self.blur, command=self.change_blur_sigma)
                                self.add_radiobutton(label="Medium (default)", value=5000, variable=self.blur, command=self.change_blur_sigma)
                                self.add_radiobutton(label="High", value=6300, variable=self.blur, command=self.change_blur_sigma)
                                self.add_radiobutton(label="Very High", value=8100, variable=self.blur, command=self.change_blur_sigma)
                                self.add_separator()
                                self.add_radiobutton(label="Custom", value=self.root.custom_blur_sigma, variable=self.blur, command=self.change_blur_sigma, state=DISABLED)
                                self.add_command(label="Fine tune", command=self.fine_tune)

                            def change_blur_sigma(self):
                                global blur_sigma
                                blur_sigma = self.blur.get() / 10e+3
                                self.root.update_image()

                            def fine_tune(self):
                                def apply():
                                    global blur_sigma
                                    self.change_blur_sigma()
                                    self.root.custom_blur_sigma = blur_sigma
                                    self.entryconfig(8, state=NORMAL, value=self.blur.get())
                                self.blur.set(self.blur.get() * 10e+3)
                                Config(self.root, self, "blur sigma", self.blur, blur_sigma * 10e+3, 10e+3, 0, True, (iteration_coefficient, None, brightness, spectrum_offset), apply)
                        
                        class brightnessMenu(Menu):
                            def __init__(self, master: menuBar):
                                super().__init__(master, tearoff=0)
                                self.root = self.master.master.master
                                self.brightness = DoubleVar(value=50000)

                                self.add_radiobutton(label="Very low", value=19000, variable=self.brightness, command=self.change_brightness)
                                self.add_radiobutton(label="Low", value=37500, variable=self.brightness, command=self.change_brightness)
                                self.add_radiobutton(label="Medium (default)", value=50000, variable=self.brightness, command=self.change_brightness)
                                self.add_radiobutton(label="High", value=63000, variable=self.brightness, command=self.change_brightness)
                                self.add_radiobutton(label="Very High", value=81000, variable=self.brightness, command=self.change_brightness)
                                self.add_separator()
                                self.add_radiobutton(label="Custom", value=self.root.custom_brightness, variable=self.brightness, command=self.change_brightness, state=DISABLED)
                                self.add_command(label="Fine tune", command=self.fine_tune)

                            def change_brightness(self):
                                global brightness
                                brightness = int(self.brightness.get() / 10e+3)
                                self.root.update_image()

                            def fine_tune(self):
                                def apply():
                                    global brightness
                                    self.change_brightness()
                                    self.root.custom_brightness = brightness
                                    self.entryconfig(6, state=NORMAL, value=self.brightness.get())
                                self.brightness.set(self.brightness.get() * 10e+3)
                                Config(self.root, self, "color complexity", self.brightness, brightness * 10e+3, 50 * 10e+3, 0.1 * 10e+3, True, (iteration_coefficient, blur_sigma, None, spectrum_offset), apply)
                        
                        class offsetMenu(Menu):
                            def __init__(self, master: menuBar):
                                super().__init__(master, tearoff=0)
                                self.root = self.master.master.master
                                self.offset = IntVar(value=0)

                                self.add_radiobutton(label="Black", value=0, variable=self.offset, command=self.change_offset)
                                self.add_radiobutton(label="Red", value=256, variable=self.offset, command=self.change_offset)
                                self.add_radiobutton(label="Green", value=768, variable=self.offset, command=self.change_offset)
                                self.add_radiobutton(label="Blue", value=1280, variable=self.offset, command=self.change_offset)
                                self.add_separator()
                                self.add_radiobutton(label="Custom", value=self.root.custom_spectrum_offset, variable=self.offset, command=self.change_offset, state=DISABLED)
                                self.add_command(label="Fine tune", command=self.fine_tune)
                        
                            def change_offset(self):
                                global spectrum_offset
                                spectrum_offset = int(self.offset.get() / brightness)
                                self.root.update_image()

                            def fine_tune(self):
                                def apply():
                                    global spectrum_offset
                                    self.offset.set(self.offset.get() * brightness)
                                    self.change_offset()
                                    self.root.custom_spectrum_offset = spectrum_offset
                                    self.entryconfig(5, state=NORMAL, value=self.offset.get())
                                Config(self.root, self, "color spectrum offset", self.offset, spectrum_offset, 0, (len(spectrum) + len(initial_spectrum)) / brightness, False, (iteration_coefficient, blur_sigma, brightness, None), apply)
                        
                        class showMenu(Menu):
                            def __init__(self, master: menuBar):
                                super().__init__(master, tearoff=0)
                                self.root = self.master.master.master
                                self.coords = IntVar(value=1)
                                self.zoom = IntVar(value=1)
                                self.iteration_count = IntVar(value=0)
                                
                                self.add_checkbutton(label="Coordinates", variable=self.coords, command=self.update)
                                self.add_checkbutton(label="Zoom", variable=self.zoom, command=self.update)
                                self.add_checkbutton(label="Iteration count", variable=self.iteration_count, command=self.update)
                            
                            def update(self):
                                global show_coordinates, show_zoom, show_iteration_count
                                show_coordinates, show_zoom, show_iteration_count = bool(self.coords.get()), bool(self.zoom.get()), bool(self.iteration_count.get())
                                self.root.update_image()
                        
                        self.iterMenu = iterMenu(self)
                        self.blurMenu = blurMenu(self)
                        self.brightness = brightnessMenu(self)
                        self.spectrum_offset = offsetMenu(self)

                        self.showMenu = showMenu(self)

                        self.add_cascade(label="Iteration count", menu=self.iterMenu)
                        self.add_cascade(label="Bluriness", menu=self.blurMenu)
                        
                        self.add_separator()
                        self.add_command(label="Change in-set color", command=self.change_inset_color)
                        self.add_separator()
                        self.add_cascade(label="Show...", menu=self.showMenu)
                        self.add_separator()
                        self.add_checkbutton(label="Subpixel Supersampling (SSAA)", variable=self.root.subpixel_supersampling, command=self.root.adjust_for_resize)
                        self.add_checkbutton(label="Continuous (smooth) coloring", variable=self.root.smooth_coloring, command=self.update_image)
                        self.add_separator()
                        self.add_cascade(label="Palette complexity", menu=self.brightness)
                        self.add_cascade(label="Offset", menu=self.spectrum_offset)
                        self.add_command(label="Palette Editor", command=self.launch_palette_editor)
                    
                    def update_image(self):
                        if self.root.use_lod.get():
                            self.root.load_lod()
                            self.root.load_image = self.root.after(1000, self.root.update_image)
                        else:
                            self.root.update_image()
                    
                    def change_inset_color(self):
                        global inset_color
                        c = colorchooser.Chooser(self.root, initialcolor=tuple(inset_color), parent=self.root, title="Choose the in-set color").show()[0]
                        if c:
                            for i in range(3):
                                inset_color[i] = c[i]
                            self.root.update_image()

                    def launch_palette_editor(self):
                        PaletteEditor(self.root)

                class zoomMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master

                        self.add_command(label="Zoom in",  accelerator="E", command=lambda: self.root.zoom_(+120))
                        self.add_command(label="Zoom out", accelerator="Q", command=lambda: self.root.zoom_(-120))
                        self.add_separator()
                        self.add_command(label="Zoom in X10", command=lambda: self.root.zoom_(+1200))
                        self.add_command(label="Zoom out X10", command=lambda: self.root.zoom_(-1200))
                        self.add_separator()
                        self.add_command(label="Reset magnification", accelerator="R", command=self.root.reset_zoom)

                class locationMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master

                        self.add_command(label="Save location", accelerator="Ctrl+C", command=self.root.save_loc)
                        self.add_command(label="Load location", accelerator="Ctrl+L", command=self.root.load_loc)
                        self.add_separator()
                        self.add_command(label="Reset location", command=self.root.reset_loc)
                        self.add_separator()
                        self.add_command(label="Change location", command=self.root.change_loc)

                class performanceMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master

                        self.add_checkbutton(label="Load low resolution first", variable=self.root.use_lod, command=self.change_lod)
                        self.add_checkbutton(label="Dynamic resolution", variable=self.root.dynamic_resolution, command=self.change_ds)
                    
                    def change_lod(self):
                        self.entryconfig(1, state=NORMAL if self.root.use_lod.get() else DISABLED)
                    def change_ds(self):
                        global lod_res
                        if not self.root.dynamic_resolution.get():
                            lod_res = 400

                self.fileMenu = fileMenu(self)
                self.settingsMenu = settingsMenu(self)
                self.zoomMenu = zoomMenu(self)
                self.locationMenu = locationMenu(self)
                self.performanceMenu = performanceMenu(self)

                self.add_cascade(label = "File", menu=self.fileMenu)
                self.add_cascade(label = "Edit", menu=self.settingsMenu)
                self.add_cascade(label = "Zoom", menu=self.zoomMenu)
                self.add_cascade(label = "Location", menu=self.locationMenu)
                self.add_cascade(label = "Performance", menu=self.performanceMenu)
                self.add_command(label = "About", command=self.about)

            def about(self):
                class About(Toplevel):
                    def __init__(self, master: Tk):
                        super().__init__(master)
                        self.root = master

                        self.wm_title("About Mandelbrot Voyage")
                        self.wm_geometry(f"550x200")
                        self.wm_resizable(height=False, width=False)
                        self.grab_set()

                        Label(self, text="""
Mandelbrot Voyage is a program written in Python, allowing an easy journey into the deeps of
the Mandelbrot set. Despite Python's infamous slowness factor, a smooth voyage is made possible
thanks to hardware acceleration, given you have a good enough GPU.

Libraries used are numba for hardware acceleration, tkinter for UI, matplotlib for visualizing
the set, mpmath for arbitrary precision, and moviepy for creating videos.""").place_configure(x=20, y=0)
                        self.focus_force()
                        self.transient(master)
                        self.mainloop()
                
                About(self.root)

        self.menuBar = menuBar(self)
        self.config(menu = self.menuBar)

        self.fps_history = [] # last 25 fps values recorded

        self.pixelinfotext = None
        
        self.info_x, self.info_y = None, None
        self.textmovement = None

        self.bind("e", lambda _: self.zoom_(+120))
        self.bind("q", lambda _: self.zoom_(-120))
        self.bind("r", lambda _: self.reset_zoom())
        self.bind("<Control_L>s", lambda _: self.save_image())
        self.bind("<F11>", lambda _: self.wm_attributes('-fullscreen', not self.attributes('-fullscreen')))

        self.wm_protocol("WM_DELETE_WINDOW", self.on_close)

        self.adjust_size_task = None
        self.bind("<Configure>", self.on_resize)

    def on_close(self):
        self.destroy()
        self.quit()
        os._exit(0)

    def on_resize(self, event):
        if (event.widget == self and (self.width != event.width or self.height != event.height)):
            if self.resize:
                self.after_cancel(self.resize)
            self.height, self.width = event.height, event.width
            self.resize = self.after(100, self.adjust_for_resize)

    def adjust_for_resize(self):
        if self.adjust_size_task:
            self.after_cancel(self.adjust_size_task)
        self.resize = None
        self.canvas.get_tk_widget().place_forget()
        self.canvas.get_tk_widget().place(x=0, y=0, height=self.height, width=self.width)
        self.rgb_colors = np.empty((int(self.height * (spss_factor if self.subpixel_supersampling.get() else 1)), int(self.width* (spss_factor if self.subpixel_supersampling.get() else 1)), 3), dtype=np.uint8)
        self.rgb_colors_gpu = cuda.to_device(self.rgb_colors)

        self.rgb_colors_lod = np.empty((int(lod_res * self.height / self.width), lod_res, 3), dtype=np.uint8)
        self.rgb_colors_lod_gpu = cuda.to_device(self.rgb_colors_lod)

        self.ax.set_aspect(self.height / self.width)
        self.load_lod()
        self.adjust_size_task = self.after(200, self.update_image)
        self.adjust_size_task = None

    def render_video(self, config: Toplevel):
        config.rendering = True
        final_zoom = self.zoom
        tempfolder = config.tempFolder.get()

        self.zoom = 4.5

        fc = mpf(config.duration.get() * config.fps.get())
        zoom_coefficient = float((final_zoom / mpf(self.zoom)) ** (mpf(1) / fc))

        config.progress.set(0)
        config.progressBar.configure(maximum = fc + 1)

        def change_state(state):
            config.renderButton.configure(state=state)
            config.destinationEntry.configure(state=state)
            config.destinationBrowseButton.configure(state=state)
            config.tempFolderEntry.configure(state=state)
            config.tempFolderBrowseButton.configure(state=state)
            config.durationSpinbox.configure(state=state)
            config.fpsSpinbox.configure(state=state)
            config.resolutionSpinboxW.configure(state=state)
            config.resolutionSpinboxH.configure(state=state)

        change_state(DISABLED)
        config.pauseButton.configure(state=NORMAL)
        config.cancelButton.configure(state=NORMAL)

        if os.path.isdir(tempfolder):
            for file in os.listdir(tempfolder):
                path = os.path.join(tempfolder, file)
                if os.path.isfile(path):
                    os.remove(path)
        else:
            os.mkdir(tempfolder)
        if config.h.get() != self.height or config.w.get() != self.width:
            self.rgb_colors = np.empty((config.h.get(), config.w.get(), 3), dtype=np.uint8)
            self.rgb_colors_gpu = cuda.to_device(self.rgb_colors)

        for i in range(int(fc)):
            mandelbrot_kernel[(g1, g2), (b1, b2)](self.zoom, np.array([float(x) for x in self.center]), iteration_coefficient, self.rgb_colors_gpu, spectrum_gpu, initial_spectrum_gpu, brightness, spectrum_offset, inset_color, self.smooth_coloring.get())
            self.rgb_colors = self.rgb_colors_gpu.copy_to_host()

            self.ax.clear()
            self.ax.imshow(gaussian_filter(self.rgb_colors, sigma=blur_sigma), extent=[-2.5, 1.5, -2, 2], interpolation=interpolation_method)
            self.canvas.draw()

            plt.imsave(tempfolder + f'\\{i}.png', gaussian_filter(self.rgb_colors, sigma=blur_sigma))
            self.zoom *= zoom_coefficient
            config.progressBar.step()
            config.update()

            while config.pause:
                config.update() # run manual mainloop

            if config.halt:
                break
        else:
            config.pauseButton.configure(state=DISABLED)
            config.cancelButton.configure(state=DISABLED)

            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip([os.path.join(tempfolder, img) for img in sorted(os.listdir(tempfolder), key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) if img.endswith(".png")], fps=config.fps.get())
            clip.write_videofile(config.destinationVar.get())

            change_state(NORMAL)
            return
        

    def make_video(self):
        class Video(Toplevel):
            def __init__(self, master: Tk):
                super().__init__(master)

                self.root = master

                self.duration = IntVar(value=10)
                self.fps = IntVar(value=30)
                self.h = IntVar(value=master.height)
                self.w = IntVar(value=master.width)

                self.rendering = False
                self.halt = False
                def halt():
                    self.halt = True
                self.pause = False
                def pause():
                    self.pause = True

                self.tempFolder = StringVar(value=tempfile.gettempdir() + "\\Mandelbrot Voyage")

                self.duration.trace("w", lambda *args: self.on_durationSpinbox_change())

                self.wm_title("Make zoom video")
                self.wm_geometry(f"532x270")
                self.wm_resizable(height=False, width=False)
                self.grab_set()

                def required(func):
                    def wrapper(*args, **kwargs):
                        result: bool = func(*args, **kwargs)
                        self.renderButton.configure(state="normal" if result else "disabled")
                        return result
                    return wrapper

                @required
                def check_destination_validity(path) -> bool:
                    return not os.path.isdir(path) and re.match(r'^[a-zA-Z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*$', path)
                @required
                def check_tempfolder_validity(path)  -> bool:
                    return os.path.isdir(os.path.dirname(path))

                self.destionationLabel = Label(self, text="Destination:", foreground="red")
                self.destinationVar = StringVar()
                self.destinationVar.trace_add("write", lambda *args, **kwargs: self.destionationLabel.configure(
                    foreground="black" if check_destination_validity(self.destinationEntry.get()) else "red"))
                self.destinationEntry = Entry(self, width=45, textvariable=self.destinationVar)
                self.destinationBrowseButton = Button(self, text="Browse...", width=18, command=self.ask_destionation, takefocus=0)

                self.tempFolderLabel = Label(self, text="Temporary Folder:")
                self.tempFolder.trace_add("write", lambda *args, **kwargs: self.tempFolderLabel.configure(
                    foreground="black" if check_tempfolder_validity(self.tempFolderEntry.get()) else "red"))
                self.tempFolderEntry = Entry(self, width=45, textvariable=self.tempFolder)
                self.tempFolderBrowseButton = Button(self, text="Browse...", width=18, command=self.ask_tempfolder, takefocus=0)

                self.durationLabel = Label(self, text="Duration:")
                self.durationSpinbox = Spinbox(self, width=10, textvariable=self.duration, increment=1, from_=0, to=240, validate="key", validatecommand=(self.register(self.validate_durationSpinbox), "%P"))
                self.durationUnitLabel = Label(self, text="seconds", foreground="gray")
                self.fpsLabel = Label(self, text="FPS:")
                self.fpsSpinbox = Spinbox(self, width=10, textvariable=self.fps, increment=1, from_=0, to=60, validate="key", validatecommand=(self.register(self.validate_durationSpinbox), "%P"))
                self.resolutionLabel = Label(self, text="Resolution:")
                self.resolutionSpinboxW = Spinbox(self, width=6, textvariable=self.w, increment=1, from_=0, to=2160, validate="key", validatecommand=(self.register(self.validate_durationSpinbox), "%P"))
                self.resolutionSpinboxH = Spinbox(self, width=6, textvariable=self.h, increment=1, from_=0, to=2160, validate="key", validatecommand=(self.register(self.validate_durationSpinbox), "%P"))
                self.resolutionUnitLabel = Label(self, text="x", foreground="gray")

                self.renderButton = Button(self, text="Render video", width=31, command=lambda: self.root.render_video(self), state="disabled", takefocus=0)

                self.progress = IntVar(value=0)

                style = Style()
                style.theme_use("vista")
                style.configure("CustomProgressbar.Horizontal", troughcolor='#e6e6e6', background="#06b025", thickness=15)
                style.layout("CustomProgressbar.Horizontal", [('CustomProgressbar.Horizontal.trough', {'children': [('CustomProgressbar.Horizontal.pbar', {'side': 'left', 'sticky': 'ns'})], 'sticky': 'nswe'})])
                
                self.progressBar = Progressbar(self, orient="horizontal", length=194, variable=self.progress, style="CustomProgressbar.Horizontal")
                self.pauseButton = Button(self, text="Pause", width=14, state=DISABLED, command=pause)
                self.cancelButton = Button(self, text="Cancel", width=14, state=DISABLED, command=halt)

                z = self.duration.get()
                x = np.linspace(0, z, 1000)

                self.destionationLabel.place(x=10, y=8)
                self.destinationEntry.place(x=115, y=7)
                self.destinationBrowseButton.place(x=400, y=5)

                self.tempFolderLabel.place(x=10, y=38)
                self.tempFolderEntry.place(x=115, y=37)
                self.tempFolderBrowseButton.place(x=400, y=35)

                self.durationLabel.place(x=19, y=68)
                self.durationSpinbox.place(x=78, y=67)
                self.durationUnitLabel.place(x=160, y=68)
                self.fpsLabel.place(x=47, y=98)
                self.fpsSpinbox.place(x=78, y=97)
                self.resolutionLabel.place(x=10, y=128)
                self.resolutionSpinboxW.place(x=78, y=127)
                self.resolutionUnitLabel.place(x=137, y=128)
                self.resolutionSpinboxH.place(x=151, y=127)

                self.renderButton.place(x=10, y=155)
                self.progressBar.place(x=11, y=188)
                self.pauseButton.place(x=10, y=216)
                self.cancelButton.place(x=112, y=216)

                self.wm_protocol("WM_DELETE_WINDOW", self.on_close)

                self.focus_force()
                self.transient(master)
                self.mainloop()

            def on_close(self):
                if self.rendering and not messagebox.askokcancel("On-going render", "If you exit now, all progress will be lost! Click OK to exit.", icon="warning"):
                    return
                self.halt = True
                self.destroy()

            @staticmethod
            def sigmoid_derivative(x, x0, k):
                return k * np.exp(-k * (x - x0)) / (1 + np.exp(-k * (x - x0))) ** 2
            @staticmethod
            def sigmoid(x, x0, k):
                return 1 / (1 + np.exp(-k * (x - x0)))
            
            @classmethod
            def velocity(cls, x, z):
                y1_1 = cls.sigmoid(x, z / 20, 0.7)
                y2_1 = cls.sigmoid(x, z - z / 20, -0.7)
                return y1_1 + y2_1
            @classmethod
            def derivative(cls, x, z):
                y1_2 = cls.sigmoid_derivative(x, z / 20, 0.7)
                y2_2 = cls.sigmoid_derivative(x, z - z / 20, -0.7)
                return y1_2 + y2_2

            def on_durationSpinbox_change(self):
                z = self.duration.get()
                x = np.linspace(0, z, 1000)
                self.ax1.clear()

                self.ax1.plot(x, self.velocity(x, z))
                self.ax1.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
                self.ax1.set_title(f'Zoom Velocity', fontsize=9)
                self.ax1.grid(True)

                self.canvas1.draw()

            def validate_durationSpinbox(self, new_value: str):
                return new_value.isdigit() or new_value == ""

            def ask_destionation(self):
                path = filedialog.asksaveasfilename(parent=self, initialfile=datetime.now().strftime("Mandelbrot Voyage %H.%M.%S %d-%m-%y"), defaultextension='.mp4', filetypes=[('MP4 (*.mp4)', '*.mp4'), ('AVI (*.avi)', '*.avi')], title="Save the video")
                if path:
                    self.destinationEntry.delete(0, END)
                    self.destinationEntry.insert(0, path.replace('/', '\\'))
            def ask_tempfolder(self):
                path = filedialog.askdirectory(parent=self, title="Choose a temporary folder to extract frames")
                if path:
                    self.tempFolderEntry.delete(0, END)
                    self.tempFolderEntry.insert(0, path.replace('/', '\\'))
            
        video = Video(self)

    def load_lod(self):
        global lod_res
        try:
            self.changeLocationUI.reVar.set(("%.100f" % float(self.center[0]))[:102])
            self.changeLocationUI.imVar.set(("%.100f" % float(self.center[1]))[:102])
        except: pass
        last_computation = time.time_ns()
        mandelbrot_kernel[(g1, g2), (b1, b2)](self.zoom, np.array([float(x) for x in self.center]), iteration_coefficient, self.rgb_colors_lod_gpu, spectrum_gpu, initial_spectrum_gpu, brightness, spectrum_offset, inset_color, 0)
        time_elapsed = time.time_ns() - last_computation
        self.rgb_colors_lod = self.rgb_colors_lod_gpu.copy_to_host()
        self.rgb_colors_lod = gaussian_filter(self.rgb_colors_lod, sigma=blur_sigma)
        self.ax.clear()
        self.ax.imshow(self.rgb_colors_lod, extent=[-2.5, 1.5, -2, 2], interpolation="bilinear")
        self.ax.set_aspect(self.height / self.width)
        coords = [str(abs(self.center[0])), str(abs(self.center[1]))]
        fps = 1 / (time_elapsed / 10e+8)
        self.ax.text(-2.5 + (5 * (1.5 - (-2.5)) / self.width), 2 - (5 * (1.5 - (-2.5)) / self.height), ((f"{'' * 8}Re: {'-' if self.center[0] < 0 else ' '}{coords[0]}" +
                                  f"\n{'' * 8}Im: {'-' if self.center[1] < 0 else ' '}{coords[1]}") if show_coordinates else '') +
                                 (f"\n{'' * 6}Zoom:  {(4.5 / self.zoom):e}" if show_zoom else '') +
                                 (f"\nIterations:  {int(initial_iteration_count / (iteration_coefficient ** (log(self.zoom / 4.5) / log(zoom_coefficient)))):e}" if show_iteration_count else '') +
                                 (f"\nFPS: {fps}"),
                     color="white", fontfamily="monospace", fontweight=10, size=7, bbox=dict(boxstyle='square', facecolor='black', alpha=0.5), horizontalalignment='left', verticalalignment='top')
        self.canvas.draw()
        if self.dynamic_resolution.get():
            d = sum(self.fps_history[-3:]) / 3 - fps
            if (d < 10 and d > 0 or fps < 4) and len(self.fps_history) > 5:
                x = int(18.33 * fps - 133.33)
                lod_res = 60 if x < 60 else 600 if x > 600 else x
                self.rgb_colors_lod = np.empty((int(lod_res * self.height / self.width), lod_res, 3), dtype=np.uint8)
                self.rgb_colors_lod_gpu = cuda.to_device(self.rgb_colors_lod)
            self.fps_history.append(fps)
            if len(self.fps_history) > 25:
                self.fps_history.pop(0)

    def update_image(self):
        try:
            self.changeLocationUI.reVar.set(("%.100f" % float(self.center[0]))[:102])
            self.changeLocationUI.imVar.set(("%.100f" % float(self.center[1]))[:102])
        except: pass
        last_computation = time.time_ns()
        mandelbrot_kernel[(g1, g2), (b1, b2)](self.zoom, np.array([float(x) for x in self.center]), iteration_coefficient, self.rgb_colors_gpu, spectrum_gpu, initial_spectrum_gpu, brightness, spectrum_offset, inset_color, self.smooth_coloring.get())
        self.rgb_colors = self.rgb_colors_gpu.copy_to_host()
        self.rgb_colors = gaussian_filter(self.rgb_colors, sigma=blur_sigma)
        self.ax.clear()
        self.ax.imshow(rescale(self.rgb_colors, 1 / (spss_factor if self.subpixel_supersampling.get() else 1), anti_aliasing=True, channel_axis=2, order=2), extent=[-2.5, 1.5, -2, 2], interpolation="nearest")
        self.ax.set_aspect(self.height / self.width)
        coords = [str(abs(self.center[0])), str(abs(self.center[1]))]
        self.ax.text(-2.5 + (5 * (1.5 - (-2.5)) / self.width), 2 - (5 * (1.5 - (-2.5)) / self.height), ((f"{'' * 8}Re: {'-' if self.center[0] < 0 else ' '}{remove_trailing_9s(str(coords[0]))}" +
                                  f"\n{'' * 8}Im: {'-' if self.center[1] < 0 else ' '}{remove_trailing_9s(str(coords[1]))}") if show_coordinates else '') +
                                 (f"\n{'' * 6}Zoom:  {(4.5 / self.zoom):e}" if show_zoom else '') +
                                 (f"\nIterations:  {int(initial_iteration_count / (iteration_coefficient ** (log(self.zoom / 4.5) / log(zoom_coefficient)))):e}" if show_iteration_count else '') +
                                 (f"\nFPS: {1 / ((time.time_ns() - last_computation) / 10e+8)}"),
            color="white", fontfamily="monospace", fontweight=10, size=7, bbox=dict(boxstyle='square', facecolor='black', alpha=0.5), horizontalalignment='left', verticalalignment='top')
        self.canvas.draw()
        self.load_image = None

    def save_image(self):
        path = filedialog.asksaveasfilename(initialfile=datetime.now().strftime("Mandelbrot Voyage %H.%M.%S %d-%m-%y"), defaultextension='.png', filetypes=[('PNG (*.png)', '*.png'), ('JPEG (*.jpg)', '*.jpg')], title="Save the screenshot")
        if not path:
            return
        plt.imsave(path, self.rgb_colors)

    def save_loc(self):
        path = filedialog.asksaveasfilename(initialfile=datetime.now().strftime("Mandelbrot Voyage Location %H.%M.%S %d-%m-%y"), defaultextension='.loc', filetypes=[('MV Location File (*.loc)', '*.loc')], title="Save the current location")
        if not path:
            return
        with open(path, 'wb') as file:
            pickle.dump(self.center, file, pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.zoom,   file, pickle.HIGHEST_PROTOCOL)

    def load_loc(self):
        if self.load_image:
            self.after_cancel(self.load_image)
        path = filedialog.askopenfilename(filetypes=[('MV Location File (*.loc)', '*.loc'), ('All types (*.*)', '*.*')], title="Load saved location")
        if not path:
            return
        with open(path, 'rb') as file:
            self.center = pickle.load(file).copy()
            self.zoom = pickle.load(file)

        if self.use_lod.get():
            self.load_lod()
            self.load_image = self.after(1000, self.update_image)
        else:
            self.update_image()
    
    def reset_loc(self):
        if self.load_image:
            self.after_cancel(self.load_image)
        self.center = np.array([mpf("-0.4" + "0" * 180), mpf(0)])
        if self.use_lod.get():
            self.load_lod()
            self.load_image = self.after(1000, self.update_image)
        else:
            self.update_image()

    def change_loc(self):
        class ChangeLocation(Toplevel):
            def __init__(self, master: MandelbrotVoyage):
                super().__init__(master)
                self.root = master

                self.wm_title("Change location")
                self.wm_geometry("770x100")
                self.wm_resizable(height=False, width=False)

                self.reVar = StringVar(self, value=("%.100f" % float(self.root.center[0]))[:102])
                self.tr1 = self.reVar.trace_add('write', lambda *args, **kwargs: self.on_entryUpdate())
                self.imVar = StringVar(self, value=("%.100f" % float(self.root.center[1]))[:102])
                self.tr2 = self.imVar.trace_add('write', lambda *args, **kwargs: self.on_entryUpdate())

                Label(self, text="Re:").place(x=8, y=10)
                Label(self, text="Im:").place(x=8, y=41)

                self.re = Entry(self, width=103, textvariable=self.reVar, font=("Consolas", 9))
                self.re.place(x=34, y=8)
                self.im = Entry(self, width=103, textvariable=self.imVar, font=("Consolas", 9))
                self.im.place(x=34, y=39)

                self.apply  = Button(self, text="Apply",  width=20, state=DISABLED, command=self.on_apply, takefocus=0)
                self.revert = Button(self, text="Revert", width=20, state=DISABLED, command=self.on_revert, takefocus=0)
                self.cancel = Button(self, text="Cancel", width=20, command=self.on_close, takefocus=0)
                self.apply.place(x=10, y=67)
                self.revert.place(x=147, y=67)
                self.cancel.place(x=284, y=67)

                self.wm_protocol("WM_DELETE_WINDOW", lambda *args, **kwargs: self.on_close())
                self.focus_force()
                self.wm_transient(master)
            
            def on_entryUpdate(self):
                s = NORMAL if self.reVar.get() != ("%.100f" % float(self.root.center[0]))[:102] or self.imVar.get() != ("%.100f" % self.root.center[1])[:102] else DISABLED
                self.apply.configure(state=s)
                self.revert.configure(state=s)
            def on_apply(self):
                self.root.center = np.array([mpf(self.reVar.get()), mpf(self.imVar.get())])
                self.on_entryUpdate()
                if self.root.use_lod.get():
                    self.root.load_lod()
                    self.load_image = self.after(1000, self.update_image)
                else:
                    self.root.update_image()
                self.on_close()
            def on_revert(self):
                self.reVar.set(("%.100f" % float(self.root.center[0]))[:102])
                self.imVar.set(("%.100f" % float(self.root.center[1]))[:102])
                self.on_entryUpdate()
            def on_close(self):
                self.reVar.trace_remove("write", self.tr1)
                self.imVar.trace_remove("write", self.tr2)
                self.destroy()

        self.changeLocationUI = ChangeLocation(self)
        self.changeLocationUI.mainloop()
        self.changeLocationUI = None

    def center_point(self, event):
        if self.load_image:
            self.after_cancel(self.load_image)
        pixel_size = self.zoom / min(self.height, self.width)
        self.center[0] -= (self.width  / 2 - event.x) * pixel_size
        self.center[1] += (self.height / 2 - event.y) * pixel_size

        # Update the image with the new center
        if self.use_lod.get():
            self.load_lod()
            self.load_image = self.after(1000, self.update_image)
        else:
            self.update_image()

    def zoom_(self, delta):
        if self.load_image:
            self.after_cancel(self.load_image)
        if delta > 0:
            self.zoom *= zoom_coefficient ** abs(delta / 120)
        else:
            self.zoom /= zoom_coefficient ** abs(delta / 120)

        if floor(log10(abs((4.5 / self.zoom)))) >= 13:
            self.alertLabel.pack_configure(side="top", pady=10)
        else:
            self.alertLabel.pack_forget()

        if self.use_lod.get():
            self.load_lod()
            self.load_image = self.after(1000, self.update_image)
        else:
            self.update_image()

    def reset_zoom(self):
        self.zoom = 4.5
        if self.use_lod.get():
            self.load_lod()
            self.load_image = self.after(1000, self.update_image)
        else:
            self.update_image()

    def _on_mousewheel(self, event):
        self.zoom_(event.delta)

    def reset_double_click(self):
        self.double_click = False
    
    def _on_button_press(self, event):
        match event.button:
            case MouseButton.LEFT:
                if self.double_click:
                    self.center_point(event)
                else:
                    self.double_click = True
                    self.after(500, self.reset_double_click)
                self.dragging = True
                self.last_x, self.last_y = event.x, event.y
            case MouseButton.RIGHT:
                if self.load_image:
                    return
                self.info_x = [event.x, event.xdata]
                self.info_y = [event.y, event.ydata]
                self.dragging_right = True
                if self.textmovement:
                    self.textmovement.event_source.stop()
                    self.textmovement.event_source.callbacks.clear()
                self.textmovement = FuncAnimation(self.fig, self.update_pixelinfo, interval=10, blit=True)

    def update_pixelinfo(self, *args, **kwargs):
        pixel_size = self.zoom / min(self.height, self.width)
        if not self.dragging_right:
            try:
                return self.pixelinfotext,
            finally:
                self.textmovement.event_source.stop()
                self.textmovement.event_source.callbacks.clear()
        c = complex(self.center[0] - (self.width / 2 - self.info_x[0]) * pixel_size, self.center[1] + (self.height / 2 - self.info_y[0]) * pixel_size)
        z = c
        max_iters = int(initial_iteration_count / (zoom_coefficient ** (log(self.zoom / 4.5) / log(zoom_coefficient))))
        for i in range(max_iters):
            if (z.real ** 2 + z.imag ** 2) >= 4:
                a = i
                break
            z = z * z + c
        else:
            a = None
        pos = self.info_x[1] + (10 * (1.5 - (-2.5)) / self.width), self.info_y[1] - (-35 * (1.5 - (-2.5)) / self.height)
        txt = f"Re: {' ' if c.real > 0 else '-'}{remove_trailing_9s(str(abs(c.real)))}\nIm: {' ' if c.imag < 0 else '-'}{remove_trailing_9s(str(abs(c.imag)))}\nEscape-time: {a if a else 'never'}"
        if self.pixelinfotext is None:
            self.pixelinfotext = self.ax.text(*pos, txt,
                color="white", fontfamily="monospace", fontweight=10, size=7, bbox=dict(boxstyle='square', facecolor='black', alpha=0.5), horizontalalignment='left', verticalalignment='top')
        else:
            self.pixelinfotext.set_position(pos)
            self.pixelinfotext.set_text(txt)
        return [self.pixelinfotext]

    def _on_button_release(self, event):
        match event.button:
            case MouseButton.LEFT:
                self.dragging = False
                self.last_x, self.last_y = None, None
                self.ax.draw_artist(self.ax)
                self.canvas.blit(self.ax.bbox)

            case MouseButton.RIGHT:
                self.dragging_right = False
                if self.pixelinfotext:
                    self.pixelinfotext.remove()
                    self.ax.relim()
                    self.textmovement.event_source.stop()
                    self.textmovement.event_source.callbacks.clear()
                    self.pixelinfotext = None

    def _on_mouse_move(self, event):
        if self.dragging:
            if self.load_image:
                self.after_cancel(self.load_image)
            if self.last_x and self.last_y:
                dy = (event.x - self.last_x)
                dx = (self.last_y - event.y)
                w, h = self.ax.get_xlim()[1] - self.ax.get_xlim()[0], self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
                self.center -= np.array([dy / self.fig.get_dpi() / w * self.zoom, dx / self.fig.get_dpi() / h * self.zoom], dtype=np.float64)
                self.last_x, self.last_y = event.x, event.y

                if self.use_lod.get():
                    self.load_lod()
                    self.load_image = self.after(1000, self.update_image)
                else:
                    self.update_image()
        if self.dragging_right:
            self.info_x = [event.x, event.xdata]
            self.info_y = [event.y, event.ydata]

if __name__ == "__main__":
    app = MandelbrotVoyage()
    app.mainloop()