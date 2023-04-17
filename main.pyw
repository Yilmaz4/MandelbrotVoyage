from tkinter import *
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from numba import cuda
from datetime import datetime
from math import *

import matplotlib.pyplot as plt
import numpy as np
import nvidia_smi

cmaps = ['CMRmap','Greys_r','RdGy_r','afmhot','binary_r','bone','copper','cubehelix','flag_r','gist_earth','gist_gray','gist_heat','gist_stern','gist_yarg_r','gnuplot','gnuplot2','gray','hot','inferno','magma','nipy_spectral']

nvidia_smi.nvmlInit()

@cuda.jit(device=True)
def mandelbrot_pixel(c, max_iters):
    z = c
    for i in range(max_iters):
        if (z.real ** 2 + z.imag ** 2) >= 4:
            return i
        z = z * z + c
    return 0

@cuda.jit
def mandelbrot_kernel(zoom, offset, max_iters, output):
    pixel_size = zoom / min(output.shape[0], output.shape[1])
    start_x, start_y = cuda.grid(2)
    grid_x, grid_y = cuda.gridsize(2)
    for i in range(start_x, output.shape[0], grid_x):
        for j in range(start_y, output.shape[1], grid_y):
            imag = (i - output.shape[0] / 2) * pixel_size - offset[0]
            real = (j - output.shape[1] / 2) * pixel_size - offset[1]
            c = complex(real, imag)
            output[i, j] = mandelbrot_pixel(c, max_iters)

class MandelbrotExplorer(Tk):
    def __init__(self):
        super().__init__()

        self.size = 800

        self.wm_title("Mandelbrot Voyage")
        self.wm_geometry(f"{self.size}x{self.size}")
        self.wm_resizable(width=False, height=False)

        self.fig = Figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=0, y=0, height=800, width=800)
        self.offset = np.array([0, 0], dtype=np.float64)
        self.zoom = 4.5
        self.max_iters = 150
        self.image_lod = np.zeros((200, 200), dtype=np.float64)
        self.image = np.zeros((self.size, self.size), dtype=np.float64)
        self.image_gpu = cuda.to_device(self.image)
        self.image_gpu_lod = cuda.to_device(self.image_lod)

        self.load_image = None

        self.bind("<MouseWheel>", self._on_mousewheel)

        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        self.display = Label(self, bg="black", fg="white")
        self.update_info()
        self.display.place(x=0, y=self.size - 20, width=self.size)

        self.cmap = "CMRmap"
        self.use_lod = BooleanVar(value=False)

        self.update_image()

        self.dragging = False
        self.double_click = False
        self.last_x, self.last_y = None, None

        self.zoom_m1 = None
        self.offset_m1 = None

        class menuBar(Menu):
            def __init__(self, root):
                super().__init__(root, tearoff=0)

                class fileMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master
                        
                        self.add_command(label="Save location", accelerator="Ctrl+C", command=self.root.save_loc)
                        self.add_command(label="Load location", accelerator="Ctrl+L", command=self.root.load_loc)
                        self.add_separator()
                        self.add_command(label="Take screenshot", accelerator="Ctrl+S", command=self.root.save_image)
                        self.add_separator()
                        self.add_command(label="Exit", accelerator="Alt+F4", command=exit)

                class settingsMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master
                        
                        self.add_checkbutton(label="Load low resolution first", variable=self.root.use_lod)

                class zoomMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master

                        self.add_command(label="Zoom in",  accelerator="E", command=lambda: self.root.zoom_(+1))
                        self.add_command(label="Zoom out", accelerator="Q", command=lambda: self.root.zoom_(-1))
                        self.add_separator()
                        self.add_command(label="Reset zoom", accelerator="R", command=self.root.reset_zoom)
                
                class colorMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master
                        self.cmap = StringVar(value="hot")
                        for cmap in cmaps:
                            self.add_radiobutton(label=cmap, value=cmap, variable=self.cmap, command=self.change_cmap)

                    def change_cmap(self):
                        root.cmap = self.cmap.get()
                        root.update_image()

                
                self.fileMenu = fileMenu(self)
                self.settingsMenu = settingsMenu(self)
                self.zoomMenu = zoomMenu(self)
                self.colorMenu = colorMenu(self)

                self.add_cascade(label = "File", menu=self.fileMenu)
                self.add_cascade(label = "Settings", menu=self.settingsMenu)
                self.add_cascade(label = "Zoom", menu=self.zoomMenu)
                self.add_cascade(label = "Color Scheme", menu=self.colorMenu)
                self.add_command(label = "Help", state="disabled")

        self.menuBar = menuBar(self)
        self.config(menu = self.menuBar)

        self.bind("e", lambda _: self.zoom_(+1))
        self.bind("q", lambda _: self.zoom_(-1))
        self.bind("r", lambda _: self.reset_zoom())
        self.bind("<Control_L>s", lambda _: self.save_image())

        self.after(1000, self.update_info)

    def update_info(self):
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)

        self.display.configure(text="Zoom: " + f"{(4.5 / self.zoom):e}" + "   Iterations: " + f"{self.max_iters:e}" + "   GPU Usage: " + str(util.gpu) + "%   Memory Usage: " + str(int(mem.used / 1048576)) + " MB")
    
        self.after(1000, self.update_info)

    def load_lod(self):
        mandelbrot_kernel[(64, 64), (32, 32)](self.zoom, self.offset, self.max_iters, self.image_gpu_lod)
        self.image_gpu_lod.copy_to_host(self.image_lod)
        self.ax.clear()
        self.ax.imshow(self.image_lod, cmap=self.cmap, extent=[-2.5, 1.5, -2, 2])
        self.canvas.draw()

    def update_image(self):
        mandelbrot_kernel[(64, 64), (32, 32)](self.zoom, self.offset, self.max_iters, self.image_gpu)
        self.image_gpu.copy_to_host(self.image)
        self.ax.clear()
        self.ax.imshow(self.image, cmap=self.cmap, extent=[-2.5, 1.5, -2, 2])
        self.canvas.draw()
        
        self.load_image = None

    def save_image(self):
        path = filedialog.asksaveasfilename(initialfile=datetime.now().strftime("Mandelbrot Voyage %H:%M:%S %d-%m-%y"), defaultextension='.png', filetypes=[('PNG (*.png)', '*.png'), ('JPEG (*.jpg)', '*.jpg')], title="Save the screenshot")
        if not path:
            return
        plt.imsave(path, self.image, cmap=self.cmap)

    def save_loc(self):
        self.zoom_m1 = self.zoom
        self.offset_m1 = self.offset.copy()

    def load_loc(self):
        self.zoom = self.zoom_m1
        self.offset = self.offset_m1.copy()
        self.max_iters = 150 * 1.1 ** int(np.emath.logn(0.9, self.zoom / 4.5))
        self.update_image()

    def center_point(self, event):
        dx = (event.x - self.size / 2) / 2
        dy = (self.size / 2 - event.y) / 2

        w, h = self.ax.get_xlim()[1] - self.ax.get_xlim()[0], self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
        self.offset -= np.array([dy / self.fig.get_dpi() / w * self.zoom, dx / self.fig.get_dpi() / h * self.zoom], dtype=np.float64)
        self.last_x, self.last_y = event.x, event.y
        
        self.update_image()

    def zoom_(self, delta):
        if self.load_image:
            self.after_cancel(self.load_image)
        if delta > 0:
            self.zoom *= 0.9
            self.max_iters = int(self.max_iters * 1.1)
        else:
            self.zoom /= 0.9
            self.max_iters = max(10, int(self.max_iters / 1.1))

        if self.use_lod.get():
            self.load_lod()
            self.load_image = self.after(1000, self.update_image)
        else:
            self.update_image()

    def reset_zoom(self):
        self.zoom = 4.5
        self.max_iters = 150
        self.update_image()

    def _on_mousewheel(self, event):
        self.zoom_(event.delta)

    def reset_double_click(self):
        self.double_click = False
    
    def _on_button_press(self, event):
        if event.button == 1:
            if self.double_click:
                self.center_point(event)
            else:
                self.double_click = True
                self.after(500, self.reset_double_click)
            self.dragging = True
            self.last_x, self.last_y = event.x, event.y

    def _on_button_release(self, event):
        if event.button == 1:
            self.dragging = False
            self.last_x, self.last_y = None, None

    def _on_mouse_move(self, event):
        if self.dragging:
            if self.load_image:
                self.after_cancel(self.load_image)
            if self.last_x is not None and self.last_y is not None:
                dx = -(event.x - self.last_x)
                dy = -(self.last_y - event.y)
                w, h = self.ax.get_xlim()[1] - self.ax.get_xlim()[0], self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
                self.offset -= np.array([dy / self.fig.get_dpi() / w * self.zoom, dx / self.fig.get_dpi() / h * self.zoom], dtype=np.float64)
                self.last_x, self.last_y = event.x, event.y

                if self.use_lod.get():
                    self.load_lod()
                    self.load_image = self.after(1000, self.update_image)
                else:
                    self.update_image()

app = MandelbrotExplorer()
app.mainloop()