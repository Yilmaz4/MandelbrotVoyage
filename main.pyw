from tkinter import *
from tkinter import filedialog, messagebox
TkLabel = Label
from tkinter.ttk import *

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from numba import cuda
from datetime import datetime
from math import *
from mpmath import mpf, mp, mpc
from scipy.ndimage import gaussian_filter
from typing import *

import matplotlib.pyplot as plt
import numpy as np
import nvidia_smi, tkinter, tempfile, os
import pickle, re, concurrent.futures
import moviepy.video.io.ImageSequenceClip
import time

initial_iteration_count = 80

iteration_coefficient = 0.96
blur_sigma = 0.0
brightness = 6
spectrum_offset = 0

zoom_coefficient = 0.9

show_coordinates = True
show_zoom = True
show_iteration_count = True

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

nvidia_smi.nvmlInit()

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
        self.wm_geometry("352x300")
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
        mandelbrot_kernel[(5, 5), (32, 32)](self.root.zoom, np.array([float(x) for x in self.root.center]), self.coefficient / 10e+6, self.root.preview_gpu, spectrum_gpu, initial_spectrum_gpu, int(self.brightness / 10e+3), self.spectrum_offset)
        self.root.preview_gpu.copy_to_host(self.root.preview)
        self.ax.clear()
        self.ax.imshow(gaussian_filter(self.root.preview, sigma=self.blur_sigma / 10e+3), extent=[-2.5, 1.5, -2, 2])
        self.canvas.draw()
    
    def on_exit(self):
        self.destroy()

@cuda.jit(device=True)
def mandelbrot_pixel(c, max_iters):
    z: complex = c
    for i in range(max_iters):
        if (z.real ** 2 + z.imag ** 2) >= 4:
            return i
        z = z * z + c
    return 0

@cuda.jit
def mandelbrot_kernel(zoom, center, coefficient, output, spectrum, initial_spectrum, brightness, spectrum_offset):
    max_iters = initial_iteration_count / (coefficient ** (log(zoom / 4.5) / log(zoom_coefficient)))
    pixel_size = zoom / min(output.shape[0], output.shape[1])
    start_x, start_y = cuda.grid(2)
    grid_x, grid_y = cuda.gridsize(2)

    x_center, y_center = center
    x_offset = x_center - output.shape[1] / 2 * pixel_size
    y_offset = y_center - output.shape[0] / 2 * pixel_size

    for i in range(start_x, output.shape[0], grid_x):
        for j in range(start_y, output.shape[1], grid_y):
            imag = (i * pixel_size + y_offset)
            real = (j * pixel_size + x_offset)
            c = complex(real, imag)

            p = mandelbrot_pixel(c, max_iters)
            for c, k in zip(spectrum[(p * brightness - 255) % len(spectrum)] if p * brightness >= 256 else initial_spectrum[(p * brightness)], range(3)):
                output[i, j, k] = c

mp.dps = 200

def mandelbrot_pixel_cpu(c, max_iters):
    z = c
    for i in range(max_iters):
        if abs(z) >= 2:
            return i
        z = z * z + c
    return 0

def calculate_mandelbrot_row(args):
    row, zoom, center, coefficient, spectrum, initial_spectrum, h, w, brightness = args
    max_iters = initial_iteration_count / (coefficient ** (log(zoom / 4.5) / log(zoom_coefficient)))
    image_row = np.empty((1, w, 3), dtype=np.uint8)

    pixel_size = mpf(zoom) / mpf(min(h, w))
    x_center, y_center = center
    x_offset = x_center - w / 2 * pixel_size
    y_offset = y_center - h / 2 * pixel_size

    for j in range(w):
        imag = mpf(row * pixel_size + y_offset)
        real = mpf(j * pixel_size + x_offset)
        c = mpc(real, imag)

        p = mandelbrot_pixel_cpu(c, int(max_iters))
        for c, k in zip(spectrum[(p * brightness - 255) % len(spectrum)] if p * brightness >= 256 else initial_spectrum[(p * brightness)], range(3)):
            image_row[0, j, k] = c
    return row, image_row

def calculate_mandelbrot_rows(args):
    start_row, end_row, zoom, center, h, w = args
    results = []
    for row in range(start_row, end_row):
        results.append(calculate_mandelbrot_row((row, zoom, center, iteration_coefficient, spectrum, initial_spectrum, h, w, brightness))[1])
    return start_row, results

class MandelbrotVoyage(Tk):
    def __init__(self):
        super().__init__()

        self.width, self.height = 600, 600

        self.wm_title("Mandelbrot Voyage")
        self.wm_geometry(f"{self.width}x{self.height}")
        self.configure(bg="black")

        self.fig = Figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, aspect=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=0, y=0, height=self.height, width=self.width)
        self.center = np.array([mpf(0), mpf(0)])
        self.zoom = 4.5

        self.custom_coefficient = 0
        self.custom_blur_sigma = 0
        self.custom_brightness = 0

        self.preview = np.empty((215, 215, 3), dtype=np.uint8)
        self.preview_gpu = cuda.to_device(self.preview)

        self.rgb_colors = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.rgb_colors_gpu = cuda.to_device(self.rgb_colors)

        self.rgb_colors_lod = np.empty((int(200 * self.height / self.width), 200, 3), dtype=np.uint8)
        self.rgb_colors_lod_gpu = cuda.to_device(self.rgb_colors_lod)

        self.update_image()

        self.load_image = None

        self.bind("<MouseWheel>", self._on_mousewheel)

        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        self.alertLabel = tkinter.Label(self, bg="black", fg="red",
            text="Higher Precision Required - Use \"Render with CPU\"")

        self.use_lod = BooleanVar(value=False)

        self.dragging = False
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
                                self.iter = DoubleVar(value=0.96)

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
                                self.blur = DoubleVar(value=0.96)

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
                                self.brightness = DoubleVar(value=0.96)

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
                                Config(self.root, self, "color complexity", self.brightness, brightness * 10e+3, 27.5 * 10e+3, 10e+3, True, (iteration_coefficient, blur_sigma, None, spectrum_offset), apply)

                        self.iterMenu = iterMenu(self)
                        self.blurMenu = blurMenu(self)
                        self.brightness = brightnessMenu(self)

                        self.spectrum_offset_var = IntVar(value=0)

                        self.add_cascade(label="Iteration count", menu=self.iterMenu)
                        self.add_cascade(label="Bluriness", menu=self.blurMenu)
                        self.add_cascade(label="Color complexity", menu=self.brightness)
                        self.add_separator()
                        self.add_checkbutton(label="Load low resolution first", variable=self.root.use_lod)

                class zoomMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master

                        self.add_command(label="Zoom in",  accelerator="E", command=lambda: self.root.zoom_(+1))
                        self.add_command(label="Zoom out", accelerator="Q", command=lambda: self.root.zoom_(-1))
                        self.add_separator()
                        self.add_command(label="Reset magnification", accelerator="R", command=self.root.reset_zoom)

                class locationMenu(Menu):
                    def __init__(self, master: menuBar):
                        super().__init__(master, tearoff=0)
                        self.root = self.master.master

                        self.add_command(label="Save location", accelerator="Ctrl+C", command=self.root.save_loc)
                        self.add_command(label="Load location", accelerator="Ctrl+L", command=self.root.load_loc)
                        self.add_separator()
                        self.add_command(label="Change location", command=self.root.change_loc)
                
                self.fileMenu = fileMenu(self)
                self.settingsMenu = settingsMenu(self)
                self.zoomMenu = zoomMenu(self)
                self.locationMenu = locationMenu(self)

                self.add_cascade(label = "File", menu=self.fileMenu)
                self.add_cascade(label = "Edit", menu=self.settingsMenu)
                self.add_cascade(label = "Zoom", menu=self.zoomMenu)
                self.add_cascade(label = "Location", menu=self.locationMenu)
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

        self.bind("e", lambda _: self.zoom_(+1))
        self.bind("q", lambda _: self.zoom_(-1))
        self.bind("r", lambda _: self.reset_zoom())
        self.bind("<Control_L>s", lambda _: self.save_image())
        self.bind("<F11>", lambda _: self.wm_attributes('-fullscreen', not self.attributes('-fullscreen')))

        self.wm_protocol("WM_DELETE_WINDOW", self.on_close)

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
        self.resize = None
        self.canvas.get_tk_widget().place_forget()
        self.canvas.get_tk_widget().place(x=0, y=0, height=self.height, width=self.width)

        self.rgb_colors = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self.rgb_colors_gpu = cuda.to_device(self.rgb_colors)

        self.rgb_colors_lod = np.empty((int(200 * self.height / self.width), 200, 3), dtype=np.uint8)
        self.rgb_colors_lod_gpu = cuda.to_device(self.rgb_colors_lod)

        self.ax.set_aspect(self.height / self.width)
        self.update_image()

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
            #if floor(log10(abs((4.5 / self.zoom)))) <= 13:
            if True:
                mandelbrot_kernel[(5, 5), (32, 32)](self.zoom, np.array([float(x) for x in self.center]), iteration_coefficient, self.rgb_colors_gpu, spectrum_gpu, initial_spectrum_gpu, brightness, spectrum_offset)
                self.rgb_colors = self.rgb_colors_gpu.copy_to_host()
            else:
                num_threads = 16
                rows_per_thread = self.rgb_colors.shape[0] // num_threads
                extra_rows = self.rgb_colors.shape[0] % num_threads

                with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                    futures = []
                    for i in range(num_threads):
                        start_row = i * rows_per_thread
                        if i < extra_rows:
                            rows_for_thread = rows_per_thread + 1
                        else:
                            rows_for_thread = rows_per_thread

                        end_row = start_row + rows_for_thread
                        futures.append(executor.submit(self.calculate_mandelbrot_rows, start_row, end_row))

                    concurrent.futures.wait(futures)

            self.ax.clear()
            self.ax.imshow(gaussian_filter(self.rgb_colors, sigma=blur_sigma), extent=[-2.5, 1.5, -2, 2])
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
        mandelbrot_kernel[(5, 5), (32, 32)](self.zoom, np.array([float(x) for x in self.center]), iteration_coefficient, self.rgb_colors_lod_gpu, spectrum_gpu, initial_spectrum_gpu, brightness, spectrum_offset)
        self.rgb_colors_lod = self.rgb_colors_lod_gpu.copy_to_host()
        self.ax.clear()
        self.rgb_colors_lod = gaussian_filter(self.rgb_colors_lod, sigma=blur_sigma)
        self.ax.imshow(self.rgb_colors_lod, extent=[-2.5, 1.5, -2, 2])
        self.ax.set_aspect(self.height / self.width)
        self.canvas.draw()

    def update_image(self):
        try:
            self.changeLocationUI.reVar.set(("%.100f" % float(self.center[0]))[:102])
            self.changeLocationUI.imVar.set(("%.100f" % float(self.center[1]))[:102])
        except: pass
        mandelbrot_kernel[(5, 5), (32, 32)](self.zoom, np.array([float(x) for x in self.center]), iteration_coefficient, self.rgb_colors_gpu, spectrum_gpu, initial_spectrum_gpu, brightness, spectrum_offset)
        self.rgb_colors = self.rgb_colors_gpu.copy_to_host()
        self.ax.clear()
        self.rgb_colors = gaussian_filter(self.rgb_colors, sigma=blur_sigma)
        self.ax.imshow(self.rgb_colors, extent=[-2.5, 1.5, -2, 2])
        self.ax.set_aspect(self.height / self.width)
        coords = [str(abs(self.center[0])), str(abs(self.center[1]))]
        self.ax.text(-2.5, 2, ((f"{'' * 8}Re: {'-' if self.center[0] < 0 else ' '}{coords[0]}\n" +
                                  f"{'' * 8}Im: {'-' if self.center[1] < 0 else ' '}{coords[1]}\n") if show_coordinates else '') +
                                 (f"{'' * 6}Zoom:  {(4.5 / self.zoom):e}\n" if show_zoom else '') +
                                 (f"Iterations:  {int(initial_iteration_count / (iteration_coefficient ** (log(self.zoom / 4.5) / log(zoom_coefficient)))):e}" if show_iteration_count else ''),
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
        path = filedialog.askopenfilename(filetypes=[('MV Location File (*.loc)', '*.loc'), ('All types (*.*)', '*.*')], title="Load saved location")
        if not path:
            return
        with open(path, 'rb') as file:
            self.center = pickle.load(file).copy()
            self.zoom = pickle.load(file)

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
                self.root.center = np.array([float(self.reVar.get()), float(self.imVar.get())], dtype=np.float64)
                self.on_entryUpdate()
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
        pixel_size = self.zoom / min(self.height, self.width)
        self.center[0] -= (self.width  / 2 - event.x) * pixel_size
        self.center[1] += (self.height / 2 - event.y) * pixel_size

        # Update the image with the new center
        self.update_image()

    def zoom_(self, delta):
        if self.load_image:
            self.after_cancel(self.load_image)
        if delta > 0:
            self.zoom *= zoom_coefficient
        else:
            self.zoom /= zoom_coefficient

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

if __name__ == "__main__":
    app = MandelbrotVoyage()
    app.mainloop()