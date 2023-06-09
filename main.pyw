from tkinter import *
from tkinter import filedialog
from tkinter.ttk import *

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from matplotlib.figure import Figure

from numba import cuda, njit, prange
from datetime import datetime
from math import *
from mpmath import mpf, mp

import matplotlib.pyplot as plt
import numpy as np
import nvidia_smi, tkinter, tempfile, os, re
import moviepy.video.io.ImageSequenceClip

initial_iteration_count = 80
iteration_coefficient = 0.95

mp.dps = 50

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
def complex_number_search(device_array, search_number, result):
    thread_id = cuda.grid(1)
    if thread_id < device_array.shape[0]:
        real = device_array[thread_id, 0]
        imag = device_array[thread_id, 1]
        if real == search_number.real and imag == search_number.imag:
            result[0] = 1

@cuda.jit()
def mandelbrot_kernel(zoom, offset, max_iters, output):
    pixel_size = zoom / min(output.shape[0], output.shape[1])
    start_x, start_y = cuda.grid(2)
    grid_x, grid_y = cuda.gridsize(2)
    for i in range(start_x, output.shape[0], grid_x):
        for j in range(start_y, output.shape[1], grid_y):
            imag = (i - output.shape[0] / 2) * pixel_size - offset[0]
            real = (j - output.shape[1] / 2) * pixel_size - offset[1]
            c = complex(real, imag)

            p = mandelbrot_pixel(c, max_iters)
            output[i, j] = p

class MandelbrotExplorer(Tk):
    def __init__(self):
        super().__init__()

        self.width, self.height = 1280, 720

        self.wm_title("Mandelbrot Voyage")
        self.wm_geometry(f"{self.width}x{self.height}")
        self.configure(bg="black")

        self.fig = Figure()
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        self.ax = self.fig.add_subplot(111, aspect=1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().place(x=0, y=0, height=self.height, width=self.width)
        self.offset = np.array([0, 0], dtype=np.float64)
        self.zoom = 4.5
        self.max_iters = initial_iteration_count
        self.image = np.zeros((self.height, self.width), dtype=np.float64)
        self.image_gpu = cuda.to_device(self.image)
        self.image_lod = np.zeros((int(self.height / 4), int(self.width / 4)), dtype=np.float64)
        self.image_gpu_lod = cuda.to_device(self.image_lod)

        self.load_image = None

        self.bind("<MouseWheel>", self._on_mousewheel)

        self.canvas.mpl_connect('button_press_event', self._on_button_press)
        self.canvas.mpl_connect('button_release_event', self._on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)

        self.display = tkinter.Label(self, bg="black", fg="white")
        self.update_info()
        self.display.place(relx=0.5, rely=1.0, anchor=S, width=self.width)

        self.cmap = "CMRmap"
        self.use_lod = BooleanVar(value=False)

        self.update_image()

        self.dragging = False
        self.double_click = False
        self.resize = None
        self.last_x, self.last_y = None, None
        self.halt_on_resize = False
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
                        self.add_command(label="Create zoom video", command=self.root.make_video)
                        self.add_separator()
                        self.add_command(label="Exit", accelerator="Alt+F4", command=lambda: os._exit(0))

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
                        self.add_command(label="Reset magnification", accelerator="R", command=self.root.reset_zoom)
                
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
                self.add_cascade(label = "Magnification", menu=self.zoomMenu)
                self.add_cascade(label = "Color Scheme", menu=self.colorMenu)
                self.add_command(label = "Help", state="disabled")

        self.menuBar = menuBar(self)
        self.config(menu = self.menuBar)

        self.bind("e", lambda _: self.zoom_(+1))
        self.bind("q", lambda _: self.zoom_(-1))
        self.bind("r", lambda _: self.reset_zoom())
        self.bind("<Control_L>s", lambda _: self.save_image())
        self.bind("<F11>", lambda _: self.wm_attributes('-fullscreen', not self.attributes('-fullscreen')))

        self.wm_protocol("WM_DELETE_WINDOW", self.on_close)

        self.bind("<Configure>", self.on_resize)

        self.after(1000, self.update_info)

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
        self.image = np.zeros((self.height, self.width), dtype=np.float64)
        self.image_gpu = cuda.to_device(self.image)
        self.image_lod = np.zeros((int(self.height / 4), int(self.width / 4)), dtype=np.float64)
        self.image_gpu_lod = cuda.to_device(self.image_lod)
        self.ax.set_aspect(self.height / self.width)
        self.update_image()

    def render_video(self, config: Toplevel):
        final_zoom = self.zoom
        tempfolder = config.tempFolder.get()

        self.zoom = 4.5
        self.max_iters = initial_iteration_count

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
            self.image = np.zeros((config.h.get(), config.w.get()), dtype=np.float64)
            self.image_gpu = cuda.to_device(self.image)
        for i in range(int(fc)):
            mandelbrot_kernel[(64, 64), (32, 32)](self.zoom, self.offset, self.max_iters, self.image_gpu)
            self.image_gpu.copy_to_host(self.image)
            plt.imsave(tempfolder + f'\\{i}.png', self.image, cmap=self.cmap)
            self.zoom *= zoom_coefficient
            self.max_iters = initial_iteration_count / (iteration_coefficient ** (log(self.zoom / 4.5) / log(0.9)))
            config.progressBar.step()
            config.update()

        config.pauseButton.configure(state=DISABLED)
        config.cancelButton.configure(state=DISABLED)
        
        print([os.path.join(tempfolder, img) for img in sorted(os.listdir(tempfolder), key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) if img.endswith(".png")])

        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip([os.path.join(tempfolder, img) for img in sorted(os.listdir(tempfolder), key=lambda x: int(os.path.splitext(os.path.basename(x))[0])) if img.endswith(".png")], fps=config.fps.get())
        clip.write_videofile(config.destinationVar.get())

        change_state(NORMAL)

    def make_video(self):
        class Video(Toplevel):
            def __init__(self, master: Tk):
                super().__init__(master)

                self.root = master

                self.duration = IntVar(value=10)
                self.fps = IntVar(value=30)
                self.h = IntVar(value=master.height)
                self.w = IntVar(value=master.width)

                self.pause = False
                def pause(): self.pause = True

                self.tempFolder = StringVar(value=tempfile.gettempdir() + "\\Mandelbrot Voyage")

                self.duration.trace("w", lambda *args: self.on_durationSpinbox_change())

                self.wm_title("Make zoom video")
                self.wm_geometry(f"532x400")
                self.wm_resizable(height=False, width=False)

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

                self.destionationLabel = Label(self, text="Destionation:", foreground="red")
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
                self.cancelButton = Button(self, text="Cancel", width=14, state=DISABLED)

                z = self.duration.get()
                x = np.linspace(0, z, 1000)

                self.fig1, self.ax1 = plt.subplots(figsize=(6, 4), facecolor="#f0f0f0")
                self.ax1.plot(x, self.velocity(x, z))
                self.ax1.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
                self.ax1.set_title(f'Zoom Velocity', fontsize=9)
                self.ax1.grid(True)

                self.canvas1 = FigureCanvasTkAgg(self.fig1, master=self)
                self.canvas1.draw()
                self.canvas1.get_tk_widget().place(x=210, y=70, height=185, width=165)

                self.fig2, self.ax2 = plt.subplots(figsize=(6, 4), facecolor="#f0f0f0")
                self.ax2.plot(x, self.derivative(x, z))
                self.ax2.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
                self.ax2.set_title(f'Zoom Acceleration', fontsize=9)
                self.ax2.grid(True)

                self.canvas2 = FigureCanvasTkAgg(self.fig2, master=self)
                self.canvas2.draw()
                self.canvas2.get_tk_widget().place(x=365, y=70, height=185, width=165)

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

                self.focus_force()
                self.transient(master)
                self.mainloop()

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
                self.ax2.clear()

                self.ax1.plot(x, self.velocity(x, z))
                self.ax1.tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
                self.ax1.set_title(f'Zoom Velocity', fontsize=9)
                self.ax1.grid(True)

                self.ax2.plot(x, self.derivative(x, z))
                self.ax2.tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
                self.ax2.set_title(f'Zoom Acceleration', fontsize=9)
                self.ax2.grid(True)

                self.canvas1.draw()
                self.canvas2.draw()

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

    def update_info(self):
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)

        self.display.configure(text="Magnification: " + f"{(4.5 / self.zoom):e}" + "   Iterations: " + f"{self.max_iters:e}" + "   GPU Usage: " + str(util.gpu) + "%   Memory Usage: " + str(int(mem.used / 1048576)) + " MB")
    
        self.after(1000, self.update_info)

    def load_lod(self):
        mandelbrot_kernel[(64, 64), (32, 32)](self.zoom, self.offset, self.max_iters, self.image_gpu_lod)
        self.image_gpu_lod.copy_to_host(self.image_lod)
        self.ax.clear()
        self.ax.imshow(self.image_lod, cmap=self.cmap, extent=[-2.5, 1.5, -2, 2])
        self.ax.set_aspect(self.height / self.width)
        self.canvas.draw()

    def update_image(self):
        mandelbrot_kernel[(64, 64), (32, 32)](self.zoom, self.offset, self.max_iters, self.image_gpu)
        self.image_gpu.copy_to_host(self.image)
        self.ax.clear()
        self.ax.imshow(self.image, cmap=self.cmap, extent=[-2.5, 1.5, -2, 2])
        self.ax.set_aspect(self.height / self.width)
        self.canvas.draw()

        self.load_image = None

        # add to in_set

    def save_image(self):
        path = filedialog.asksaveasfilename(initialfile=datetime.now().strftime("Mandelbrot Voyage %H.%M.%S %d-%m-%y"), defaultextension='.png', filetypes=[('PNG (*.png)', '*.png'), ('JPEG (*.jpg)', '*.jpg')], title="Save the screenshot")
        if not path:
            return
        plt.imsave(path, self.image, cmap=self.cmap)

    def save_loc(self):
        self.zoom_m1 = self.zoom
        self.offset_m1 = self.offset.copy()

    def load_loc(self):
        self.zoom = self.zoom_m1
        self.offset = self.offset_m1.copy()
        self.max_iters = initial_iteration_count / (0.95 ** (log(self.zoom / 4.5) / log(0.9)))
        self.update_image()

    def center_point(self, event):
        dx = (event.x - self.width / 2) / self.width
        dy = (self.height / 2 - event.y) / self.height

        # adjust the offset based on the aspect ratio
        if self.width > self.height:
            aspect_ratio = self.width / self.height
            self.offset -= np.array([dy * self.zoom, dx * self.zoom * aspect_ratio], dtype=np.float64)
        else:
            aspect_ratio = self.height / self.width
            self.offset -= np.array([dy * self.zoom * aspect_ratio, dx * self.zoom], dtype=np.float64)

        self.last_x, self.last_y = event.x, event.y
        self.update_image()

    def zoom_(self, delta):
        if self.load_image:
            self.after_cancel(self.load_image)
        if delta > 0:
            self.zoom *= 0.9
            self.max_iters = int(self.max_iters / iteration_coefficient)
        else:
            self.zoom /= 0.9
            self.max_iters = max(initial_iteration_count, int(self.max_iters * iteration_coefficient))

        if self.use_lod.get():
            self.load_lod()
            self.load_image = self.after(1000, self.update_image)
        else:
            self.update_image()

    def reset_zoom(self):
        self.zoom = 4.5
        self.max_iters = initial_iteration_count
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