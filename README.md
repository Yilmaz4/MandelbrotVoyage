# Mandelbrot Voyage

Mandelbrot Voyage is a program written in Python and Tkinter where you can explore the Mandelbrot set with high customizability and speed. GPU is used for calculations, so it is much faster compared to the traditional method of using the CPU.

![Screenshot 2023-11-03 235207](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/3a103353-6e5f-4f40-bb29-da16681de6f7)

## Gallery
![Mandelbrot Voyage 23 59 57 07-12-23](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/83303d10-3b54-4480-b553-c87e3f743e56)
![Mandelbrot Voyage 00 09 28 08-12-23](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/89f0b3b8-730e-45b7-8206-7b66d263a6be)



### Easily create a zoom video
![video](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/bfa78158-c7f9-4550-823c-28d9ab447d74)
### High customizability
![customization](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/2cf1b83c-66f0-4873-96e6-c31070d6715f)

The most you can zoom in is 10^14 due to the limiations of floating point numbers. I do plan to use a library to increase the precision, however the library does not support CUDA.

System requirements:

- A CUDA Compute Capability >3.5 GPU
