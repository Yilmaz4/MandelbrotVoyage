# Mandelbrot Voyage

Mandelbrot Voyage is a Python app where you can explore the Mandelbrot set with high customizability. GPU is used for calculations, so it is much faster compared to the traditional method of using the CPU.

![Screenshot 2023-11-03 235207](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/3a103353-6e5f-4f40-bb29-da16681de6f7)

The most you can zoom in is 10^14 due to the limiations of floating point numbers. I do plan to use a library to increase the precision, however the library does not support CUDA.

System requirements:

- A CUDA Compute Capability >3.5 GPU
