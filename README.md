# Mandelbrot Voyage

Mandelbrot Voyage is a program written in Python and Tkinter where you can explore the Mandelbrot set with high customizability and speed. GPU is used for calculations, so it is much faster compared to the traditional method of using the CPU.

![Screenshot 2023-11-03 235207](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/3a103353-6e5f-4f40-bb29-da16681de6f7)

## Gallery
![gallery](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/ceb944ba-4517-4526-965d-b78778bb3b88)

### Easily create a zoom video
![Screenshot 2023-11-04 180912](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/4ccf2d00-99e0-4180-bb7c-336d99620c60)
### High customizability
![Screenshot 2023-11-04 180323](https://github.com/Yilmaz4/MandelbrotVoyage/assets/77583632/97468c22-d61c-490a-9dfb-5125f3518070)



The most you can zoom in is 10^14 due to the limiations of floating point numbers. I do plan to use a library to increase the precision, however the library does not support CUDA.

System requirements:

- A CUDA Compute Capability >3.5 GPU
