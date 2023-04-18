# Mandelbrot Voyage

Mandelbrot Voyage is a yet another program where you can locally explore the Mandelbrot set. It uses your GPU to draw the set, so it is much faster compared to the traditional method of using the CPU.
![image](https://user-images.githubusercontent.com/77583632/232304758-a1965c1f-1d7f-4bc0-9fbb-1c7ece32dd23.png)

The Mandelbrot set is calculated in real time as you zoom in/out or move the set around. The following video was recorded with an RTX 3070 and Ryzen 7 5800X.

![](https://github.com/Yilmaz4/MandelbrotVoyage/blob/main/ezgif-4-23ea85e69c.gif)

The most you can zoom in is 10^14 due to the limiations of floating point numbers. I do plan to use a library to increase the precision but I'm kind of a slow learner...

System requirements:

- A CUDA Compute Capability >3.5 GPU

- Minimum 200 MB VRAM
