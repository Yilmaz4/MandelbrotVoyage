cimport cython
import numpy as np
cimport numpy as cnp

cnp.import_array()

DTYPE = np.uint8

cdef int mandelbrot_pixel_cpu(complex c, int max_iters):
    cdef complex z = c
    cdef int i
    for i in range(max_iters):
        if z.imag ** 2 + z.real ** 2 > 4:
            return i
        z = z * z + c
    return 0

cpdef list calculate_mandelbrot_row(list args):
    cdef int row = args[0]
    cdef double zoom = args[1]
    cdef cnp.ndarray offset = args[2]
    cdef int max_iters = args[3]
    cdef cnp.ndarray spectrum = args[4]
    cdef cnp.ndarray initial_spectrum = args[5]
    cdef int h = args[6]
    cdef int w = args[7]
    cdef double brightness = args[8]

    cdef cnp.ndarray image_row = np.empty((1, w, 3), dtype=DTYPE)

    cdef double pixel_size = (zoom) / (min(h, w))
    cdef int j

    cdef double imag
    cdef double real
    cdef int p
    cdef int c
    cdef int k

    for j in range(w):
        imag = ((row - h / 2) * pixel_size - offset[0])
        real = ((j - w / 2) * pixel_size - offset[1])

        p = mandelbrot_pixel_cpu(complex(real, imag), max_iters)
        
        for c, k in zip((spectrum[int((p * brightness - 255) % len(spectrum))] if ((p * brightness) >= 256) else initial_spectrum[int(p * brightness)]), range(3)):
            image_row[0, j, k] = c

    return [row, image_row]