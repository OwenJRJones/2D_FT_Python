# Demo of how to produce gratings in Python

# 1D Sin
import numpy as np
import matplotlib.pyplot as plt

# Create array from -500 to 500 to represent the
# x-axis using steps of 1 - this array has 1001 elements
x = np.arange(-500, 501, 1)

# Wavelength will be 200 units long - changing this will
# alter the frequency of the wave
WAVELENGTH = 200

# y = sin(2 * pi * x / lambda) where lambda = wavelength
y = np.sin(2 * np.pi * x / WAVELENGTH)

# Show sinewave
plt.plot(x,y)
plt.title("Sinewave")
plt.show()

# --------------------------------------------

# 2D Sinusoidal grating
X, Y = np.meshgrid(x, x)

grating = np.sin(2 * np.pi * X / WAVELENGTH)

# Change colour map to grayscale
plt.set_cmap("gray")

# Show grating
plt.imshow(grating)
plt.title("2D Sinusoidal Grating")
plt.show()

# --------------------------------------------

# Rotate 2D grating by changing the axes

# pi/9 radians = 20 degrees
angle = np.pi / 9

grating = np.sin(
    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / WAVELENGTH
)

# Change colour map to grayscale
plt.set_cmap("gray")

# Show grating
plt.imshow(grating)
plt.title("2D Rotated Grating")
plt.show()

# --------------------------------------------

# Calculating the Fourier Transform

# Set angle back to 0
angle = 0

grating = np.sin(
    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / WAVELENGTH
)

plt.set_cmap("gray")

# Subplot to show two plots in same figure
plt.subplot(121)
plt.title('Sinusoidal Grating')
plt.imshow(grating)

# Calculate the Fourier transform of grating
ft = np.fft.ifftshift(grating)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

plt.subplot(122)
plt.imshow(abs(ft))
plt.xlim([480, 520])
plt.ylim([520, 480])
plt.title('Fourier Transfrom')
plt.show()

# --------------------------------------------

# Set angle back to 20 degrees and display plots
angle = np.pi / 9

grating = np.sin(
    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / WAVELENGTH
)

plt.set_cmap("gray")

# Subplot to show two plots in same figure
plt.subplot(121)
plt.title('Grating')
plt.imshow(grating)

# Calculate the Fourier transform of grating
ft = np.fft.ifftshift(grating)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

plt.subplot(122)
plt.imshow(abs(ft))
plt.xlim([480, 520])
plt.ylim([520, 480])
plt.title('Fourier Transform')
plt.show()

# --------------------------------------------

# Adding more than one grating
WAVELENGTH_1 = 200
angle_1 = 0

grating_1 = np.sin(
    2*np.pi*(X*np.cos(angle_1) + Y*np.sin(angle_1)) / WAVELENGTH_1
)

WAVELENGTH_2 = 100
angle_2 = np.pi/4

grating_2 = np.sin(
    2*np.pi*(X*np.cos(angle_2) + Y*np.sin(angle_2)) / WAVELENGTH_2
)

plt.set_cmap("gray")
plt.subplot(121)
plt.title('Grating 1')
plt.imshow(grating_1)

plt.subplot(122)
plt.imshow(grating_2)
plt.title('Grating 2')
plt.show()

gratings = grating_1 + grating_2

# Calculate the Fourier transform of the sum of the gratings
ft = np.fft.ifftshift(gratings)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

plt.set_cmap("gray")
plt.subplot(121)
plt.title('Sum of Gratings')
plt.imshow(gratings)

plt.subplot(122)
plt.imshow(abs(ft))
plt.xlim([480, 520])
plt.ylim([520, 480])
plt.title('Fourier Transform')
plt.show()

# --------------------------------------------

# Add even more gratings
amplitudes = 0.5, 0.25, 1, 0.75, 1
wavelengths = 200, 100, 250, 300, 60
angles = 0, np.pi / 4, np.pi / 9, np.pi / 2, np.pi / 12
gratings = np.zeros(X.shape)

# Loop through values to create 5 gratings
for amp, w_len, angle in zip(amplitudes, wavelengths, angles):
    gratings += amp * np.sin(
        2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / w_len
    )

# Add constant term to represent the backgruind of an image if desired
gratings += 1.25

# Calculate Fourier transform of the sum of the gratings
ft = np.fft.ifftshift(gratings)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

plt.set_cmap("gray")
plt.subplot(121)
plt.title('Sum of Gratings')
plt.imshow(gratings)

plt.subplot(122)
plt.imshow(abs(ft))
plt.xlim([480, 520])
plt.ylim([520, 480])
plt.title('Fourier Transform')
plt.show()

# --------------------------------------------

# Demo of how to find inverse of a FT

WAVELENGTH = 100
angle = np.pi/9

grating = np.sin(
    2*np.pi*(X*np.cos(angle) + Y*np.sin(angle)) / WAVELENGTH
)

plt.set_cmap("gray")
plt.subplot(131)
plt.title('Original Grating')
plt.imshow(grating)
plt.axis("off")

# Calculate the Fourier transform of the grating
ft = np.fft.ifftshift(grating)
ft = np.fft.fft2(ft)
ft = np.fft.fftshift(ft)

plt.subplot(132)
plt.title('FT of Original')
plt.imshow(abs(ft))
plt.axis("off")
plt.xlim([480, 520])
plt.ylim([520, 480])

# Calculate the inverse Fourier transform of the Fourier transform
ift = np.fft.ifftshift(ft)
ift = np.fft.ifft2(ift)
ift = np.fft.fftshift(ift)
ift = ift.real

plt.subplot(133)
plt.title('Inverse FT of FT')
plt.imshow(ift)
plt.axis("off")
plt.show()
