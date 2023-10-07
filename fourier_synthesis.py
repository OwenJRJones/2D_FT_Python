# Fourier Synthesis

import numpy as np
import matplotlib.pyplot as plt

IMAGE_FILENAME = "images/Earth.png"

# Functiojn to calculate FT
def calculate_2dft(inputs):
    ft = np.fft.ifftshift(inputs)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)

# Function to calculate inverse FT
def calculate_2dift(inputs):
    ift = np.fft.ifftshift(inputs)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

# Function to calculate distance from centre
def calc_dist_from_centre(coords, centre):
    # Distance from cente is sqrt(x^2 + y^2)
    return np.sqrt(
        (coords[0] - centre)**2 + (coords[1] - centre)**2)

# Function to find symetrical pairs of coordinates
def find_sym_coords(coords, centre):
    return (
        centre + (centre - coords[0]), centre + (centre - coords[1]))

# Function to display plots
def display_plots(individual_grating, reconstruction, idx):
    plt.subplot(121)
    plt.title('Grating')
    plt.imshow(individual_grating)
    plt.axis("off")
    plt.subplot(122)
    plt.title('Running Result')
    plt.imshow(reconstruction)
    plt.axis("off")
    plt.suptitle(f"Terms: {idx}")
    plt.pause(0.01)

# Read and process the image
image = plt.imread(IMAGE_FILENAME)
image = image[:, :, :3].mean(axis=2) # Convert to grayscale

# Array dimensions (square array) and centre is only 1 pixel
# Use smallest of the dimensions and ensure it's odd
array_size = min(image.shape) - 1 + min(image.shape) % 2

# Crop image so it's square
image = image[:array_size, :array_size]

# Find centre point
centre = int((array_size - 1) / 2)

# Get all coordinate pairs for left half of array plus centre column
coords_left_half = (
    (x, y) for x in range(array_size) for y in range(centre + 1))

# Sort coorinates based on distance from centre
coords_left_half = sorted(
    coords_left_half, key=lambda x: calc_dist_from_centre(x, centre))

plt.set_cmap("gray")

ft = calculate_2dft(image)

# Show grayscale image and its FT
plt.subplot(121)
plt.imshow(image)
plt.axis("off")

plt.subplot(122)
plt.imshow(np.log(abs(ft)))
plt.axis("off")
plt.pause(2)

# Reconstruct image
fiq = plt.figure()

# Step 1: set up emtpy arrays for final image and ind. gratings
rec_image = np.zeros(image.shape)
individual_grating = np.zeros(image.shape, dtype="complex")
idx = 0

# All steps displayed until limit is hit, then skip and use step
display_all_until = 20
display_step = 10
next_dsiplay = display_all_until + display_step

# Step 2
for coords in coords_left_half:
    # Central column: only include if points in top half of
    # centre column
    if not (coords[1] == centre and coords[0] > centre):
        idx += 1
        symm_coords = find_sym_coords(coords, centre)

    # Step 3
    # Copy values from FT into ind. grating fo rthe pair
    # of points in current iteration
    individual_grating[coords] = ft[coords]
    individual_grating[symm_coords] = ft[symm_coords]

    # Step 4
    # Calculate inverse FT to give the reconstructed
    # grating and add it to the reconstructed image
    rec_grating = calculate_2dift(individual_grating)
    rec_image += rec_grating

    # Clear ind. grating array for next iteration
    individual_grating[coords] = 0
    individual_grating[symm_coords] = 0

    # Don't display every step
    if idx < display_all_until or idx == next_dsiplay:
        if idx > display_all_until:
            next_dsiplay += display_step
            # Increase display step as iteration goes on
            display_step += 10
        display_plots(rec_grating, rec_image, idx)

plt.show()
