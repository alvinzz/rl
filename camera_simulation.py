import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imshow, imresize

PATTERN_LEN = 8.8
CAMERA_RESOLUTION = (2448, 3264)
FOCAL_LEN = 2822 # based on a 4.15mm focal length, a 3264x2448 picture resolution, and a 4.8x3.6mm sensor size
PATTERN_INTERNAL_LEN = PATTERN_LEN * 250/330
PATTERN = imread('pattern.png', flatten=True)

def get_image_plane(pitch, yaw, roll, xstride=1, ystride=1):
    pitch, yaw, roll = np.pi / 180 * pitch, np.pi / 180 * yaw, np.pi / 180 * roll
    X = np.empty((CAMERA_RESOLUTION[1] // ystride + 1, CAMERA_RESOLUTION[0] // xstride + 1))
    Y = np.empty((CAMERA_RESOLUTION[1] // ystride + 1, CAMERA_RESOLUTION[0] // xstride + 1))
    Z = np.empty((CAMERA_RESOLUTION[1] // ystride + 1, CAMERA_RESOLUTION[0] // xstride + 1))
    for y_pixel in range(-CAMERA_RESOLUTION[1] // 2, CAMERA_RESOLUTION[1] // 2 + 1, ystride):
        for x_pixel in range(-CAMERA_RESOLUTION[0] // 2, CAMERA_RESOLUTION[0] // 2 + 1, xstride):
            # apply roll
            rolled_x_pixel, rolled_y_pixel = y_pixel * -np.sin(roll) + x_pixel * np.cos(roll), \
                y_pixel * np.cos(roll) + x_pixel * np.sin(roll)
            image_plane_x = rolled_x_pixel
            image_plane_y = FOCAL_LEN
            image_plane_z = rolled_y_pixel
            # apply pitch
            image_plane_z, image_plane_y = image_plane_y * -np.sin(pitch) + image_plane_z * np.cos(pitch), \
                image_plane_y * np.cos(pitch) + image_plane_z * np.sin(pitch)
            # apply yaw
            image_plane_x, image_plane_y = image_plane_y * -np.sin(yaw) + image_plane_x * np.cos(yaw), \
                image_plane_y * np.cos(yaw) + image_plane_x * np.sin(yaw)

            X[(y_pixel + CAMERA_RESOLUTION[1] // 2) // ystride][(x_pixel + CAMERA_RESOLUTION[0] // 2) // xstride]                 = image_plane_x
            Y[(y_pixel + CAMERA_RESOLUTION[1] // 2) // ystride][(x_pixel + CAMERA_RESOLUTION[0] // 2) // xstride]                 = image_plane_y
            Z[(y_pixel + CAMERA_RESOLUTION[1] // 2) // ystride][(x_pixel + CAMERA_RESOLUTION[0] // 2) // xstride]                 = image_plane_z
    return X, Y, Z

# At pitch, yaw, roll = 0, 0, 0, the image plane is assumed to be perpendicular to the y-axis.
def get_view(x, y, z, pitch, yaw, roll, xstride=10, ystride=10):
    result = 255 * np.ones((CAMERA_RESOLUTION[1] // ystride, CAMERA_RESOLUTION[0] // xstride))
    X, Y, Z = get_image_plane(pitch, yaw, roll, xstride, ystride)
    for y_pixel in range(0, CAMERA_RESOLUTION[1] // ystride):
        for x_pixel in range(0, CAMERA_RESOLUTION[0] // xstride):
            if Z[y_pixel][x_pixel] != 0:
                x_coord = x + X[y_pixel][x_pixel] / Z[y_pixel][x_pixel] * -z
                y_coord = y + Y[y_pixel][x_pixel] / Z[y_pixel][x_pixel] * -z
                if x_coord >= -PATTERN_LEN / 2 and y_coord >= -PATTERN_LEN / 2 \
                    and x_coord < PATTERN_LEN / 2 and y_coord < PATTERN_LEN / 2:
                    result[y_pixel][x_pixel] \
                        = PATTERN[int((y_coord + PATTERN_LEN / 2) * PATTERN.shape[1] / PATTERN_LEN)] \
                        [int((x_coord + PATTERN_LEN / 2) * PATTERN.shape[0] / PATTERN_LEN)]
    return result

# Converts a point's position in world coordinates to coordinates in the camera's image plane.
def world_to_im_coordinates(x, y, z, xc, yc, zc, pitch, yaw, roll, xstride=10, ystride=10):
    pitch, yaw, roll = np.pi / 180 * pitch, np.pi / 180 * yaw, np.pi / 180 * roll
    # Shift coordinates s.t. the camera is at the origin.
    x, y, z = x - xc, y - yc, z - zc
    # Un-yaw the camera, and find equivalent coordinates of the point.
    x, y = y * -np.sin(-yaw) + x * np.cos(-yaw), y * np.cos(-yaw) + x * np.sin(-yaw)
    # Un-pitch the camera, and find equivalent coordinates of the point.
    z, y = y * -np.sin(-pitch) + z * np.cos(-pitch), y * np.cos(-pitch) + z * np.sin(-pitch)
    # Un-roll the camera, and find equivalent coordinates of the point.
    x, z = z * -np.sin(-roll) + x * np.cos(-roll), z * np.cos(-roll) + x * np.sin(-roll)
    return x / y * FOCAL_LEN // xstride, z / y * FOCAL_LEN // ystride

if __name__ == "__main__":
    # plt.imshow(PATTERN, cmap='gray')
    # plt.show()

    # Plot one view and the bottom-left corner of the pattern.
    plt.figure(1)
    plt.subplot(121)
    projection = get_view(10*np.cos(45*np.pi/180)*np.sin(30*np.pi/180)/np.sin(45*np.pi/180), -10*np.cos(45*np.pi/180)*np.cos(30*np.pi/180)/np.sin(45*np.pi/180), 10, 45, 30, 10)
    plt.imshow(projection, cmap='gray')
    x1, y1 = world_to_im_coordinates(-4.4, -4.4, 0, 10*np.cos(45*np.pi/180)*np.sin(30*np.pi/180)/np.sin(45*np.pi/180), -10*np.cos(45*np.pi/180)*np.cos(30*np.pi/180)/np.sin(45*np.pi/180), 10, 45, 30, 10)
    x1 += CAMERA_RESOLUTION[0] // 2 // 10
    y1 += CAMERA_RESOLUTION[1] // 2 // 10
    plt.plot(x1, y1, marker='o', color='b')

    # Plot another view, which is the same except that the camera has been rotated 90deg about the z-axis.
    plt.subplot(122)
    projection = get_view(-10*np.cos(45*np.pi/180)*np.cos(30*np.pi/180)/np.sin(45*np.pi/180), -10*np.cos(45*np.pi/180)*np.sin(30*np.pi/180)/np.sin(45*np.pi/180), 10, 45, -60, 10)
    plt.imshow(projection, cmap='gray')
    x1, y1 = world_to_im_coordinates(-4.4, -4.4, 0, -10*np.cos(45*np.pi/180)*np.cos(30*np.pi/180)/np.sin(45*np.pi/180), -10*np.cos(45*np.pi/180)*np.sin(30*np.pi/180)/np.sin(45*np.pi/180), 10, 45, -60, 10)
    x1 += CAMERA_RESOLUTION[0] // 2 // 10
    y1 += CAMERA_RESOLUTION[1] // 2 // 10
    plt.plot(x1, y1, marker='o', color='b')
    plt.show()
