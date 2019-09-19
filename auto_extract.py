import cv2
import numpy as np
import argparse
import math
import os

def crop_panorama_image(img, theta=0.0, phi=0.0, res_x=512, res_y=512, fov=60.0, debug=False):
    img_x = img.shape[0]
    img_y = img.shape[1]

    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi

    fov_x = fov
    aspect_ratio = res_y * 1.0 / res_x
    half_len_x = math.tan(fov_x / 180 * math.pi / 2)
    half_len_y = aspect_ratio * half_len_x

    pixel_len_x = 2 * half_len_x / res_x
    pixel_len_y = 2 * half_len_y / res_y

    map_x = np.zeros((res_x, res_y), dtype=np.float32)
    map_y = np.zeros((res_x, res_y), dtype=np.float32)

    axis_y = math.cos(theta)
    axis_z = math.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    theta_rot_mat = np.array([[1, 0, 0], \
            [0, cos_theta, -sin_theta], \
            [0, sin_theta, cos_theta]], dtype=np.float32)

    # phi rotation matrix
    cos_phi = math.cos(phi)
    sin_phi = -math.sin(phi)
    phi_rot_mat = np.array([[cos_phi + axis_x**2 * (1 - cos_phi), \
            axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
            axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
            cos_phi + axis_y**2 * (1 - cos_phi), \
            axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
            axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
            cos_phi + axis_z**2 * (1 - cos_phi)]], dtype=np.float32)

    map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
    map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
    map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
            np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    vec_len = np.sqrt(np.sum(ind**2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + math.pi/2) / math.pi * img_x
    map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    if debug:
        for x in range(res_x):
            for y in range(res_y):
                print ('(%.2f, %.2f)\t' % (map_x[x, y], map_y[x, y]) )

    return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


if __name__ == '__main__':
    imgname = 'ww_sample.jpg'
    portion = os.path.splitext(imgname)
    img = cv2.imread(imgname)
    for theta in range(-180,180,30):
        for fov in range(60,120,20):
            for phi in range(-20,20,10):
                out_img = crop_panorama_image(img, theta=theta, phi=phi, res_x=512,
                                            res_y=512, fov=fov, debug=False)
                outputname = portion[0] + '_theta@' + str(theta) + '_fov@' + str(fov) + '_phi@' + str(phi) + '.jpg'
                cv2.imwrite(outputname, out_img)

