import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt


def main():
    # Load calibration images
    images = glob.glob('camera_cal/*.jpg')
    mtx, dist = calib_cam(images, num_checker_pts=(9, 6))

    # Save the camera calibration result for later use
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("calib_cam_dist_pickle.p", "wb" ) )

    # Show an example of undistortion
    show_test_undist(mtx, dist,
                     src_path='camera_cal/calibration3.jpg', dst_path='output_images/test_undist.jpg')
    return


def calib_cam(image_paths, num_checker_pts):
    """ Do camera calibration given object points and image points
    https://github.com/udacity/CarND-Camera-Calibration
    """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    n_px = num_checker_pts[0]
    n_py = num_checker_pts[1]
    objp = np.zeros((n_px * n_py, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_px, 0:n_py].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(image_paths):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (n_px, n_py), None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1][1:], None, None)
    return mtx, dist


def show_test_undist(mtx, dist, src_path, dst_path):
    img = cv2.imread(src_path)
    img_size = (img.shape[1], img.shape[0])
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(dst_path, dst)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()
    return


if __name__ == '__main__':
    main()
