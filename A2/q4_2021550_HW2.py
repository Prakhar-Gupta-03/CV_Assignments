# %% [markdown]
# #### Importing Libraries

# %%
import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

# %% [markdown]
# #### Setup

# %%
n_rows = 6
n_cols = 9

# %%
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# %%
# Vectors for storing 3D points and corresponding 2D points for each checkerboard image
objectPoints = []
imagePoints = []

# %%
# Defining the world coordinates for the 3D points
objectCoordinates = np.zeros((1, n_rows * n_cols, 3), np.float32)
objectCoordinates[0, :, :2] = np.mgrid[0:n_rows, 0:n_cols].T.reshape(-1, 2)

# %%
# Extracting path of individual image stored in a given directory
images = glob.glob('./CV_images/*.jpg')

# %% [markdown]
# #### Finding checkerboard corners

# %%
for img in images:
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    returnValue, corners = cv2.findChessboardCorners(gray, (n_rows, n_cols), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if returnValue == True:
        objectPoints.append(objectCoordinates)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imagePoints.append(corners2)
        image = cv2.drawChessboardCorners(image, (n_rows, n_cols), corners2, returnValue)
        plt.imshow(image)
        plt.show()

# %% [markdown]
# #### Camera Calibration

# %%
# Calibrating the camera
reprojectionError, cameraMatrix, distorsionCoefficients, rotationVectors, translationVectors = cv2.calibrateCamera(objectPoints, imagePoints, gray.shape[::-1], None, None)

# %% [markdown]
# #### Part 1: Reporting estimated intrinsic camera parameters

# %%
# Intrinsic camera parameters
print("Camera Matrix: \n", cameraMatrix)
print("Focal Length (fx, fy): ", (cameraMatrix[0, 0], cameraMatrix[1, 1]))
print("Principal Point (cx, cy): ", (cameraMatrix[0, 2], cameraMatrix[1, 2]))
print("Skew Coefficient: ", cameraMatrix[0, 1])
print("Distorsion Coefficients: ", distorsionCoefficients.ravel())
print("Mean Reprojection Error: ", reprojectionError)

# %% [markdown]
# #### Part 2: Reporting estimated extrinsic camera parameters for each image

# %%
# Extrinsic camera parameters
for i in range(len(images)):
    print("Image ", i)
    print("Rotation Vector: ", rotationVectors[i].flatten())
    print("Translation Vector: ", translationVectors[i].flatten())

# %% [markdown]
# #### Part 3: Reporting estimated distortion parameters and undistorting images

# %%
print("Radial Distortion Coefficients: ", distorsionCoefficients[0:2])

# %%
# Distorting images using radial distortion coefficients
random_images = np.random.choice(images, 5)
for i in range(5):
    image = cv2.imread(random_images[i])
    height, width = image.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distorsionCoefficients, (width, height), 0, (width, height))
    undistortedImage = cv2.undistort(image, cameraMatrix, distorsionCoefficients, None, newCameraMatrix)
    # Subplot for original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    # Subplot for undistorted image
    plt.subplot(1, 2, 2)
    plt.imshow(undistortedImage)
    plt.title("Undistorted Image")
    plt.show()

# %% [markdown]
# #### Part 4: Reporting re-projection error for each image and plotting results

# %%
# Computing the reprojection error for each image using the estimated intrinsic and extrinsic camera parameters
overallError = 0
errors = []
for i in range(len(objectPoints)):
    imagePoints2, _ = cv2.projectPoints(objectPoints[i], rotationVectors[i], translationVectors[i], cameraMatrix, distorsionCoefficients)
    error = cv2.norm(imagePoints[i], imagePoints2, cv2.NORM_L2) / len(imagePoints2)
    overallError += error
    errors.append(error)
    print("Reprojection Error for Image", i, ": ", error)

# %%
meanError = overallError / len(objectPoints)
standardDeviation = np.std(errors)
print("Mean Reprojection Error: ", meanError)
print("Standard Deviation of Reprojection Error: ", standardDeviation)

# %%
# plot the error using a bar chart
plt.bar(range(len(errors)), errors)
plt.xlabel("Image Number")
plt.ylabel("Reprojection Error")
plt.title("Reprojection Error for Each Image")
plt.grid()
plt.show()

# %% [markdown]
# #### Part 5: Plotting of corners detection on images before and after re-projection

# %%
# Iterate through each image
for i in range(len(images)):
    image = cv2.imread(images[i])
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect chessboard corners
    returnValue, corners = cv2.findChessboardCorners(gray, (n_rows, n_cols), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if returnValue == True:
        # Refining corner locations
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Project corners onto image plane using camera parameters
        projected_points, _ = cv2.projectPoints(objectCoordinates, rotationVectors[i], translationVectors[i], cameraMatrix, distorsionCoefficients)
        # Subplot for detected corners
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Detected Corners')
        plt.scatter(corners_refined[:, 0, 0], corners_refined[:, 0, 1], c='red')
        # Subplot for reprojection of corners
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Reprojected Corners')
        plt.scatter(projected_points[:, 0, 0], projected_points[:, 0, 1], c='blue')
        plt.show()


# %% [markdown]
# #### Part 6: Reporting checkerboard plane normals for each image

# %%
# Assuming the checkerboard lies in the XY plane of the world coordinate frame
normal_world = np.array([[0], [0], [1]])
plane_normals = []
for i in range(len(rotationVectors)):
    # Using Rodrigues' formula to find rotation matrix
    R, _ = cv2.Rodrigues(rotationVectors[i])
    # Transforming the normal vector to the camera coordinate frame using the rotation matrix
    normal_camera = np.dot(R, normal_world)
    plane_normals.append(normal_camera)
plane_normals = np.array(plane_normals)
for i in range(len(plane_normals)):
    print(f"Image {i}: Checkerboard Plane Normal (in Camera Coordinate Frame):\n{plane_normals[i].flatten()}")


