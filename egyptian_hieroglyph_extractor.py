"""
    File name: egyptian_hieroglyph_extractor.py
    Author: Matthew Carter
    Date created: 17/08/2020
    Date last modified: 10/01/2021
    Python Version: 3.8
"""

import cv2
import numpy as np

# Read in image and convert it into greyscale.
orig_img = cv2.imread("sample_hieroglyphs.jpg")
grey_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

# Apply blurring to try and reduce noise.
grey_img = cv2.GaussianBlur(grey_img, (5, 5), 0)

# Apply Canny edge detection.
lower_threshold = 50
upper_threshold = 150
edges = cv2.Canny(grey_img, lower_threshold, upper_threshold, apertureSize=3)

# Try to isolate the long horizontal and vertical lines in the image (may then be possible to isolate the regions
# and hieroglyphs bounded within those regions).

# # Apply standard Hough lines transform.
# lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
# for line in lines:
#     # First element of line gives you rho and theta. rho is distance from 0,0 (top left corner of image). theta is
#     # the line rotation angle in radians.
#     rho, theta = line[0]
#     # Get cos(theta) and sin(theta) values because we want to convert Polar coordinates into Cartesian coordinates
#     # for use with the line method.
#     a = np.cos(theta)
#     b = np.sin(theta)
#     # Get the origin, (0, 0), i.e. top left corner of image.
#     x0 = a * rho
#     y0 = b * rho
#     # Get x1, y1 and x2, y2 coordinates of the line.
#     # x1 stores the rounded off value of (rho * cos(theta) - 1000 * sin(theta))
#     x1 = int(x0 + 1000 * (-b))
#     # y1 stores the rounded off value of (rho * sin(theta) + 1000 * cos(theta))
#     y1 = int(y0 + 1000 * (a))
#     # x2 stores the rounded off value of (rho * cos(theta) + 1000 * sin(theta))
#     x2 = int(x0 - 1000 * (-b))
#     # y2 stores the rounded off value of (rho * sin(theta) - 1000 * cos(theta))
#     y2 = int(y0 - 1000 * (a))
#     # Plot the line on the image.
#     cv2.line(grey_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Apply the probabilistic Hough transform. Uses only a random subset of the points (unlike standard Hough which uses
# all).
min_line_length = 100
max_line_gap = 20
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=min_line_length, maxLineGap=max_line_gap)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(grey_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Show hough lines.
cv2.imshow("Hough Line", grey_img)

cv2.waitKey(0)

# Destroy all open windows.
cv2.destroyAllWindows()







# Contour features to bound each hieroglyph inside a rectangle for extraction?
# https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html







###### HSV APPROACH ######
# Investigating if using HSV colour space will provide anything useful when highlighting hieroglyphs.

# orig_img = cv2.imread("sample_hieroglyphs.jpg", cv2.IMREAD_COLOR)
# img = orig_img.copy()
#
# # # Resize the image if necessary.
# # scale = 1
# # img_height, img_width, img_channels = img.shape
# # width = int(img_width * scale)
# # height = int(img_height * scale)
# # img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
#
# # Apply blurring to try and reduce noise.
# img = cv2.GaussianBlur(img, (5, 5), 0)
# # Convert to the HSV colour space.
# img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#
# # Function to be used as the onChange method in createTrackbar. x is the value of the trackbar.
# def custom_on_change(x):
#     print(x)
#
#
# # Create a window to hold the trackbars and image.
# cv2.namedWindow("Image")
#
# # Want to change the BGR values of the image using trackbars.
# cv2.createTrackbar("H_channel_low", "Image", 0, 180, custom_on_change)
# cv2.createTrackbar("H_channel_high", "Image", 180, 180, custom_on_change)
# cv2.createTrackbar("S_channel_low", "Image", 0, 180, custom_on_change)
# cv2.createTrackbar("S_channel_high", "Image", 180, 180, custom_on_change)
# cv2.createTrackbar("V_channel_low", "Image", 0, 180, custom_on_change)
# cv2.createTrackbar("V_channel_high", "Image", 180, 180, custom_on_change)
# cv2.createTrackbar("0:OFF  1:ON", "Image", 0, 1, custom_on_change)
#
# while True:
#     # Wait for key, and if it is the ESC key then break out of loop.
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break
#
#     # Return each trackbar's position.
#     h_channel_low = cv2.getTrackbarPos("H_channel_low", "Image")
#     h_channel_high = cv2.getTrackbarPos("H_channel_high", "Image")
#     s_channel_low = cv2.getTrackbarPos("S_channel_low", "Image")
#     s_channel_high = cv2.getTrackbarPos("S_channel_high", "Image")
#     v_channel_low = cv2.getTrackbarPos("V_channel_low", "Image")
#     v_channel_high = cv2.getTrackbarPos("V_channel_high", "Image")
#     switch = cv2.getTrackbarPos("0:OFF  1:ON", "Image")
#
#     # Show HSV colour image using threshold values if switch is set to 1.
#     if switch == 0:
#         # Show unmasked HSV image.
#         cv2.imshow("Image", img)
#     else:
#         # Show HSV image using threshold mask.
#         mask = cv2.inRange(img,
#                            (h_channel_low, s_channel_low, v_channel_low),
#                            (h_channel_high, s_channel_high, v_channel_high))
#         final_img = cv2.bitwise_and(img, img, mask=mask)
#         cv2.imshow("Image", final_img)
#
# # Destroy all open windows.
# cv2.destroyAllWindows()
