"""
    File name: egyptian_hieroglyph_extractor.py
    Author: Matthew Carter
    Date created: 17/08/2020
    Date last modified: 11/01/2021
    Python Version: 3.8
"""

import cv2
import numpy as np

# Read in image and convert it into greyscale.
orig_img = cv2.imread("sample_hieroglyphs.jpg")
grey_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)

# Apply blurring to try and reduce noise.
blurred_img = cv2.GaussianBlur(grey_img, (7, 7), 0)

# Apply Canny edge detection.
lower_threshold = 50
upper_threshold = 150
edges = cv2.Canny(blurred_img, lower_threshold, upper_threshold, apertureSize=3)

# Try to isolate the long horizontal and vertical lines in the image (may then be possible to isolate the regions
# and hieroglyphs bounded within those regions).

# # Apply standard Hough transform.
# hough_threshold = 270
# lines = cv2.HoughLines(edges, 1, np.pi/180, hough_threshold)
# for line in lines:
#     # TODO: If needed for future param alterations filter out all lines that aren't in region of 0 rads (horizontal)
#     #  and 1.57 rads (vertical)
#     # First element of "line" gives you rho and theta. Rho is distance of line perpendicular to line from origin
#     # (0, 0), which is the top left corner of the image. Theta is the line rotation angle in radians.
#     rho, theta = line[0]
#     # A line in polar coordinate system is: xcos(theta) + ysin(theta) = rho. Therefore get cos(theta) and sin(theta)
#     # values to convert Polar coordinates into Cartesian coordinates to plot the lines.
#     a = np.cos(theta)
#     b = np.sin(theta)
#     # Get the origin (0, 0).
#     x0 = a * rho
#     y0 = b * rho
#     # Convert back to get x1, y1 and x2, y2 coordinates of the line.
#     # x1 stores the rounded off value of (rho * cos(theta) - 1000 * sin(theta))
#     x1 = int(x0 + 1000 * (-b))
#     # y1 stores the rounded off value of (rho * sin(theta) + 1000 * cos(theta))
#     y1 = int(y0 + 1000 * (a))
#     # x2 stores the rounded off value of (rho * cos(theta) + 1000 * sin(theta))
#     x2 = int(x0 - 1000 * (-b))
#     # y2 stores the rounded off value of (rho * sin(theta) - 1000 * cos(theta))
#     y2 = int(y0 - 1000 * (a))
#     # Plot the line on the original image.
#     cv2.line(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Apply the probabilistic Hough transform. Uses only a random subset of the points unlike standard Hough so is less
# computationally intensive.
# TODO: Min line length, max line gap and threshold need to be dynamically changed depending on image.
min_line_length = 50
max_line_gap = 400
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 270, minLineLength=min_line_length, maxLineGap=max_line_gap)
# Plot the lines on the original colour image.
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # TODO: If needed check distances in x and y direction between lines, so only those bordering hieroglyphs are drawn.

# Show Hough lines.
cv2.imshow("Hough Lines", orig_img)

cv2.waitKey(0)

# Destroy all open windows.
cv2.destroyAllWindows()

# Contour features to bound each hieroglyph inside a rectangle for extraction?
# https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
