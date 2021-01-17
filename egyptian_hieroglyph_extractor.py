"""
    File name: egyptian_hieroglyph_extractor.py
    Author: Matthew Carter
    Date created: 17/08/2020
    Date last modified: 14/01/2021
    Python Version: 3.8
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read in image and convert it into greyscale.
orig_img = cv2.imread("sample_hieroglyphs.jpg")
grey_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Original", orig_img)
# cv2.imshow("Grey", grey_img)

# Apply Gaussian blur to try reducing noise.
blurred_img = cv2.GaussianBlur(grey_img, (5, 5), 0)
# cv2.imshow("Blurred", blurred_img)

# Link suggests using median of image histogram to provide threshold values for Canny edge detection:
# https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
# http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
# Create histogram for the image.
histogram = cv2.calcHist([blurred_img], [0], None, [256], [0, 256])
plt.plot(histogram)
plt.show()
# Find the pixel value (x-axis of histogram) associated with the median count value. Convert histogram ndarray to a
# list and create a list for the histogram bins which represent pixel values 0-255.
counts = [count for [count] in histogram]
pixel_values = list(range(0, 256))
# Combine lists so count values are stored with their associated pixel values and sort it by counts in ascending order.
counts_values_combined = sorted(zip(counts, pixel_values))
median_value_location = len(counts_values_combined) // 2
# Tuples in counts_values_combined list are structured (count, pixel value).
median_pixel_value = counts_values_combined[median_value_location][1]
# Calculate lower and upper threshold for Canny edge detection based on z-scores (0.66 and 1.33) which are the number
# of standard deviations from the mean (or in this case applied to the median as it is not as affected by extremes).
lower_threshold = 0.66 * median_pixel_value
upper_threshold = 1.33 * median_pixel_value

# Adaptive thresholding. Use inv thresholding function to make hieroglyphs in foreground white which is desired by
# morphological transformations.
# thresh1_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
thresh2_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
# cv2.imshow("adaptive_mean", thresh1_img)
cv2.imshow("adaptive_gauss", thresh2_img)

# # Use Otsu's thresholding to establish an upper and lower threshold value for Canny edge detection. It works best on
# # bimodal images where the foreground is distinct from the background.
# ret3, thresh3_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# ret4, thresh4_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# ret5, thresh5_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_TRUNC+cv2.THRESH_OTSU)
# ret6, thresh6_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
# ret7, thresh7_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
# cv2.imshow("thresh_binary_otsu", thresh3_img)
# cv2.imshow("thresh_binary_inv_otsu", thresh4_img)
# cv2.imshow("thresh_trunc", thresh5_img)
# cv2.imshow("thresh_tozero_otsu", thresh6_img)
# cv2.imshow("thresh_tozero_inv_otsu", thresh7_img)

# Morphological transformation. It works best on binary images.
kernel = np.ones((3, 3), np.uint8)
# erosion_img = cv2.erode(thresh1_img, kernel, iterations=1)
# dilation_img = cv2.dilate(thresh1_img, kernel, iterations=1)
opening_img = cv2.morphologyEx(thresh2_img, cv2.MORPH_OPEN, kernel)
# closing_img = cv2.morphologyEx(thresh1_img, cv2.MORPH_CLOSE, kernel)
# gradient_img = cv2.morphologyEx(thresh1_img, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow("erosion", erosion_img)
# cv2.imshow("dilation", dilation_img)
cv2.imshow("opening", opening_img)
# cv2.imshow("closing", closing_img)
# cv2.imshow("gradient", gradient_img)

# Apply Canny edge detection.
# TODO: Canny threshold values need dynamically adjusting depending on image passed in.
# lower_threshold = ret3 * 0.5
# upper_threshold = ret3
edges = cv2.Canny(opening_img, lower_threshold, upper_threshold, apertureSize=3)
cv2.imshow("Canny", edges)

# Create a window to hold the trackbars and image.
cv2.namedWindow("Hough")


def custom_on_change(x):
    pass


# Create trackbars that can be used to adjust Hough transform parameters.
cv2.createTrackbar("min_line_length", "Hough", 50, 100, custom_on_change)
cv2.createTrackbar("max_line_gap", "Hough", 250, 500, custom_on_change)
cv2.createTrackbar("threshold", "Hough", 200, 400, custom_on_change)

while True:
    # Wait 1ms for a ESC key (ASCII code 27) to be pressed (0 param would leave it waiting for infinity). If pressed
    # break out of loop.
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == 27:
        break

    # Return position of each trackbar.
    min_line_length = cv2.getTrackbarPos("min_line_length", "Hough")
    max_line_gap = cv2.getTrackbarPos("max_line_gap", "Hough")
    threshold = cv2.getTrackbarPos("threshold", "Hough")

    # ----- START PROBABILISTIC HOUGH ----- #

    # Find/highlight the long horizontal and vertical lines that bound the hieroglyphs in the image by applying the
    # probabilistic Hough Transform (unlike standard Hough it uses only a random subset of the points so is less
    # computationally intensive). May then be possible to isolate these regions of interest.
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    # Plot the Hough lines (if there are any) on a copy of the original colour image.
    hough_lines_img = orig_img.copy()
    if lines is not None:
        for line in lines:
            # Only plot horizontal and vertical lines. Lines won't be exactly horizontal/vertical (i.e. x1 != x2 and
            # y1 != y2) but within tolerance. If x1 and x2 are within tolerance, the line is considered vertical. If
            # y1 and y2 are within tolerance, the line is considered horizontal.
            tolerance = 20
            x1, y1, x2, y2 = line[0]
            if x1 - tolerance <= x2 <= x1 + tolerance or y1 - tolerance <= y2 <= y1 + tolerance:
                cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # TODO: If needed for future param alterations filter out all lines that aren't in region of 0 rads
            #  (horizontal) and 1.57 rads (vertical)
            # TODO: If needed check distances in x and y direction between lines, so only those bordering hieroglyphs
            #  are drawn.

    # ----- END PROBABILISTIC HOUGH ----- #

    # ----- START STANDARD HOUGH ----- #

    # # Apply standard Hough transform.
    # lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold)
    # if lines is not None:
    #     for line in lines:
    #         # TODO: If needed for future param alterations filter out all lines that aren't in region of 0 rads
    #         #  (horizontal) and 1.57 rads (vertical)
    #         # First element of "line" gives you rho and theta. Rho is distance of line perpendicular to line from
    #         # origin (0, 0), which is the top left corner of the image. Theta is the line rotation angle in radians.
    #         rho, theta = line[0]
    #         # A line in polar coordinate system is: xcos(theta) + ysin(theta) = rho. Therefore get cos(theta) and
    #         # sin(theta) values to convert Polar coordinates into Cartesian coordinates to plot the lines.
    #         a = np.cos(theta)
    #         b = np.sin(theta)
    #         # Get the origin (0, 0).
    #         x0 = a * rho
    #         y0 = b * rho
    #         # Convert back to get x1, y1 and x2, y2 coordinates of the line.
    #         # x1 stores the rounded off value of (rho * cos(theta) - 1000 * sin(theta))
    #         x1 = int(x0 + 1000 * (-b))
    #         # y1 stores the rounded off value of (rho * sin(theta) + 1000 * cos(theta))
    #         y1 = int(y0 + 1000 * (a))
    #         # x2 stores the rounded off value of (rho * cos(theta) + 1000 * sin(theta))
    #         x2 = int(x0 - 1000 * (-b))
    #         # y2 stores the rounded off value of (rho * sin(theta) - 1000 * cos(theta))
    #         y2 = int(y0 - 1000 * (a))
    #         # Plot the Hough lines (if there are any) on a copy of the original colour image.
    #         hough_lines_img = orig_img.copy()
    #         cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ----- END STANDARD HOUGH ----- #

    # Show Hough lines.
    cv2.imshow("Hough", hough_lines_img)

# Destroy all open windows.
cv2.destroyAllWindows()


# TODO: Contour features to bound each hieroglyph inside a rectangle for extraction?
#  https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html
