"""
    File name: egyptian_hieroglyph_extractor.py
    Author: Matthew Carter
    Date created: 17/08/2020
    Date last modified: 23/01/2021
    Python Version: 3.8

    Dedicated to Peanut the mouse, for being an incredible little fighter.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Global variables.
# List to store coordinates of user defined area in image to remove.
selected_area_vertices = []


# Mouse callback function to mark and save the coordinates of where the user has clicked on the image.
def outline_area_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Mark point where user has clicked.
        cv2.circle(drawing_img, (x, y), 4, (0, 0, 0), -1)
        # Save click point x,y coordinates as a tuple into a list.
        selected_area_vertices.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        # Clear coordinates in current selection to allow user to restart selection.
        clear_selected_area_vertices()
        print("Current selection cleared. Restart selecting.")


# Function to ask user a yes/no question.
def request_user_input(question):
    while True:
        response = input(question).lower()
        if response in ["y", "n"]:
            # Valid response provided.
            if response == "n":
                return False
            else:
                return True


# Function to draw and fill an area using the coordinates of vertices chosen by the user through their mouse clicks.
def draw_fill_area(image, vertices_list):
    # If there are three or more vertices in the list, draw and fill area.
    if len(vertices_list) > 2:
        cv2.fillPoly(image, np.array([vertices_list], np.int32), (0, 0, 0))
    else:
        print("Not enough points selected to draw area.")
    # Clear coordinates list of what has been drawn or couldn't be drawn due to insufficient points.
    clear_selected_area_vertices()


# Function to clear selected area vertices list.
def clear_selected_area_vertices():
    selected_area_vertices.clear()


# On change function for trackbar.
def custom_on_change(x):
    pass


# Read in the image.
orig_img = cv2.imread("sample_hieroglyphs.jpg")
# cv2.imshow("Original", orig_img)

# Scale the image to ensure it is 800 pixels in width while maintaining its aspect ratio.
img_height, img_width, img_channels = orig_img.shape
scale = 800 / img_width
width = int(img_width * scale)
height = int(img_height * scale)
scaled_img = cv2.resize(orig_img, (width, height), interpolation=cv2.INTER_AREA)
# cv2.imshow("Scaled", scaled_img)

# Convert image to greyscale.
grey_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Grey", grey_img)

# Create a window and bind the mouse callback function to it.
cv2.namedWindow("Area Selection")
cv2.setMouseCallback("Area Selection", outline_area_callback)
# Create an image on which to mark out areas.
drawing_img = grey_img.copy()
# Use the mouse click callback function to mark out areas that are not of interest in the image.
mark_out = True
image_modified = False
question_mark_out = "Mark out a region to remove from the image? (y/n): "
question_save_image = "Save the modified image? (y/n): "
while mark_out:
    # Check whether the user wishes to mark out an area.
    if request_user_input(question_mark_out) is False:
        # User chose not to mark out area.
        mark_out = False
        if image_modified is True:
            # Check whether the user wishes to save the image with marked out areas.
            if request_user_input(question_save_image) is True:
                # Save image.
                cv2.imwrite("area_of_interest.jpg", grey_img)
    else:
        # User chose to mark out area.
        while True:
            # Wait 10ms for the spacebar key (ASCII code 32) to be pressed. If pressed break out of loop.
            key_pressed = cv2.waitKey(10) & 0xFF
            if key_pressed == 32:
                break
            cv2.imshow("Area Selection", drawing_img)
        # Draw marked out area and mark the image as being modified.
        draw_fill_area(grey_img, selected_area_vertices)
        image_modified = True
cv2.imshow("Area Of Interest", grey_img)

# Apply Gaussian blur to reduce noise in the image.
blurred_img = cv2.GaussianBlur(grey_img, (5, 5), 0)
# cv2.imshow("Blurred", blurred_img)

# Apply adaptive thresholding. Use inv thresholding function to make hieroglyphs in foreground white which is desired by
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
# lower_threshold = ret3 * 0.5  # Use for Canny edge if histogram method not used.
# upper_threshold = ret3  # Use for Canny edge if histogram method not used.

# Apply morphological transformation to the binary image created by thresholding.
kernel = np.ones((3, 3), np.uint8)
# erosion_img = cv2.erode(thresh1_img, kernel, iterations=1)
# dilation_img = cv2.dilate(thresh1_img, kernel, iterations=1)
# opening_img = cv2.morphologyEx(thresh1_img, cv2.MORPH_OPEN, kernel)
closing_img = cv2.morphologyEx(thresh2_img, cv2.MORPH_CLOSE, kernel)
# gradient_img = cv2.morphologyEx(thresh1_img, cv2.MORPH_GRADIENT, kernel)
# cv2.imshow("erosion", erosion_img)
# cv2.imshow("dilation", dilation_img)
# cv2.imshow("opening", opening_img)
cv2.imshow("closing", closing_img)
# cv2.imshow("gradient", gradient_img)

# TODO: Canny is not needed for finding the horizontal/vertical lines but will be used to find individual hieroglyphs in
#  the regions of interest that are found using Hough.
# # Link suggests using median of image histogram to provide threshold values for Canny edge detection:
# # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
# # http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
# # Create histogram for the image.
# histogram = cv2.calcHist([blurred_img], [0], None, [256], [0, 256])
# plt.plot(histogram)
# plt.show()
# # Find the pixel value (x-axis of histogram) associated with the median count value. Convert histogram ndarray to a
# # list and create a list for the histogram bins which represent pixel values 0-255.
# counts = [count for [count] in histogram]
# pixel_values = list(range(0, 256))
# # Combine lists so count values are stored with their associated pixel values and sort it by counts in ascending order.
# counts_values_combined = sorted(zip(counts, pixel_values))
# median_value_location = len(counts_values_combined) // 2
# # Tuples in counts_values_combined list are structured (count, pixel value).
# median_pixel_value = counts_values_combined[median_value_location][1]
# # Calculate lower and upper threshold for Canny edge detection based on z-scores (0.66 and 1.33) which are the number
# # of standard deviations from the mean (or in this case applied to the median as it is not as affected by extremes).
# lower_threshold = 0.66 * median_pixel_value
# upper_threshold = 1.33 * median_pixel_value

# # Apply Canny edge detection.
# edges = cv2.Canny(INSERT_img, lower_threshold, upper_threshold, apertureSize=3)
# cv2.imshow("Canny", edges)

# Create a window to hold the trackbars and image.
cv2.namedWindow("Hough")

# Create trackbars that can be used to adjust Hough transform parameters.
cv2.createTrackbar("min_line_length", "Hough", 150, 300, custom_on_change)
cv2.createTrackbar("max_line_gap", "Hough", 150, 300, custom_on_change)
cv2.createTrackbar("threshold", "Hough", 150, 300, custom_on_change)
# # Create trackbar providing a tolerance value for use in ensuring only horizontal/vertical Hough lines are plotted.
# cv2.createTrackbar("tolerance", "Hough", 10, 20, custom_on_change)

# Initiate the Hough image.
hough_lines_img = scaled_img.copy()

while True:
    # Wait 10ms for the ESC key (ASCII code 27) to be pressed. If pressed break out of loop.
    key_pressed = cv2.waitKey(10) & 0xFF
    if key_pressed == 27:
        break

    # Return position of each trackbar.
    min_line_length = cv2.getTrackbarPos("min_line_length", "Hough")
    max_line_gap = cv2.getTrackbarPos("max_line_gap", "Hough")
    threshold = cv2.getTrackbarPos("threshold", "Hough")
    # tolerance = cv2.getTrackbarPos("tolerance", "Hough")

    # Find/highlight the long horizontal and vertical lines that bound the hieroglyphs in the image by applying the
    # probabilistic Hough Transform (unlike standard Hough it uses only a random subset of the points so is less
    # computationally intensive). May then be possible to isolate these regions of interest.
    lines = cv2.HoughLinesP(closing_img, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=min_line_length,
                            maxLineGap=max_line_gap)

    # Plot only the horizontal and vertical Hough lines (if there are any) on a copy of the scaled colour image. With
    # each loop, the Hough lines image is reset to a clean scaled image with no lines on it before plotting again.
    # Lines are unlikely to be exactly horizontal/vertical (i.e. x1 != x2 and y1 != y2) but are assumed to be if within
    # a tolerance value (in pixels). If x1 and x2 are within tolerance the line is considered vertical. If y1 and y2 are
    # within tolerance the line is considered horizontal.
    hough_lines_img = scaled_img.copy()
    if lines is not None:
        for line in lines:
            tolerance = 10
            x1, y1, x2, y2 = line[0]
            if x1 - tolerance <= x2 <= x1 + tolerance or y1 - tolerance <= y2 <= y1 + tolerance:
                cv2.line(hough_lines_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show Hough lines.
    cv2.imshow("Hough", hough_lines_img)

# Show final Hough lines image.
cv2.imshow("Final Hough", hough_lines_img)
cv2.waitKey(0)

# Destroy all open windows.
cv2.destroyAllWindows()
