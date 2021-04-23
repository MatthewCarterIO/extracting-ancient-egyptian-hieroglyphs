"""
    File name: egyptian_hieroglyph_extractor.py
    Author: Matthew Carter
    Date created: 17/08/2020
    Date last modified: 23/04/2021
    Python Version: 3.8

    Dedicated to Peanut the mouse for being an incredible little fighter. Always.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Function to scale the image while maintaining its aspect ratio.
def scale_image(image_to_scale, desired_width_pixels):
    img_height, img_width, img_channels = image_to_scale.shape
    scale = desired_width_pixels / img_width
    width = int(img_width * scale)
    height = int(img_height * scale)
    # Return the scaled image.
    return cv2.resize(image_to_scale, (width, height), interpolation=cv2.INTER_AREA)


# Function to exclude areas from the image if desired.
def select_excluded_areas(grey_image, scaled_image):
    # Create a window in which the user can select areas of the image to exclude from analysis if desired and set the
    # mouse click callback function to be used within it.
    cv2.namedWindow("Exclude Area")
    cv2.setMouseCallback("Exclude Area", select_point_callback)
    # Create an image on which the user will select areas to exclude.
    exclude_area_img = grey_image.copy()
    # Ask user if they wish to select any areas and proceed accordingly.
    select_area = True
    image_modified = False
    question_select_area = "\nSelect an area to remove " \
                           "\nInfo:" \
                           "\nLeft mouse button - Select vertices" \
                           "\nRight mouse button - Clear current vertices group selection" \
                           "\nSpacebar - Finish current vertices group selection / exit" \
                           "\n(y/n): "
    question_proceed = "\nProceed using excluded areas? (y/n): "
    while select_area:
        # Check whether the user wishes to select an area.
        if ask_user(question_select_area) is False:
            # User chose not to select an area.
            select_area = False
            if image_modified is True:
                # Show user the modified image with all excluded areas included.
                print("\nReview excluded areas on image and press any key when done to continue.")
                cv2.imshow("Area Of Interest", grey_image)
                cv2.waitKey(0)
                # Check whether the user wishes to proceed with the selected excluded areas.
                if ask_user(question_proceed) is True:
                    continue
                else:
                    # User chose not to use the excluded areas that were selected, so reset grey image to its state
                    # before any selection was done.
                    grey_image = cv2.cvtColor(scaled_image, cv2.COLOR_BGR2GRAY)
        else:
            # User chose to select an area.
            while True:
                # Wait 10ms for the spacebar key (ASCII code 32) to be pressed. If pressed break out of loop.
                key_pressed = cv2.waitKey(10) & 0xFF
                if key_pressed == 32:
                    break
                # Show image on which user will select vertices of area.
                cv2.imshow("Exclude Area", exclude_area_img)
                # Mark the points (area vertices) where user has clicked on image (purely for user feedback).
                if len(selected_area_vertices) > 0:
                    # Add the latest selected vertex to the image.
                    cv2.circle(exclude_area_img, (selected_area_vertices[-1][0], selected_area_vertices[-1][1]),
                               4, (0, 0, 0), -1)
            # Using current selection of vertices, draw excluded area onto grey image.
            draw_fill_area(grey_image, selected_area_vertices)
            # Update excluded areas image to match the current grey image (which shows any previously selected areas).
            exclude_area_img = grey_image.copy()
            # Image has been modified.
            image_modified = True
    # Return grey image with or without excluded areas.
    return grey_image


# Mouse callback function to save the coordinates of where the user has clicked on the image.
def select_point_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # User presses left mouse button. Save click point x,y coordinates as a tuple into a list.
        selected_area_vertices.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        # User presses right mouse button. Clear coordinates in current selection to allow user to restart selection.
        clear_selected_area_vertices()
        print("Current selection cleared. Start again.")


# Function to ask user a yes/no question.
def ask_user(question):
    while True:
        response = input(question).lower()
        if response in ["y", "n"]:
            # Valid response provided.
            if response == "y":
                return True
            else:
                return False


# Function to draw and fill an area using the coordinates of vertices chosen by the user through their mouse clicks.
def draw_fill_area(image, vertices_list):
    # If there are three or more vertices in the list, draw and fill area.
    if len(vertices_list) > 2:
        cv2.fillPoly(image, np.array([vertices_list], np.int32), (0, 0, 0))
    else:
        print("Cannot draw area. Minimum of three points required.")
    # Clear coordinates list of what has been drawn or couldn't be drawn due to insufficient points.
    clear_selected_area_vertices()


# Function to clear selected area vertices list.
def clear_selected_area_vertices():
    selected_area_vertices.clear()
    # TODO: When user clears exits selection, points on exclude_areas_img are currently not removed (since change made).


# On change function for trackbar.
def custom_on_change(x):
    pass


# Main function.
def main():
    # Read in the image.
    orig_img = cv2.imread("sample_hieroglyphs.jpg")
    # cv2.imshow("Original", orig_img)

    # Scale the image to 1000 pixels in width while maintaining its aspect ratio.
    scaled_img = scale_image(orig_img, 1000)
    # cv2.imshow("Scaled", scaled_img)

    # Convert image to greyscale.
    grey_img = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Grey", grey_img)

    # Select any areas to remove from the image before analysis.
    grey_img = select_excluded_areas(grey_img, scaled_img)
    # Show area of interest (whether or not areas for exclusion were selected).
    cv2.imshow("Area Of Interest", grey_img)

    # Due to the nature of how light interacts with carvings and how the shadows fall, the edges of hieroglyphs in
    # images can be both light (whiter colour) and dark (greys/blacks). To obtain useful contours or edges the majority
    # of a hieroglyph edge must be uniform.
    # Create a mask from the grey image that allows all colours through except those near black (0,0,0) and those near
    # white (255, 255, 255). The areas with colours that are not let through (i.e. the edges of the hieroglyphs) are
    # set to black in the mask.
    mask = cv2.inRange(grey_img, 100, 210)
    # Invert the mask, setting the areas of the mask showing edges from the image to white.
    mask_inv = cv2.bitwise_not(mask)
    # Overlay the mask on the scaled colour image to leave just the edges.
    # mask_applied_img = cv2.bitwise_and(scaled_img, scaled_img, mask=mask_inv)
    cv2.imshow("Mask", mask)
    cv2.imshow("Inverted Mask", mask_inv)
    # cv2.imshow("Mask Applied", mask_applied_img)

    # # Apply Gaussian blur to reduce noise in the image.
    # blurred_img = cv2.GaussianBlur(mask_inv, (5, 5), 0)
    # cv2.imshow("Blurred", blurred_img)

    # Apply adaptive thresholding. Use inv thresholding function to make hieroglyphs in foreground white which is
    # desired by morphological transformations.
    # thresh1_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # thresh2_img = cv2.adaptiveThreshold(blurred_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5)
    # cv2.imshow("adaptive_mean", thresh1_img)
    # cv2.imshow("adaptive_gauss", thresh2_img)

    # Use Otsu's thresholding to establish an upper and lower threshold value for Canny edge detection. It works best on
    # bimodal images where the foreground is distinct from the background.
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
    # kernel = np.ones((3, 3), np.uint8)
    # erosion_img = cv2.erode(thresh1_img, kernel, iterations=1)
    # dilation_img = cv2.dilate(thresh1_img, kernel, iterations=1)
    # opening_img = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
    # closing_img = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
    # gradient_img = cv2.morphologyEx(thresh1_img, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow("erosion", erosion_img)
    # cv2.imshow("dilation", dilation_img)
    # cv2.imshow("opening", opening_img)
    # cv2.imshow("closing", closing_img)
    # cv2.imshow("gradient", gradient_img)

    # # Find contours on the threshold image and draw them onto a copy of the scaled image.
    # contours_thresh, hierarchy_thresh = cv2.findContours(blurred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # threshold_contours_img = scaled_img.copy()
    # cv2.drawContours(threshold_contours_img, contours_thresh, -1, (255, 0, 0), 1)
    # cv2.imshow("Threshold Contours", threshold_contours_img)

    # # Link suggests using median of image histogram to provide threshold values for Canny edge detection:
    # # https://stackoverflow.com/questions/4292249/automatic-calculation-of-low-and-high-thresholds-for-the-canny-operation-in-open
    # # http://www.kerrywong.com/2009/05/07/canny-edge-detection-auto-thresholding/
    # # Create image histogram.
    # histogram = cv2.calcHist([IMAGENAME], [0], None, [256], [0, 256])
    # plt.plot(histogram)
    # plt.show()
    # # Find the pixel value (x-axis of histogram) associated with the median count value. Convert histogram ndarray to a
    # # list and create a list for the histogram bins which represent pixel values 0-255.
    # counts = [count for [count] in histogram]
    # pixel_values = list(range(0, 256))
    # # Combine lists so count values are stored with their associated pixel values. Sort it by counts in ascending
    # # order.
    # counts_values_combined = sorted(zip(counts, pixel_values))
    # median_value_location = len(counts_values_combined) // 2
    # # Tuples in counts_values_combined list are structured (count, pixel value).
    # median_pixel_value = counts_values_combined[median_value_location][1]
    # # Calculate lower and upper threshold for Canny edge detection based on z-scores (0.66 and 1.33) which are the
    # # number standard deviations from the mean (or in this case applied to the median as it is not as affected by
    # # extremes).
    # lower_threshold = 0.66 * median_pixel_value
    # upper_threshold = 1.33 * median_pixel_value
    #
    # # Apply Canny edge detection.
    # edges_img = cv2.Canny(IMAGENAME, lower_threshold, upper_threshold, apertureSize=3)
    # cv2.imshow("Canny", edges_img)
    #
    # # Find contours on the Canny image and draw them onto a copy of the scaled image.
    # contours_canny, hierarchy_canny = cv2.findContours(edges_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # canny_contours_img = scaled_img.copy()
    # cv2.drawContours(canny_contours_img, contours_canny, -1, (0, 0, 255), 1)
    # cv2.imshow("Canny Contours", canny_contours_img)

    # TODO: Hough needs to be next after thresholding to isolate the edges.
    #  1. Isolate the Hough lines along the vertical / horizontal lines/guides surrounding the hieroglyphs.
    #  2. Likely several Hough lines along each individual line/guide. Combine them for each into a single horizontal or
    #  vertical line.
    #  3. Using a distance tolerance between them find pairs so areas between them can be isolated. Start from left for
    #  vertical and top for horizontal, this way no duplicate areas. Extract those sections into their own images.
    #  4. From these images try to extract hieroglyphs using contours or canny. Remove contours below certain lengths
    #  /areas if possible to avoid small imperfectons/holes etc?
    # Find the Hough lines. Create a window to hold the trackbars and image.
    cv2.namedWindow("Hough")
    # Create trackbars that can be used to adjust Hough transform parameters.
    cv2.createTrackbar("min_line_length", "Hough", 150, 300, custom_on_change)
    cv2.createTrackbar("max_line_gap", "Hough", 150, 300, custom_on_change)
    cv2.createTrackbar("threshold", "Hough", 150, 300, custom_on_change)
    # Create a copy of the scaled image onto which the Hough lines will be drawn.
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

        # Find/highlight the long horizontal and vertical lines that bound the hieroglyphs in the image by applying the
        # probabilistic Hough Transform (unlike standard Hough it uses only a random subset of the points so is less
        # computationally intensive). May then be possible to isolate these regions of interest.
        lines = cv2.HoughLinesP(mask_inv, rho=1, theta=np.pi/180, threshold=threshold, minLineLength=min_line_length,
                                maxLineGap=max_line_gap)

        # Plot only the horizontal and vertical Hough lines (if there are any) on a copy of the scaled colour image.
        # With each loop the Hough lines image is reset to a clean scaled image with no lines on it before plotting them
        # again. Lines are unlikely to be exactly horizontal/vertical (i.e. x1 != x2 and y1 != y2) but are assumed to be
        # if within a set tolerance value (in pixels). If x1 and x2 are within tolerance the line is considered
        # vertical. If y1 and y2 are within tolerance the line is considered horizontal.
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

    # Wait for keypress then close all open windows.
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Global variables.
    # List to store coordinates of user defined area in image to remove.
    selected_area_vertices = []

    main()
