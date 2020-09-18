"""
    File name: egyptian_hieroglyph_extractor.py
    Author: Matthew Carter
    Date created: 17/08/2020
    Date last modified: 18/09/2020
    Python Version: 3.8
"""

import cv2

###### CANNY APPROACH ######

# orig_img = cv2.imread("sample_hieroglyphs.jpg", cv2.IMREAD_GRAYSCALE)
# img = orig_img.copy()
#
# # Resize the image if necessary.
# scale = 0.8
# img_height, img_width = img.shape
# width = int(img_width * scale)
# height = int(img_height * scale)
# img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
#
# # Apply blurring to try and reduce noise.
# img = cv2.GaussianBlur(img, (5, 5), 0)
#
# # Create a window to hold the trackbars and image.
# cv2.namedWindow("image")
#
#
# def custom_on_change(x):
#     print(x)
#
#
# # Create trackbars that can be used to change the threshold values for Canny edge detection.
# cv2.createTrackbar("low_threshold", "image", 0, 255, custom_on_change)
# cv2.createTrackbar("up_threshold", "image", 255, 255, custom_on_change)
#
# while True:
#     # Wait 1ms for a ESC key (ASCII code 27) to be pressed (0 param would leave it waiting for infinity). If pressed
#     # break out of loop.
#     key_pressed = cv2.waitKey(1) & 0xFF
#     if key_pressed == 27:
#         break
#
#     # Return position of each trackbar.
#     lower_threshold = cv2.getTrackbarPos("low_threshold", "image")
#     upper_threshold = cv2.getTrackbarPos("up_threshold", "image")
#     # Apply Canny edge detector.
#     edges = cv2.Canny(img, lower_threshold, upper_threshold)
#     # Show edges.
#     cv2.imshow("image", edges)
#
# # Destroy all open windows.
# cv2.destroyAllWindows()

# Contour features to bound each hieroglyph inside a rectangle for extraction?
# https://docs.opencv.org/master/dd/d49/tutorial_py_contour_features.html




###### HSV APPROACH ######
# Investigating if using HSV colour space will provide anything useful when highlighting hieroglyphs.

orig_img = cv2.imread("sample_hieroglyphs.jpg", cv2.IMREAD_COLOR)
img = orig_img.copy()

# # Resize the image if necessary.
# scale = 1
# img_height, img_width, img_channels = img.shape
# width = int(img_width * scale)
# height = int(img_height * scale)
# img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

# Apply blurring to try and reduce noise.
img = cv2.GaussianBlur(img, (5, 5), 0)
# Convert to the HSV colour space.
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# Function to be used as the onChange method in createTrackbar. x is the value of the trackbar.
def custom_on_change(x):
    print(x)


# Create a window to hold the trackbars and image.
cv2.namedWindow("Image")

# Want to change the BGR values of the image using trackbars.
cv2.createTrackbar("H_channel_low", "Image", 0, 180, custom_on_change)
cv2.createTrackbar("H_channel_high", "Image", 180, 180, custom_on_change)
cv2.createTrackbar("S_channel_low", "Image", 0, 180, custom_on_change)
cv2.createTrackbar("S_channel_high", "Image", 180, 180, custom_on_change)
cv2.createTrackbar("V_channel_low", "Image", 0, 180, custom_on_change)
cv2.createTrackbar("V_channel_high", "Image", 180, 180, custom_on_change)
cv2.createTrackbar("0:OFF  1:ON", "Image", 0, 1, custom_on_change)

while True:
    # Wait for key, and if it is the ESC key then break out of loop.
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # Return each trackbar's position.
    h_channel_low = cv2.getTrackbarPos("H_channel_low", "Image")
    h_channel_high = cv2.getTrackbarPos("H_channel_high", "Image")
    s_channel_low = cv2.getTrackbarPos("S_channel_low", "Image")
    s_channel_high = cv2.getTrackbarPos("S_channel_high", "Image")
    v_channel_low = cv2.getTrackbarPos("V_channel_low", "Image")
    v_channel_high = cv2.getTrackbarPos("V_channel_high", "Image")
    switch = cv2.getTrackbarPos("0:OFF  1:ON", "Image")

    # Show HSV colour image using threshold values if switch is set to 1.
    if switch == 0:
        # Show unmasked HSV image.
        cv2.imshow("Image", img)
    else:
        # Show HSV image using threshold mask.
        mask = cv2.inRange(img,
                           (h_channel_low, s_channel_low, v_channel_low),
                           (h_channel_high, s_channel_high, v_channel_high))
        final_img = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("Image", final_img)

# Destroy all open windows.
cv2.destroyAllWindows()
