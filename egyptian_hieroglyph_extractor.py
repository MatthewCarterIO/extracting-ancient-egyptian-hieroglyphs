import cv2

orig_img = cv2.imread("sample_hieroglyphs.jpg", 1)
img = orig_img.copy()
img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("image", img)


def custom_on_change():
    pass


# Want to change the threshold values of the image using trackbars.
cv2.createTrackbar("low_threshold", "image", 0, 255, custom_on_change)
cv2.createTrackbar("up_threshold", "image", 255, 255, custom_on_change)

while True:
    # Wait 1ms for a ESC key (ASCII code 27) to be pressed (0 param would leave it waiting for infinity). If pressed
    # break out of loop.
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == 27:
        break

    # Return position of each trackbar.
    lower_threshold = cv2.getTrackbarPos("low_threshold", "image")
    upper_threshold = cv2.getTrackbarPos("up_threshold", "image")
    # Apply Canny edge detector.
    edges = cv2.Canny(img, lower_threshold, upper_threshold)
    # Show edges.
    cv2.imshow("Edges", edges)

# Destroy all open windows.
cv2.destroyAllWindows()
