# Load the image
image = cv2.imread('img/Morphy33.jpg')

# Perform automatic cropping
cropped_image = auto_crop(image)

# Display the cropped image
crop = cv2.imwrite('imgcrop/Morphycrop33.jpg', cropped_image)

# Load the image
image = cv2.imread('img/Morphy34.jpg')

# Perform automatic cropping
cropped_image = auto_crop(image)

# Display the cropped image
crop = cv2.imwrite('imgcrop/Morphycrop34.jpg', cropped_image)
def label_chessboard(image_path):
    # Load the image
    image = cv2.imread(image_path)
    grid_size = 8
    square_height = image.shape[0] // grid_size
    square_width = image.shape[1] // grid_size
    
    # Annotate each square with shifted labels
    for i in range(grid_size):
        for j in range(grid_size):
            label = f"{chr(65 + j)}{8 - i}"
            # Shift text to the right by adding an offset to the x-coordinate
            cv2.putText(image, label, (j * square_width + 30, (i + 1) * square_height - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Resize image for display if it's too large
    display_width = 800  # Set this to the maximum width that fits your display
    display_height = int(image.shape[0] * (display_width / image.shape[1]))
    resized_image = cv2.resize(image, (display_width, display_height))
    cv2.imwrite('imglabel/Morphylabel33.jpg', resized_image)
    # Display the resized image
    # cv2.imshow('Labeled Chessboard', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Replace 'path_to_chessboard_image.jpeg' with the path to your chessboard image
label_chessboard('imgcrop/Morphycrop33.jpg')

def label_chessboard(image_path):
    # Load the image
    image = cv2.imread(image_path)
    grid_size = 8
    square_height = image.shape[0] // grid_size
    square_width = image.shape[1] // grid_size
    
    # Annotate each square with shifted labels
    for i in range(grid_size):
        for j in range(grid_size):
            label = f"{chr(65 + j)}{8 - i}"
            # Shift text to the right by adding an offset to the x-coordinate
            cv2.putText(image, label, (j * square_width + 30, (i + 1) * square_height - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Resize image for display if it's too large
    display_width = 800  # Set this to the maximum width that fits your display
    display_height = int(image.shape[0] * (display_width / image.shape[1]))
    resized_image = cv2.resize(image, (display_width, display_height))
    cv2.imwrite('imglabel/Morphylabel34.jpg', resized_image)
    # Display the resized image
    # cv2.imshow('Labeled Chessboard', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Replace 'path_to_chessboard_image.jpeg' with the path to your chessboard image
label_chessboard('imgcrop/Morphycrop34.jpg')

img1 = cv2.imread("imglabel/Morphylabel33.jpg")
img2 = cv2.imread("imglabel/Morphylabel34.jpg")
img1 = cv2.resize(img1, (600,360))
img2 = cv2.resize(img2, (600,360))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)
blur2 = cv2.GaussianBlur(gray2, (5, 5), 0)
diff = cv2.absdiff(blur1, blur2)
thresh = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((6,6), np.uint8)
dilate = cv2.dilate(thresh, kernel, iterations=2)
contours = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
for contour in contours:
    if cv2.contourArea(contour) > 500:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img1, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.rectangle(img2, (x, y), (x+w, y+h), (0,255,0), 2)
#cv2.imwrite('imgdiff/Morphydiff33.jpg',img1)
cv2.imwrite('imgdiff/Morphydiff3334.jpg',img2)
# cv2.imshow("Differences", img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

def remove_background_with_green_contours(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading the image.")
        return None

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for green color in HSV
    lower_green = np.array([40, 40, 40])  # Adjust these values based on your specific green
    upper_green = np.array([80, 255, 255])

    # Create a mask using the green color range
    mask = cv2.inRange(hsv_image, lower_green, upper_green)
    mask = cv2.dilate(mask, None, iterations=2)  # Dilate the mask to fill gaps

    # Optionally, clean up the mask using morphological operations
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # Create an output image that keeps the green area with optional transparency
    output = np.zeros_like(image)
    output[:, :, :] = 255  # Set background to white or any color

    # Use the mask to combine the original image with the background
    for i in range(3):  # Copy color channels
        output[:,:,i] = np.where(mask==255, image[:,:,i], output[:,:,i])

    # Save or display the result
    cv2.imwrite('output_with_green_contours_only.png', output)
    # cv2.imshow('Output with Green Contours Only', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return output

# Replace 'path_to_your_image.jpg' with your image file path
remove_background_with_green_contours('imgdiff/Morphydiff3334.jpg')

def label_chessboard(image_path):
    # Load the image
    image = cv2.imread(image_path)
    grid_size = 8
    square_height = image.shape[0] // grid_size
    square_width = image.shape[1] // grid_size
    
    # Annotate each square with shifted labels
    for i in range(grid_size):
        for j in range(grid_size):
            label = f"{chr(65 + j)}{8 - i}"
            # Shift text to the right by adding an offset to the x-coordinate
            cv2.putText(image, label, (j * square_width + 30, (i + 1) * square_height - 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

    # Resize image for display if it's too large
    display_width = 800  # Set this to the maximum width that fits your display
    display_height = int(image.shape[0] * (display_width / image.shape[1]))
    resized_image = cv2.resize(image, (display_width, display_height))
    cv2.imwrite('imglabel/Morphylabeltry.jpg', resized_image)
    # Display the resized image
    # cv2.imshow('Labeled Chessboard', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Replace 'path_to_chessboard_image.jpeg' with the path to your chessboard image
label_chessboard('output_with_green_contours_only.png')

def detect_and_crop_green_areas(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error loading the image.")
        return

    # Convert image to HSV color space for easier color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range of green color in HSV
    lower_green = np.array([40, 40, 40])  # Lower HSV threshold for green
    upper_green = np.array([80, 255, 255])  # Upper HSV threshold for green

    # Create a mask that only contains green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Optional: Dilate the mask to cover gaps
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cropped_images = []

    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Adjust this value as needed
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = image[y:y+h, x:x+w]
            cropped_images.append(cropped_image)

    # Concatenate images horizontally
    if cropped_images:
        max_height = max(img.shape[0] for img in cropped_images)
        resized_images = [cv2.resize(img, (int(img.shape[1] * max_height / img.shape[0]), max_height)) for img in cropped_images]
        final_image = np.hstack(resized_images)
        # Save the concatenated image
        cv2.imwrite('boxcrop\concatenated_image.jpg', final_image)
        print("Concatenated image saved as 'concatenated_image.jpg'")
    else:
        print("No cropped images to concatenate.")

# Call the function with the path to your image
image_path = 'imglabel/Morphylabeltry.jpg'
detect_and_crop_green_areas(image_path)

#Read the red characters present inside the green boxes. If the characters are not exactly visible or are overlapping with the green box. Deduce them based on the other characters in the image. Give the answer in the form['','']
def move_difference(png1):
    model = genai.GenerativeModel("gemini-pro-vision")
    image1 = Image.open(png1)
    #image2 = Image.open(png2)
    response = model.generate_content([image1, f'''Read the red characters present inside the green boxes. Give the answer in the form['','']'''])
    return response.text

move_difference("boxcrop/concatenated_image.jpg")
