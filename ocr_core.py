try:
    from PIL import Image
    import numpy as np
    import cv2
    import imutils
except ImportError:
    import Image

import pytesseract


def ocr_core(filename):
    """
    This function will handle the core OCR processing of images.
    """
    #text = pytesseract.image_to_string(Image.open(filename))  # We'll use Pillow's Image class to open the image and pytesseract to detect the string in the image
    #return text  # Then we will print the text in the image


    # Read the image file
    #image = cv2.imread('Car Images/7.jpg')

    # Resize the image - change width to 500
    image = imutils.resize(filename, width=500)
    #image=filename

    # Display the original image
    #cv2.imshow("Original Image", image)
    #cv2.waitKey(0)

    # RGB to Gray scale conversion
    gray = cv2.cvtColor(filename, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(cv2.UMat(imgUMat), cv2.COLOR_RGB2GRAY)
    #cv2.imshow("1 - Grayscale Conversion", gray)
    #cv2.waitKey(0)

    # Noise removal with iterative bilateral filter(removes noise while preserving edges)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    #cv2.imshow("2 - Bilateral Filter", gray)
    #cv2.waitKey(0)

    # Find Edges of the grayscale image
    edged = cv2.Canny(gray, 170, 200)
    #cv2.imshow("3 - Canny Edges", edged)
    #cv2.waitKey(0)

    # Find contours based on Edges
    cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Create copy of original image to draw all contours
    img1 = image.copy()
    cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)
    #cv2.imshow("4- All Contours", img1)
    #cv2.waitKey(0)

    # sort contours based on their area keeping minimum required area as '30' (anything smaller than this will not be considered)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
    NumberPlateCnt = None  # we currently have no Number plate contour

    # Top 30 Contours
    img2 = image.copy()
    cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)
    #cv2.imshow("5- Top 30 Contours", img2)
    #cv2.waitKey(0)

    # loop over our contours to find the best possible approximate contour of number plate
    count = 0
    idx = 7
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # print ("approx = ",approx)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour

            # Crop those contours and store it in Cropped Images folder
            x, y, w, h = cv2.boundingRect(c)  # This will find out co-ord for plate
            new_img = gray[y:y + h, x:x + w]  # Create new image
            cv2.imwrite('Cropped Images-Text/' + str(idx) + '.png', new_img)  # Store new image
            idx += 1

            break

    # Drawing the selected contour on the original image
    # print(NumberPlateCnt)
    cv2.drawContours(image, [NumberPlateCnt], -1, (0, 255, 0), 3)
    #cv2.imshow("Final Image With Number Plate Detected", image)
    #cv2.waitKey(0)

    #Cropped_img_loc = 'Cropped Images-Text/7.png'
    #cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))

    # Use tesseract to covert image into string
    text = pytesseract.image_to_string(image, lang='eng')
    #print("Number is :", text)
    return text

    #cv2.waitKey(0)  # Wait for user input before closing the images displayed
