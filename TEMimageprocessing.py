import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import skeletonize
from matplotlib.patches import Rectangle
from scipy.stats import skew
from PIL import Image, ImageTk, ImageDraw
from paddleocr import PaddleOCR
import easyocr
import statistics
from  matplotlib.colors import LinearSegmentedColormap
import re


def nanoImagePreprocessing(image, threshold, plot):

    grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    grayThresholdedImage = (grayImage < threshold).astype(float)

    if plot:
        fig, axs = plt.subplots(1, 3, figsize = (12,6))
        axs[0].imshow(image, cmap = "gray")
        axs[0].set_title("Original Image")
        axs[0].axis("off")

        axs[1].imshow(grayImage, cmap = "gray")
        axs[1].set_title("Gray Image")
        axs[1].axis("off")

        axs[2].imshow(grayThresholdedImage, cmap = "gray")
        axs[2].set_title("Gray Thresholded Image")
        axs[2].axis("off")
        plt.show()
    return grayThresholdedImage


def contourExtractor(thresholdedImage):

    if np.max(thresholdedImage) == 1:
        # Load your binary image (0 or 1 values)
        binaryImage = thresholdedImage.astype(np.uint8) * 255
    else:
        binaryImage = thresholdedImage

    skeleton = skeletonize(binaryImage)
    skeleton255 = (255 * skeleton) % 256
    skeleton255 = skeleton255.astype(np.uint8)
    return skeleton255


def getPixelNumberForScale(thresholdedImage, isScaleBlack, printInfo, plotRuler):
    # Load the binary image (white pixels on a black background)
    if np.max(thresholdedImage) == 1:
        binaryImage = thresholdedImage * 255

    # Initialize variables to keep track of the longest run
    longestWhiteRunLength = 0
    longest_run_start = 0
    longest_run_end = 0
    longest_run_row = 0

    # Iterate through each row in the binary image
    current_run_length = 0
    current_run_start = 0
    current_run_end = 0

    if isScaleBlack: 
        runCheckColorValue = 255
    else: 
        runCheckColorValue = 0

    for row in range(binaryImage.shape[0]):
        inRun = False  # Flag to track if we are in a white pixel run
        for col in range(binaryImage.shape[1]):
            pixelValue = binaryImage[row, col]    

            if int(pixelValue) == runCheckColorValue:  # Check for a white or black pixel (255 in grayscale)
                if not inRun:
                    current_run_length = 0
                    current_run_start = col
                    inRun = True
                current_run_length += 1
                current_run_end = col

            else:
                if current_run_length >= longestWhiteRunLength:
                    longestWhiteRunLength = current_run_length
                    longest_run_start = current_run_start
                    longest_run_end = current_run_end
                    longest_run_row = row
                inRun = False
                current_run_length = 0

    if printInfo:
        print("Longest White Pixel Run:")
        print("Start Column:", longest_run_start)
        print("End Column:", longest_run_end)
        print("Length:", longestWhiteRunLength)
        print("Row index:", longest_run_row)
    rulerImage = binaryImage.copy()
    if isScaleBlack: 
        rulerImage[:, :longest_run_start] = 0
        rulerImage[:, longest_run_end + 1:] = 0
        rulerImage[:longest_run_row, :] = 0
        rulerImage[longest_run_row + 1:, :] = 0
    else:
        rulerImage[:, :longest_run_start] = 255
        rulerImage[:, longest_run_end + 1:] = 255
        rulerImage[:longest_run_row, :] = 255
        rulerImage[longest_run_row + 1:, :] = 255

    if plotRuler:
        # Create a copy of the binary image
        plt.figure(figsize=(2,2))
        plt.imshow(rulerImage, cmap="gray")
        plt.title("Extract Ruler Lenght Image")
        plt.show()
    return longestWhiteRunLength, rulerImage


def rotateImage(image, rotationAngle):
    imageArray = np.array(image)
    imageWidth, imageHeight = imageArray.shape[1], imageArray.shape[0]  #for gray
    rotationMatrix = cv2.getRotationMatrix2D((imageWidth / 2, imageHeight / 2), rotationAngle, 1)
    imageDraw = cv2.warpAffine(imageArray, rotationMatrix, (imageWidth, imageHeight))
    return imageDraw

def magnifyImage(image, magnificationFactorSet):
    if not isinstance(image, Image.Image):
        ImagePIL = Image.fromarray(image)
    else:
        ImagePIL = image
    width, height = ImagePIL.size
    magnifWidth = int(width * float(magnificationFactorSet))
    magnifHeight = int(height * float(magnificationFactorSet))
    magnifiedImagePIL = ImagePIL.resize((magnifWidth, magnifHeight))
    return magnifiedImagePIL

def drawAndColorBoxes(rotatedImage, magnificationFactorSet, blurFactorSet, distanceBaseinNm, longestWhiteRunLength, nanoResolution, medianPixelValue):
    longestWhiteRunLength = longestWhiteRunLength * magnificationFactorSet

    rotatedImageCopy = rotatedImage.copy()

    ########magnifying the rotatedImageCopy########
    if len(rotatedImageCopy.shape) == 2:
        pass
    else:
        rotatedImageCopy = cv2.cvtColor(rotatedImageCopy, cv2.COLOR_BGR2GRAY)
    rotatedImagePIL = Image.fromarray(rotatedImageCopy)

    ####debugging####
    if rotatedImagePIL == None:
        print('the image is None')
    else:
        pass

    magnifiedRotatedImage =  magnifyImage(rotatedImagePIL, magnificationFactorSet)
    
    binaryThresholdedSegmentImage = (magnifiedRotatedImage > medianPixelValue).astype(float)
    binaryThresholdedSegmentImageArr = np.array(binaryThresholdedSegmentImage)

    
    skeleton255 = contourExtractor(binaryThresholdedSegmentImageArr)
    # plt.imsave("skeleton255.jpg", skeleton255, cmap='gray')
    newImageWithColoringRGB, percentageList, colorList = coloring(skeleton255, distanceBaseinNm, longestWhiteRunLength, nanoResolution, blurFactorSet)
    
    return newImageWithColoringRGB, percentageList, colorList

def assign_color_and_modify_image(image, entry, colorList, percentageList, distanceBaseinPixels, longestWhiteRunLength, nanoResolution):
    row_index, col_index1, col_index2 = entry
    distance = col_index2 - col_index1 
    
    distancePerPixel = nanoResolution / longestWhiteRunLength  #in nm
    distanceBaseinNm = distanceBaseinPixels * distancePerPixel
    # Define color logic based on segment length (example logic)
    color, signedPercentage = distanceColorMapper(distance, distanceBaseinNm, longestWhiteRunLength, nanoResolution)

    # Assign color to the image segment
    image[row_index, col_index1:col_index2] = color
    colorList.append(color)
    percentageList.append(signedPercentage)

    return image  # Return modified image

def process_image(image, index_list, distanceBaseinPixels, longestWhiteRunLength, nanoResolution):
    colorList = []  # List to store the colors
    percentageList = []  # List to store the colors
    
    # Use map to apply the assignment function to each entry in the list
    list(map(lambda entry: assign_color_and_modify_image(image, entry, colorList, percentageList, distanceBaseinPixels, longestWhiteRunLength, nanoResolution), index_list))
    
    return image, colorList, percentageList  # Return modified image and color list



def coloring(skeleton255, distanceBaseinNm, longestWhiteRunLength, nanoResolution, blurFactorSet):
    newImageWithColoring = skeleton255.copy()
    newImageWithColoringRGB = cv2.cvtColor(newImageWithColoring, cv2.COLOR_GRAY2RGB)
    percentageList = list()
    colorList = list()
    if (distanceBaseinNm == -1):
        segmentIndexList = list()
    else:
        distanceList = list()

    for rowIndex in range(skeleton255.shape[0]):
        oldPixel = 255
        distanceStartIndex = 0
        distanceEndIndex = 0

        for columnIndex in range(skeleton255.shape[1]):
            newPixel = skeleton255[rowIndex][columnIndex]
            if ((oldPixel == 0) & (newPixel == 255)):
                distanceEndIndex = columnIndex
                if distanceBaseinNm == -1:
                    segmentIndexList.append([rowIndex, distanceStartIndex, distanceEndIndex])
                else:
                    distance = distanceEndIndex - distanceStartIndex
                    distanceList.append(distance)

                    color, signedPercentage = distanceColorMapper(distance, distanceBaseinNm, longestWhiteRunLength, nanoResolution)
                    newImageWithColoringRGB[rowIndex, distanceStartIndex: distanceEndIndex] = color
                    
                    percentageList.append(signedPercentage * 100)
                    colorList.append(color)
            elif (oldPixel == 255) & (newPixel == 0):
                distanceStartIndex = columnIndex - 1
            oldPixel = newPixel
    if distanceBaseinNm == -1:
        distanceBaseinPixels = int(statistics.median(k - j for i, j, k in segmentIndexList))
        maximumDistance = max(k - j for i, j, k in segmentIndexList)

        newImageWithColoringRGB_, colorList, percentageList = process_image(newImageWithColoringRGB, segmentIndexList, distanceBaseinPixels, longestWhiteRunLength, nanoResolution)
    else:
        newImageWithColoringRGB_ = newImageWithColoringRGB
        
    newImageWithColoringRGB__ = cv2.GaussianBlur(newImageWithColoringRGB_, (blurFactorSet, blurFactorSet), 0)
    return newImageWithColoringRGB__, percentageList, colorList


def getEasyOCRPrediction(img, model):#img
    if model == 'easy':
        print('ocr model is easyocr')
        reader = easyocr.Reader(['en'], gpu=False)
    else:
        print('ocr model is paddleocr')
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    if img is None:
        raise ValueError("Error loading the image. Please check the file path.")
    # Get the height of the image

    height = img.shape[0]

    # Extract the lower half of the image
    img = img[height // 2:, :]
    if model == 'easy':
        text_detections = reader.readtext(img) #for easyocr
    else:
        text_detections = ocr.ocr(img) #for paddle
    

    image_height = img.shape[0]

    # Filter detections
    filtered_result = []
    try:
        for item in text_detections:
            if item[-2]:  # Ensure there is text detected
                match = re.search(r"\d+", item[-2])  # Find the first number in the text
                if match:
                    filtered_result.append(match.group()[0])
    except:
        filtered_result = []

    return filtered_result


def find_longest_line(img, black_threshold = 25, white_threshold = 240):
    best_line = None

    max_length = 0
    imageGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imageBlackThres = (imageGray < black_threshold).astype(float)
    imageWhiteThres = (imageGray > white_threshold).astype(float)

    for imageIndex, image in enumerate([imageBlackThres, imageWhiteThres]):
        height, width = image.shape
        for i in range((height//2), height):
            current_length = 0
            start_j = None

            for j in range(width):
                pixel_value = image[i, j]
                
                if int(pixel_value) == 1:

                    if current_length == 0:
                        start_j = j
                    current_length += 1
                
                else:
                    if (current_length > max_length) :
                        max_length = current_length
                        best_line = (i, start_j, i, start_j + current_length - 1)
                    current_length = 0
            
            # Check last segment in the row
            if (current_length > max_length) and ((current_length < height*0.6) and (current_length < width*0.6)):
                max_length = current_length
                (y1, x1, y2, x2) = (i, start_j, i, start_j + current_length - 1)
                best_line = (y1, x1, y2, x2)
    
    if best_line:
        i, start_j, _, end_j = best_line
        y1, x1, y2, x2 = best_line
    lineLengthInPixelNum = x2 - x1
    return (y1, x1, y2, x2), lineLengthInPixelNum


def drawBoundingBoxes(image, detections, threshold=False):
    height, width = image.shape[0], image.shape[1]
    if threshold:
        for bbox, text, score in detections:
            if score > threshold:
                cv2.rectangle(image, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 5)
    else:
        cv2.rectangle(image, detections[0], detections[1], (0, 255, 0), 2)

    return image


def distanceColorMapper(distance, distanceBaseinNm, longestWhiteRunLength, nanoResolution):
    distancePerPixel = nanoResolution / longestWhiteRunLength  #in nm

    distanceBase = int(distanceBaseinNm / distancePerPixel) # how many pixels
    signedPercentage = ((distance - distanceBase) / distanceBase)
    cmap=LinearSegmentedColormap.from_list('Interatomic distance coloring rule',['b', "g", "r"], N = 256) 
    
    if (distance > distanceBase):
        if signedPercentage > 100:
            color_255 = (0,0,0)
        else:
            color = cmap(int(128 + 128 * signedPercentage))[:3]       # Color at index 0 (start of the colormap)
            color_255 = tuple(255 * x for x in color)

    elif (distance < distanceBase):
        if signedPercentage < -100:
            color_255 = (8, 36, 176)
        else:
            color = cmap(int(128 + 128 * signedPercentage))[:3]
            color_255 = tuple(255 * x for x in color)
    
    else:
        color = cmap(128)[:3]
        color_255 = tuple(255 * x for x in color)
    return color_255, signedPercentage