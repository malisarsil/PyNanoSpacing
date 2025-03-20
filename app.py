from PIL import Image,ImageTk, ImageDraw
from tkinter import filedialog
from tkinter import Canvas
import tkinter as tk
from tkinter import ttk
from tkinter import font

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from scipy.fft import fft2, fftshift
from ultralytics import SAM
import TEMimageprocessing

class SampleApp(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Defining global class variables.
        ###Title of the application window
        self.title("PyNanospacing - TEM Image Strain Analysis App")

        self.panelWidth, self.panelHeight = 400, 300
        self.originalUploadImage = None
        self.originalImageRGB = None
        self.originalImagePIL = None

        self.magnifiedImage = None
        self.rotatedImage = None
        self.binaryThresholdedImage = None
        
        #Variables for metadata of the TEM image
        self.imageResolution = None
        self.longestWhiteRunLength = 0
        self.referenceDistanceInNm = None

        # Image Magnification Factor to enhance the output image and have more resolution during the interplanar distance calculation.
        self.magnificationFactor = None

        self.showFrame(TEMimageUpload)


    def showFrame(self, cont, *args, **kwargs):

        # Destroy any existing frames to ensure only one frame is displayed
        for widget in self.winfo_children():
            widget.destroy()

        # Raise the specified frame to the top
        frame = cont(self, self, *args, **kwargs)  # Pass imageResult to the next frame
        frame.grid(row=0, column=0, sticky="nsew")  # Use grid instead of pack


    def rescaleImageForDisplay(self, imageToDisplay, panelWidth, panelHeight):

        # Get the dimensions of the magnified image
        if isinstance(imageToDisplay, Image.Image) == False:
            imageToDisplay = Image.fromarray(imageToDisplay)
        imageWidth, imageHeight = imageToDisplay.width, imageToDisplay.height

        # Calculate aspect ratio
        aspectRatio = imageWidth / imageHeight

        # Calculate new dimensions while maintaining aspect ratio
        if imageWidth > panelWidth or imageHeight > panelHeight:

            if aspectRatio > 1:  # Image is wider than tall
                imageWidth = panelWidth
                imageHeight = int(panelWidth / aspectRatio)
            
            else:  # Image is taller than wide or square
                imageHeight = panelHeight
                imageWidth = int(panelHeight * aspectRatio)

        else:
            # Upscale the image while preserving the aspect ratio
            widthScale = panelWidth / imageWidth
            heightScale = panelHeight / imageHeight
            scaleFactor = min(widthScale, heightScale)  # Choose the smallest factor to fit in the panel

            imageWidth = int(imageWidth * scaleFactor)
            imageHeight = int(imageHeight * scaleFactor)


        # Resize the image to fit the panel while maintaining aspect ratio
        rescaledImageForDisplay = imageToDisplay.resize((imageWidth, imageHeight), Image.LANCZOS)
        return rescaledImageForDisplay, imageWidth, imageHeight

    def drawToPanel(self, imageToDisplay, displayPanel):
        imageToDisplay, updatedWidth, updatedHeight = self.rescaleImageForDisplay(imageToDisplay, self.panelWidth, self.panelHeight)

        imageDisplay = ImageTk.PhotoImage(imageToDisplay)
        displayPanel.config(image=imageDisplay)
        displayPanel.image = imageDisplay 
        return updatedWidth, updatedHeight




class TEMimageUpload(tk.Frame):
    def __init__(self, parent, controller, **kwargs):
        super().__init__(parent)


        self.label = tk.Label(self, text="Step I: Uploading TEM Image", font=("Arial", 25))
        self.label.grid(row=0, column=1, columnspan=2, padx=5, pady=5)

        self.leftImage = tk.Label(self, text="")
        self.leftImage.grid(row=1, column=0, columnspan=2,  padx=5, pady=5)

        self.parent = parent

        self.imageScaler = tk.StringVar()  
        self.referenceDistanceinNM = tk.StringVar()

        self.imageUploadButton = tk.Button(self, text = "Upload TEM Image", fg = "black", bg = "lavender", command=self.imageUpload, font=("Arial", 14, "bold"))
        self.imageUploadButton.grid(row=1, column=1, columnspan=2, padx=10, pady=10)


    def imageUpload(self):
        fileTypes = [('Jpg Files', '*.jpg'),('PNG Files','*.png'), ('Jpeg Files', '*.jpeg')]
        filePath = filedialog.askopenfilename(filetypes = fileTypes)
        self.parent.originalUploadImage = cv2.imread(filePath)

        self.parent.originalImageRGB = cv2.cvtColor(self.parent.originalUploadImage, cv2.COLOR_BGR2RGB)
        self.parent.originalImagePIL = Image.fromarray(self.parent.originalImageRGB)
        
        self.parent.drawToPanel(self.parent.originalImagePIL, self.leftImage)

        # Remove the upload button
        self.imageUploadButton.destroy()

        self.label.configure(text="Extracting the ruler length in pixels and image scale...", font=("Arial", "15"))
        self.update_idletasks()

        #here comes the number bounding box coordinates.
        text_detections = TEMimageprocessing.getEasyOCRPrediction(self.parent.originalImageRGB, model = 'easy')
        if len(text_detections) > 0:
            self.imageScaler.set(str(text_detections[0]))
        else:
            self.imageScaler.set('')

        #here comes the ruler bounding box coordinates.
        (y1, x1, y2, x2), lineLengthInPixelNum  = TEMimageprocessing.find_longest_line(self.parent.originalImageRGB)
        self.parent.longestWhiteRunLength = lineLengthInPixelNum
        boundingBoxImage = TEMimageprocessing.drawBoundingBoxes(self.parent.originalImageRGB, [(x1, y1), (x2, y2)])
        self.parent.drawToPanel(self.parent.originalImagePIL, self.leftImage)

        self.boundingBoxImageArr= Image.fromarray(boundingBoxImage)
        self.rightImage = tk.Label(self, text = "Updated Image:")
        self.rightImage.grid(row=1, column=2, columnspan=2, padx=5, pady=5)
        self.parent.drawToPanel(self.boundingBoxImageArr, self.rightImage)
    
        self.label.configure(text="Extracted the ruler length in pixels and image scale...", font=("Arial", "15"))

        imageScalerInput = tk.Label(self, text = "TEM Image Scale (nm):")
        imageScalerInput.grid(row=2, column=0, columnspan=2, padx=2, pady=2, sticky="w")
        referenceDistanceinNM = tk.Label(self, text="Reference Distance in nm:")
        referenceDistanceinNM.grid(row=2, column=2, columnspan=2, padx=2, pady=2, sticky="w")
        
        self.imageResolutionLabel = tk.Entry(self, textvariable=self.imageScaler)
        self.imageResolutionLabel.grid(row=3, column=0, columnspan=2, padx=2, pady=2, sticky="ew")
        self.referenceDistanceLabel = tk.Entry(self, textvariable=self.referenceDistanceinNM)
        self.referenceDistanceLabel.grid(row=3, column=2, columnspan=2, padx=2, pady=2, sticky="ew")
        

        button1 = tk.Button(self, text="Update Ruler", command = lambda: self.goToNextFrame(function = 'update'))
        button1.grid(row=8, columnspan= 2, column=0, pady=5)  # Adjust row and column as needed

        button2 = tk.Button(self, text="Continue", command=lambda: self.goToNextFrame(function = 'continue'))
        button2.grid(row=8, columnspan= 2, column=2, pady=5)  # Adjust row and column as needed


    def goToNextFrame(self, function):
        try:
            imageResolution = self.imageScaler.get()
            self.parent.imageResolution = float(imageResolution)
            referenceDistance = self.referenceDistanceinNM.get()
            self.parent.referenceDistanceInNm = float(referenceDistance)

            if function == 'continue':
                self.parent.showFrame(extractTEMRegionsandColor)

            else:
                self.parent.showFrame(scalePixelLengthExtraction) 

        except:
            self.label.configure(text="Please enter the image scale and the reference distance values correctly first!")


class extractTEMRegionsandColor(tk.Frame):
    def __init__(self, parent, controller, dynamicText= "Will be Extracting the region of interest. Please wait!", *args, **kwargs):
        super().__init__(parent)
        self.controller = controller
        self.parent = parent

        self.label = tk.Label(self, text="Step II: Extracting Image regions and apply coloring based on strain ", font=("Arial", 25))
        self.label.grid(row=0, column=1, columnspan=2, padx=2, pady=2, sticky="w")

        self.label1 = tk.Label(self, text="Original Uploaded Image")
        self.label1.grid(row=1, column=0, columnspan=2, padx=5, pady=5)
        self.label1.bind("<Button-1>", lambda event: self.onLeftClick(event, self.label1))
        self.label1.bind("<Button-2>", lambda event: self.onRightClick(event, self.label1))

        image1 = self.parent.originalUploadImage
        
        self.originalImagePILCopy = self.parent.originalImagePIL.copy()
        self.dummyImage = self.parent.originalImagePIL.copy()
        
        self.regionDict = dict()
        self.regionDict['region coordinates'] = []
        self.percentageList = list()
        self.colorList = list()

        self.updatedWidth, self.updatedHeight = self.parent.drawToPanel(self.parent.originalImagePIL, self.label1)
        self.clickCoordinatesDict = dict()
        
        self.width, self.height = self.parent.originalImagePIL.size
        self.xScaler = self.width / self.updatedWidth
        self.yScaler =  self.height / self.updatedHeight


        self.label0 = tk.Label(self, text= dynamicText, font=("Arial", 12, "italic"))
        self.label0.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="ew")
        self.initialRadiusCoef = 0.02
        self.dot_radius = self.width * self.initialRadiusCoef  # adjust the ratio


        self.button = tk.Button(self, text="Extract Region", command=self.extract)
        self.button.grid(row = 7, column = 2, columnspan = 2, padx = 5, pady=5)  # Adjust row and column as needed

    
    def onLeftClick(self, event, label):

        x = event.x
        y = event.y
        
        newx = x*self.xScaler
        newy = y*self.yScaler
        close_points = self.findPointsCloseToClick(newx, newy)

        if len(close_points) > 0:
            closePointKey = f'{close_points[0][0]}-{close_points[0][1]}'
            if closePointKey in self.clickCoordinatesDict.keys():
                self.clickCoordinatesDict[closePointKey][1] = 'lime'
                currentRadiusCoef = self.clickCoordinatesDict[closePointKey][0]
                self.drawPoints(label)
                
                self.dotMagnify = ttk.Scale(self, from_ = 0.01, to=0.8, orient="horizontal", command=lambda value: self.dotRadiusHandling(value, closePointKey, label))
                self.dotMagnify.set(currentRadiusCoef)
                self.dotMagnify.grid(row = 5, column = 2, columnspan=1, padx=(5, 20), pady=5, sticky="ew")
                self.approvebutton = tk.Button(self, text="Confirm Region", command=lambda: self.removeDotComponents(closePointKey, label))
                self.approvebutton.grid(row=5, column=3, columnspan = 2, padx = 5, pady=5)  # Adjust row and column as needed

        else:
            self.clickCoordinatesDict[f'{newx}-{newy}'] = []
            self.clickCoordinatesDict[f'{newx}-{newy}'].append(self.initialRadiusCoef)
            self.clickCoordinatesDict[f'{newx}-{newy}'].append('red')
            self.drawPoints(label)

    def onRightClick(self, event, label):
        # Handle right mouse button click
        x = event.x
        y = event.y
        newx = x*self.xScaler
        newy = y*self.yScaler

        if len(self.clickCoordinatesDict.keys()) == 0:
            pass
        else:
            close_points = self.findPointsCloseToClick(newx, newy)
            if len(close_points) != 0:
                self.clickCoordinatesDict = {x_y_key: [self.clickCoordinatesDict[x_y_key][0], self.clickCoordinatesDict[x_y_key][1]] for x_y_key in self.clickCoordinatesDict.keys() if (float(x_y_key.split('-')[0]), float(x_y_key.split('-')[1])) not in close_points}
                for [x, y] in close_points:
                    del self.clickCoordinatesDict[f'{x}-{y}']
                self.drawPoints(label)

            else:
                pass

    def removeDotComponents(self, closePointKey, label):
        self.clickCoordinatesDict[closePointKey][1] = 'red'
        self.approvebutton.destroy()
        self.dotMagnify.destroy()
        self.update_idletasks()
        self.drawPoints(label)
    

    def drawPoints(self, label):
        self.dummyImage = self.originalImagePILCopy.copy()
        draw = ImageDraw.Draw(self.dummyImage)
        for x_y_key in self.clickCoordinatesDict.keys():
            newx, newy = float(x_y_key.split('-')[0]), float(x_y_key.split('-')[1])
            radiusCoef = self.clickCoordinatesDict[x_y_key][0]
            dot_radius = self.width * radiusCoef
            color = self.clickCoordinatesDict[x_y_key][1]
            
            draw.ellipse((newx - dot_radius, newy - dot_radius, newx + dot_radius, newy + dot_radius), fill=color)
            
        self.parent.drawToPanel(self.dummyImage, label) #this is where the maskes generated by sam will be visualized.
    
    def findPointsCloseToClick(self, newx, newy):
        close_points = []
        for x_y_key in self.clickCoordinatesDict.keys():
            distanceThreshold = self.clickCoordinatesDict[x_y_key][0]*self.width * 1.2
            (x, y) = float(x_y_key.split('-')[0]), float(x_y_key.split('-')[1])
            # Calculate Euclidean distance
            distance = math.sqrt((x - newx) ** 2 + (y - newy) ** 2)
            
            # Check if distance is below the threshold
            if distance < distanceThreshold:
                close_points.append([x, y])

        return close_points

    def dotRadiusHandling(self, value, closePointKey, label):
        
        newRadiusCoeff = float(value)
        newRadius = newRadiusCoeff * self.width
        oldRadiusCoeff = self.dotMagnify.get()
        self.clickCoordinatesDict[closePointKey][0] = newRadiusCoeff

        self.drawPoints(label)

        if (newRadiusCoeff != oldRadiusCoeff):
            self.dotMagnify.set(newRadiusCoeff)


    def plot_mask_on_image(self, original_image, masks):
        """
        Plot the masks on the original image.

        :param original_image: PIL Image, the original image.
        :param masks: NumPy array, the array of masks to overlay.
        """

        # Convert the original PIL image to a NumPy array for plt.imshow
        original_image_np = np.array(original_image)

        # Create a blank image for the overlay
        maskOverlay = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(maskOverlay)

        # Iterate over all masks
        for mask in masks:
            # Convert the mask to a PIL image
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            
            # Set the color and transparency for the mask overlay
            mask_colored = Image.new("RGBA", mask_image.size, (255, 0, 0, 100))  # Red color with 100 alpha

            # Paste the mask on the blank overlay image
            maskOverlay = Image.composite(mask_colored, maskOverlay, mask_image)

        # Convert the overlay to a NumPy array
        maskOverlaynp = np.array(maskOverlay)
        
        # Combine the original image with the overlay
        combinedImagenp = np.array(original_image.convert("RGBA"))  # Ensure original image is in RGBA
        combinedImagenp[:, :, :3] = (combinedImagenp[:, :, :3] * (1 - (maskOverlaynp[:, :, 3:] / 255))).astype(np.uint8) + (maskOverlaynp[:, :, :3] * (maskOverlaynp[:, :, 3:] / 255)).astype(np.uint8)


        return combinedImagenp

    def extract(self):
        if len(self.clickCoordinatesDict) == 0:
            self.label0.configure(text= "Please select at least one region first!")
            pass

        else:
            self.label0.configure(text="Extracting the region of interest. Please wait!", font=("Arial", "15"))
            self.update_idletasks()

            self.model = SAM("sam2_b.pt")

            SAMpoints = list()
            customPoints = list()
            image_width, image_height = self.parent.originalImagePIL.size
            all_masks = np.zeros((1, image_height, image_width), dtype=bool)
            
            for coordinateDict, attribute in self.clickCoordinatesDict.items():
                x, y = float(coordinateDict.split('-')[0]), float(coordinateDict.split('-')[1])
                if attribute[0] != self.initialRadiusCoef:
                    mask = np.zeros((image_height, image_width), dtype=bool)
                    customPoints.append([x, y])

                    center = (int(x), int(y))
                    radius = int(self.width * attribute[0])
                    circle_mask = np.zeros_like(mask, dtype=np.uint8)
                    cv2.circle(circle_mask, center, radius, color=1, thickness=-1)
                    mask[circle_mask.astype(bool)] = True
                    new_mask_expanded = mask[np.newaxis, :, :]
                    all_masks = np.vstack((all_masks, new_mask_expanded))

                else:
                    SAMpoints.append([x, y])
            
            self.clickCoordinatesDict = dict()
            self.drawPoints(self.label1)
            self.label1.unbind("<Button-1>")
            self.label1.unbind("<Button-2>")


            ####SAM Results#####
            if len(SAMpoints) != 0:

                results = self.model(self.parent.originalImagePIL, points=SAMpoints, labels= [1] * len(SAMpoints))
                all_masksSAM = results[0].masks.data.numpy()  # Convert mask data to NumPy array
                if np.sum(all_masks.astype(float)) != 0:
                    all_masks = np.concatenate((all_masks, all_masksSAM), axis=0)
                else:
                    all_masks = all_masksSAM
            else:
                pass

            combinedMaskedImage = self.plot_mask_on_image(self.parent.originalImagePIL, all_masks)

            self.label2 = tk.Label(self, text="Updated Image:")
            self.label2.grid(row=1, column=2, columnspan=2, padx=5, pady=5)
            self.label2.bind("<Button-1>", lambda event: self.onLeftClick(event, self.label2))
            self.label2.bind("<Button-2>", lambda event: self.onRightClick(event, self.label2))
            originalimageArray = np.array(self.parent.originalImagePIL)
            self.originalimageArrayDraw = originalimageArray.copy()
            self.originalimageArrayDrawWithoutText = self.originalimageArrayDraw.copy()
            ######################

            blurFactorSet = 15
            distanceBaseinNm = self.parent.referenceDistanceInNm
            longestWhiteRunLength = self.parent.longestWhiteRunLength
            nanoResolution = self.parent.imageResolution
            
            for maskIndex, mask in enumerate(all_masks):
                if (np.max(mask) != False):
                    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_array = np.array(mask_image)
                    
                    binary_mask = mask_array > 128
                    coords = np.column_stack(np.where(binary_mask))
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    cropped_image = originalimageArray[y_min:y_max, x_min:x_max]
                    cropped_mask = binary_mask[y_min:y_max, x_min:x_max]
                    white_pixels = np.column_stack(np.where(cropped_mask))

                    segment_values = originalimageArray[white_pixels[:, 0], white_pixels[:, 1]]
                    outputImage = np.copy(originalimageArray)
                    outputImage[~binary_mask] = 0

                    segmentPixels = outputImage[binary_mask]
                    medianPixelValue = np.median(segmentPixels)

                    avg_color = tuple(np.mean(segment_values, axis=0).astype(int))
                    background_color = avg_color
                    colored_image = np.array(cropped_image)
                    colored_image[~cropped_mask] = background_color  # Set background to average color
                    masked_image = Image.fromarray(colored_image)
                    try:
                        rotatedImage, rotationAngle = self.getFourierSpectrum(masked_image)
                        magnificationFactorSet = int(1000 / (y_max - y_min)) if (y_max - y_min) < 1000 else 1

                        newImageWithColoringRGB, percentageList, colorList = TEMimageprocessing.drawAndColorBoxes(rotatedImage, magnificationFactorSet = magnificationFactorSet, blurFactorSet = blurFactorSet,
                                                                        distanceBaseinNm = distanceBaseinNm, longestWhiteRunLength = longestWhiteRunLength , nanoResolution = nanoResolution, medianPixelValue = medianPixelValue)
                        deMagnifiedImage = TEMimageprocessing.magnifyImage(newImageWithColoringRGB, 1 / magnificationFactorSet)
                        
                        deRotatedImage = TEMimageprocessing.rotateImage(deMagnifiedImage, -rotationAngle)
                        cropped_image[np.where(cropped_mask == True)] = deRotatedImage[np.where(cropped_mask == True)]

                        self.originalimageArrayDraw[y_min:y_max, x_min:x_max] = cropped_image
                        self.originalimageArrayDrawWithoutText[y_min:y_max, x_min:x_max] = cropped_image
                        height, width = self.originalimageArrayDraw.shape[:2]
                        font_scale = max(1.75, min(width, height) / 750)  # Adjust font size proportionally
                        thickness = int(max(2.25, int(min(width, height) / 100)))
                        text_x = x_max + 10  # Add some padding to the right
                        text_y = y_min + 30
                        cv2.putText(self.originalimageArrayDraw, str(maskIndex + 1), (text_x, text_y), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=font_scale, color=(255, 0, 0), thickness=thickness)

                        self.regionDict['region coordinates'].append([(x_min, y_min), (x_max, y_max)])

                        self.percentageList.append(percentageList)
                        self.colorList.append(colorList)

                    except Exception as e:
                        print('with exception:', e)
                        self.parent.showFrame(extractTEMRegionsandColor, dynamicText="Selected points are not eligible for further processing. Please select another region!")



            combinedMaskedImagePIL = Image.fromarray(combinedMaskedImage)
            self.originalImagePILCopy = combinedMaskedImagePIL.copy()
            self.parent.drawToPanel(combinedMaskedImagePIL, self.label2)
            self.label0.configure(text="Proceed to color the regions or reselect the region!", font=("Arial", "15"))
            self.button.config(text="Color Regions", command = self.plotColoredRegions)

            self.retry_button = tk.Button(self, text="Reselect region", command=self.retry_function)
            self.retry_button.grid(row = 7, column=0, columnspan=2, padx = 5, pady=5)
            self.retry_button.bind("<Button-1>", lambda event: self.onLeftClick(event, self.retry_button))

    def plotColoredRegions(self):
        self.label1.destroy()
        self.label2.destroy()
        self.retry_button.destroy()
        
        self.image = Image.fromarray(self.originalimageArrayDraw)
        self.imageWithoutText = Image.fromarray(self.originalimageArrayDrawWithoutText)
        imageWidth, imageHeight = self.image.size
        self.tk_image = ImageTk.PhotoImage(self.image)

        canvas_width = self.parent.panelWidth
        canvas_height = self.parent.panelWidth
        displayImage, updatedWidth, updatedHeight = self.parent.rescaleImageForDisplay(self.image, canvas_width, canvas_height)
        self.aspectCoefficientWidth = updatedWidth / imageWidth
        self.aspectCoefficientHeight = updatedHeight / imageHeight
        self.tk_image = ImageTk.PhotoImage(displayImage)

        # Create a canvas to display the image
        self.canvas = Canvas(self, width=canvas_width, height=canvas_height)
        self.canvas.grid(row=1, column=1, columnspan=2, padx=5, pady=5)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

        # Load Excel icon
        excel_icon_path = "excel.jpeg"
        excel_icon = Image.open(excel_icon_path)
        
        # Resize the icon (e.g., shrink to 50x50 pixels)
        excel_icon_resized = excel_icon.resize((25, 25), Image.Resampling.LANCZOS)
        
        self.excel_icon = ImageTk.PhotoImage(excel_icon_resized)
        self.excel_label = tk.Label(self, image=self.excel_icon)

        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<Button-1>", self.download_info)
        
        self.label0.configure(text="Extracted and colored the region of interest. Save your results!", font=("Arial", "15"))
        self.label0.grid(row=2, column=1, columnspan = 2, padx=5, pady=5)

        self.button.grid(row = 3, column = 0, columnspan = 2, padx = 5, pady=5)  # Adjust row and column as needed
        self.button.config(text="Finish", command = self.close_window)

        self.button1 = tk.Button(self, text="Save Colored Image", command=self.saveImage)
        self.button1.grid(row = 3, column = 2, columnspan = 2, padx = 5, pady=5)  # Adjust row and column as needed    

    def close_window(self):
        self.parent.destroy()  # This will close the window
    
    def retry_function(self):
        self.parent.showFrame(extractTEMRegionsandColor, dynamicText="Reselect another region to approximate the intented region(s) better.")

    def saveImage(self):
        self.label0.configure(text="Saved the colored Image to the local folder!")
        # Open a file dialog to ask the user where to save the image
        file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                   filetypes=[("PNG files", "*.png"), 
                                                              ("JPEG files", "*.jpg;*.jpeg"), 
                                                              ("All files", "*.*")])
        
        if file_path:  # Check if the user selected a file
            self.image.save(file_path)
            self.imageWithoutText.save(f'{file_path.split(".")[0]}-without-Text.png')
            for colorListIndex, colorList in enumerate(self.colorList):
                rgbValues = colorList
                values = self.percentageList[colorListIndex]

                normalizedColors = [(r / 255, g / 255, b / 255) for r, g, b in rgbValues]
                plt.figure(figsize=(5,1))
                for index, value in enumerate(values):
                    plt.axvline(x=value, color= normalizedColors[index], linestyle='solid', linewidth = 6)
                plt.xlabel("Compression or Expansion Percentages")
                plt.xticks([-150, -120, -100, -75, -50, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 50, 75, 100, 120, 150], rotation = 90)
                plt.yticks([])
                plt.xlim(-50, 50)
                plt.xlabel('Strain Level in Percentage')
                plt.tight_layout()
                plt.savefig(file_path.split('.')[0]+ f'_colorBar - Region{colorListIndex + 1}.png')
                plt.close()

    def on_mouse_move(self, event):
        # Check if the mouse is within the hover region
        for hover_regionIndex, hover_region in enumerate(self.regionDict['region coordinates']):
            if ((hover_region[0][0] * self.aspectCoefficientWidth) <= event.x) and (event.x <= (hover_region[1][0] * self.aspectCoefficientWidth)) and \
                ((hover_region[0][1] * self.aspectCoefficientHeight) <= event.y) and (event.y<= (hover_region[1][1] * self.aspectCoefficientHeight)):
                
                # Show the Excel icon at the mouse position
                self.excel_label.place(x = event.x + 150, y = event.y)
                self.downloadPercentageListIndex = hover_regionIndex
                break
            else:
                # Hide the Excel icon if outside the hover region
                self.excel_label.place_forget()

    def download_info(self, event):

        index = self.downloadPercentageListIndex
        data = self.percentageList[index]

        # Manually define the bin edges
        binEdges2 = np.arange(-150, -20, 10)  # From -100 to -10 with a step of 20
        binEdges1 = np.arange(-20, 20, 2)  # From -10 to 10 with a step of 2
        binEdges3 = np.arange(20, 150, 10)  # From 10 to 100 with a step of 10

        # Combine all bin edges into one list
        binEdges = np.concatenate([binEdges2, binEdges1, binEdges3])

        # Create a histogram based on the specified bin edges
        hist, bins = np.histogram(data, bins = binEdges)

        xLabels = [f'{binEdges[i]}-{binEdges[i+1]}' for i in range(len(binEdges) - 1)]

        # Create a bar plot
        plt.bar(range(len(hist)), hist, width=1, align='edge', color='b')

        # Add labels and title
        plt.xlabel('Percentage Bins')
        plt.ylabel('Frequency')
        plt.title('Distribution of Compression or Expansion Percentages')
        # Customize x-axis labels
        plt.xticks(np.arange(len(hist)) + 0.5, xLabels, rotation = 90)
        plt.savefig(f'Region-{index+1}_histogram_plot.png', bbox_inches='tight')
        
        # Create a DataFrame from the data
        histogramDf = pd.DataFrame({'Bin Edges': binEdges[:-1], 'Frequency': hist})
        histogramDf.to_excel(f'Region-{index+1} histogram data.xlsx', index=False)
        self.label0.configure(text="Saved the excel file of strain percentages of the selected field!")



    def getFourierSpectrum(self, image):
        imageArray = np.array(image)
        if imageArray.shape[-1] == 4:
            imageArray = imageArray[..., :3]
        if len(imageArray.shape) == 3:
            imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            
        # Compute the 2D FFT of the image
        fft_image = fft2(imageArray)
        fft_image_shifted = fftshift(fft_image)  # Shift the zero frequency component to the center

        # Compute the magnitude and phase spectra
        magnitude_spectrum = np.abs(fft_image_shifted)
        magnitude_spectrum = np.log(1 + magnitude_spectrum)
        phase_spectrum = np.angle(fft_image_shifted)

        imageWidth = magnitude_spectrum.shape[1]
        imageHeight = magnitude_spectrum.shape[0]

        midPointx, midPointy = int(imageWidth*0.5), int(imageHeight*0.5)
        
        coefficient = 0.35
        while True:
            try:
                if coefficient > 0.9:  # Replace with your actual condition
                    rotationAngle = 0
                    break
                bbox = [midPointx, midPointy, imageWidth * coefficient, imageHeight * coefficient]  # Replace with your bounding box coordinates
                x, y, width, height = bbox
                cropImage = magnitude_spectrum[int(y - height/2) : int(y + height/2), int(x - width/2) : int(x + width/2)]
                
                k = 10
                # plt.imsave('magnitude_spectrum.png', magnitude_spectrum, cmap='gray')
                # plt.imsave('cropImage.png', cropImage, cmap='gray')
                flat_indices = np.argpartition(cropImage.flatten(), -k)[-k:]
                coordinates = np.column_stack(np.unravel_index(flat_indices, cropImage.shape))
                brightest_pixels = coordinates[np.argsort(-cropImage[coordinates[:, 0], coordinates[:, 1]])]
                
                brightest_pixel = brightest_pixels[0]
                scaledCentery = brightest_pixel[0]/midPointy
                scaledCenterx =  brightest_pixel[1]/midPointx
                percentageThres = 0.1
                rotationAngle = self.fourierSpectrumGetCenterPixel(brightest_pixels, midPointy, midPointx, percentageThres = percentageThres)
                break
            except:
                print('failed fourier transform point finding for the coefficient:', coefficient, ', retrying with a different coefficient..')
                coefficient = coefficient + 0.025

        imageDraw = TEMimageprocessing.rotateImage(imageArray, rotationAngle)

    
        x = x - width / 2
        y = y - height / 2
        return imageDraw, rotationAngle

    def fourierSpectrumGetCenterPixel(self, brightPixels, midPointy, midPointx, percentageThres = 0.1):
        pixelProximityThres = 3
        for brightPixelIndex, brightPixel in enumerate(brightPixels):
            scaledCentery = brightPixel[0]/midPointy
            scaledCenterx =  brightPixel[1]/midPointx
            scaledCentersDiffPercentile = abs((scaledCentery - scaledCenterx) / (scaledCentery))
            if scaledCentersDiffPercentile < percentageThres:
                centerPixelIndex = brightPixelIndex
                centerPixelArray = brightPixel
                break
            else:
                pass
        
        filtered_brightPixels = np.array([pixel for pixel in brightPixels if not np.array_equal(pixel, centerPixelArray)])
        
        filtered_brightPixels = np.array([pixel for pixel in filtered_brightPixels if np.sum(np.abs(pixel - centerPixelArray)) > pixelProximityThres])
        
        # The array you want to drop (remove)
        array_to_drop = filtered_brightPixels[0]

        brightPixelsTemp = np.array([pixel for pixel in filtered_brightPixels if not np.array_equal(pixel, array_to_drop)])

        iterStatus = True
        while iterStatus:
            noDropIndex = np.where(np.sum(np.abs(array_to_drop - brightPixelsTemp), axis = 1) > pixelProximityThres)[0]
            if (noDropIndex.shape[0] == brightPixelsTemp.shape[0]):
                brightPixelsTemp = np.vstack((brightPixelsTemp, array_to_drop))
                iterStatus = False
            else:
                brightPixelsTemp = brightPixelsTemp[noDropIndex]
                brightPixelsTemp = np.vstack((brightPixelsTemp, array_to_drop))
                array_to_drop = brightPixelsTemp[0]
                brightPixelsTemp = np.array([pixel for pixel in brightPixelsTemp if not np.array_equal(pixel, array_to_drop)])

        brightPixelsLeft = brightPixelsTemp
        brightPixelsLeft = brightPixelsLeft[np.argsort(brightPixelsLeft[:, 0])]

        centery, centerx = centerPixelArray[0], centerPixelArray[1]
        point1y, point1x = brightPixelsLeft[0][0], brightPixelsLeft[0][1]
        point2y, point2x = brightPixelsLeft[1][0], brightPixelsLeft[1][1]
        tangentRotationAngle = ((point1y - centery) / (point1x - centerx) + (centery - point2y) / (centerx - point2x))/2
        rotationAngle = np.degrees(np.arctan(tangentRotationAngle))
        return rotationAngle
        
        

    def goToNextFrame(self): 
        pass

class scalePixelLengthExtraction(tk.Frame):
    def __init__(self, parent, controller, *args, **kwargs):
        super().__init__(parent)
        self.controller = controller
        self.parent = parent

        label = tk.Label(self, text="Step: Extract Scale Ruler Pixel Length", font=("Arial", 25))
        label.grid(row = 0, column = 0, columnspan=2, padx=5, pady=5)

        self.isScalerBlack = tk.BooleanVar()
        self.label1 = tk.Label(self, text="Original Image")
        self.label1.grid(row=1, column=0, padx = 5, pady = 5)
        self.label2 = tk.Label(self, text="Modified Image")
        self.label2.grid(row=1, column=1, padx = 5, pady=5)

        image1 = self.parent.originalUploadImage
        originalImageRGB = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        self.originalImagePIL = Image.fromarray(originalImageRGB)
        self.parent.drawToPanel(self.originalImagePIL, self.label1)

        self.image2 = self.parent.originalUploadImage
        image2RGB = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        self.image2PIL = Image.fromarray(image2RGB)
        self.parent.drawToPanel(self.image2PIL, self.label2)

        caption1 = tk.Label(self, text="Original Image", font=font.Font(weight="bold"))
        caption1.grid(row = 2, column = 0, padx=5, pady=5)
        caption2 = tk.Label(self, text="Modified Image", font=font.Font(weight="bold"))
        caption2.grid(row = 2, column = 1, padx=5, pady=5)

        self.greyThresholdInputForRuler = tk.StringVar()  # Variable to store the slider value
        self.greyThresholdInputForRuler.set("1")  # Set initial value


        label = tk.Label(self, text="Adjust the grey threshold to emphasize the scale on the image. Observe the changes in the right image", font=("Arial", 12, "italic"))
        label.grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        # Label to display the slider value
        self.sliderValueLabelVar = tk.StringVar()
        self.sliderValueLabelVar.set(f'Set grey-scale threshold: {str(self.greyThresholdInputForRuler.get())} || {self.parent.longestWhiteRunLength / self.parent.imageResolution:.2f} pixels per nm!')
        self.sliderValueLabel = tk.Label(self, textvariable=self.sliderValueLabelVar)
        self.sliderValueLabel.grid(row=4, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        self.greyThresholdInputForRuler.trace_add("write", self.update_slider_label)

        # Create the checkbox
        self.scaler_include_black_checkbox = tk.Checkbutton(self, text="Does the scaler include black component?", variable = self.isScalerBlack)
        self.scaler_include_black_checkbox.grid(row=5, column=0, columnspan=1, padx=5, pady=5, sticky="ew")  # Adjust row and column as needed

        # Create a scale widget (slider)
        self.slider = ttk.Scale(self, from_ = 1, to = 254, orient="horizontal", command=self.handleSliderRuler)
        self.slider.grid(row=5, column=1, columnspan=1, padx=5, pady=5, sticky="ew")

        button = tk.Button(self, text="Continue to Next Step", command=self.goToNextFrame)
        button.grid(row = 6, column=0, columnspan=2, padx=5, pady=5)  # Adjust row and column as needed

    
    def update_slider_label(self, *args):
        self.sliderValueLabelVar.set(f"Set grey-scale threshold: {str(self.greyThresholdInputForRuler.get())} || {self.parent.longestWhiteRunLength / self.parent.imageResolution:.2f} pixels per nm")

    def handleSliderRuler(self, value):
        greyThresholdRuler = int(float(value))

        # Update the value of the label
        self.greyThresholdInputForRuler.set(str(greyThresholdRuler))
        thresholdedImageForRuler = TEMimageprocessing.nanoImagePreprocessing(self.image2, greyThresholdRuler, False)
        self.parent.longestWhiteRunLength, rulerImage = TEMimageprocessing.getPixelNumberForScale(thresholdedImageForRuler, self.isScalerBlack.get(), True, False)
        image2 = Image.fromarray(rulerImage)
        self.controller.drawToPanel(image2, self.label2)

    def goToNextFrame(self):
        self.parent.showFrame(extractTEMRegionsandColor)


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()
