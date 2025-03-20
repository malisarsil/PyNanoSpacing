# TEM Image Strain Analysis Tool - PyNanoSpacing

---

## Introduction  
This software tool, developed in Python 3.11, facilitates the transformation of Transmission Electron Microscopy (TEM) images into color-mapped strain indicator images. By utilizing advanced image processing algorithms and artificial intelligence (AI) modules, the tool allows users to effortlessly generate the strain map by simply setting up the environment and executing the command `python app.py`. This will launch a user-friendly interface, guiding the user through the process.

The core functionality of the tool includes the extraction of atomic layer distances, strain calculation, and the visualization of the results. This approach significantly enhances the study of material properties at the nanoscale, providing valuable insights into the structural characteristics of the material under examination.

## Features  
- **OCR-based Scale Detection:** Automatically extracts the scale from the image ruler.  
- **Region of Interest (ROI) Selection:** Users can select specific atomic regions for analysis.  
- **Interplanar Spacing Calculation:** Computes atomic plane distances and compares them to a reference value.  
- **Strain Visualization:** Uses a red-green color map (red = expansion, green = compression).  
- **Export Options:** Allows saving results in both image and Excel formats.  

## Setup and Installation  
1. Ensure you have `conda` installed on your system.
2. Create a Conda Environment:

```bash
conda create --name tem-strain python=3.11.5
```

2. To install the required dependencies, run:  

```bash
pip install -r requirements.txt
```

## Usage  
3. To run the application, follow these steps:

1. **Run the application:**
   
```bash
   python app.py
```

2. Upload a TEM image for processing.
3. The ruler in the image is automatically detected and used to extract the scale. In cases where the ruler length is not detected accurately, the user has the option to manually input the value (e.g., entering "2" for a 2 nm resolution).
4. Enter the reference interplanar spacing value to ensure accurate calculations. This reference value serves as the basis for determining the compression and expansion rates, which are critical for strain analysis.
5. Manually select the region of interest (ROI) by clicking on the desired atomic area. This selected region is then processed by the "Segment Anything" model to identify the atomic region boundaries.
6. The program calculates atomic distances, performs strain analysis, and visualizes the results through color-mapping. Users can save the resulting image, as well as the histogram image that depicts strain level percentages and their corresponding counts. Additionally, the strain percentages and counts can be exported to an Excel file. 
7. Processed images and calculated strain values can be saved for subsequent analysis.


## Strain Calculation Method
The tool compares detected atomic distances to a known reference value. The strain percentage is computed using the following formula:

**Strain** = \(\frac{d_p - d_s}{d_s} \times 100\)

Where:
- \(d_p\) represents the strained interplanar spacing.
- \(d_s\) is the unstrained interplanar spacing of the bulk (the reference).

  
A color-mapped strain visualization is generated, where:
Red represents expansion (positive strain).
Green represents compression (negative strain).

## Results
The tool provides:

✅ Processed TEM images with strain mapping.

✅ Numerical results exported to an Excel file.

✅ An interactive workflow with real-time strain visualization.

## Documentation & Video Guide
📹 An instructional video guide is included in the repository for reference. You can find the documentation and video guide here (https://github.com/malisarsil/PyNanoSpacing/blob/main/instruction_video.mp4).
