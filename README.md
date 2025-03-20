# TEM Image Strain Analysis Tool - PyNanoSpacing

---

## Introduction  
This software tool, written entirely in Python 3.11, enables the analysis of interplanar spacing distortions in transmission electron microscopy (TEM) images. It extracts atomic layer distances, calculates strain, and visualizes the results, making it easier to study and understand the material properties at the nanoscale.

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
3. The ruler in the image is detected and used to extract the scale.
4. Manually select the region of interest (ROI) by clicking on the desired atomic region.
5. Enter the reference interplanar spacing value for accurate calculations.
6. The program calculates atomic distances, applies strain analysis, and visualizes the results.
7. Processed images and calculated strain values can be saved for further analysis.


## Strain Calculation Method
The tool compares detected atomic distances to a known reference value.
The strain percentage is computed as:

Strain = 
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
