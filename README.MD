# Nearfield Morphological MSS

## Overview

The **Nearfield Morphological MSS** is a Python-based tool designed to analyze and discriminate elliptical features within near-field morphological images. It utilizes image processing techniques to enhance, threshold, and perform morphological operations, ultimately fitting ellipses to identified contours for further analysis.

## Features

- **Image Processing**: Applies Gaussian blur, adaptive thresholding, and morphological operations to prepare images for analysis.
- **Ellipse Fitting**: Identifies and fits ellipses to the largest contours in processed images.
- **Monte Carlo Simulation**: Utilizes perturbation methods to optimize ellipse fitting parameters.
- **Machine Safety System (MSS)**: Evaluates results against predefined safety criteria and generates summaries.

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/tardyon/Nearfield_Morph_MSS.git
   ```
2. **Navigate to the Project Directory**
   ```bash
   cd Nearfield_Morphological_Ellipse_Discriminator
   ```
3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Your Images**
   - Ensure your images are in a supported format (e.g., TIFF, PNG, JPG). Note Tiff is prefered

2. **Run the Program**
   ```bash
   python nearfield_MSS.py
   ```
   - A file dialog will prompt you to select one or more image files for processing.

3. **Review Results**
   - Processed images, statistical summaries, and evaluation results will be saved in the `analysis_results` directory.

## Configuration Parameters

- **Image Processing**
  - `APPLY_GAUSSIAN_BLUR`: Apply Gaussian blur to smooth images.
  - `GAUSSIAN_KERNEL_SIZE`: Kernel size for Gaussian blur.

- **Thresholding**
  - `THRESH_METHOD`: Method for image thresholding (e.g., Otsu, adaptive).
  - `ADAPTIVE_BLOCK_SIZE`: Block size for adaptive thresholding.
  - `ADAPTIVE_C`: Constant subtracted from the mean in adaptive thresholding.

- **Morphological Operations**
  - `MORPH_ITERATIONS_CLOSE`: Number of iterations for morphological closing.
  - `MORPH_ITERATIONS_DILATE`: Number of iterations for dilation.
  - `KERNEL_SIZE`: Structuring element size for morphological operations.

- **Ellipse Fitting**
  - `SCALE_FACTOR`: Scaling factor for ellipse dimensions.
  - `ELLIPSE_THICKNESS`: Thickness of the ellipse overlay in visualizations.

- **Monte Carlo Simulation**
  - `EPOCH_SIZE`: Number of samples per epoch.
  - `MAX_NO_CHANGE_SAMPLES`: Maximum epochs without significant score change before stopping.
  - `SCORE_CHANGE_THRESHOLD`: Threshold for considering a score change as significant.
  - `MAX_SAMPLES`: Maximum total samples to process.

- **Machine Safety System (MSS)**
  - `INCLUDE_MSS_ANALYSIS`: Enable or disable MSS analysis.
  - `CRITERIA_CSV_PATH`: Path to the CSV file containing MSS criteria.

## Processing Pipeline

1. **Image Loading and Validation**
2. **Optional Gaussian Blurring**
3. **Thresholding to Create Binary Mask**
4. **Morphological Operations (Closing and Dilation)**
5. **Contour Detection and Largest Contour Identification**
6. **Initial Ellipse Fitting**
7. **Monte Carlo Simulation for Parameter Optimization**
8. **Statistical Analysis and Summary Generation**
9. **Machine Safety System Evaluation (If Enabled)**
10. **Result Visualization and Saving**

## Machine Safety System (MSS)

The Machine Safety System (MSS) is an integral component that evaluates image analysis results against predefined safety criteria. 

### Input Format

The MSS expects a CSV file with the following structure:
```
Parameter Name, Value Low, Value High
Each row represents a criterion that the analysis results must meet.
```

### How MSS Functions

1. **Criteria Loading**: The MSS loads safety criteria from a specified CSV file. Each criterion includes a parameter name, acceptable range, and other relevant details.
2. **Evaluation**: After the image processing and ellipse fitting steps, the MSS evaluates the results against the loaded criteria. Each fitted ellipse and its properties are checked to ensure they fall within the acceptable ranges.

```python
function evaluate_parameter(parameter_name, actual_value, criteria):
    # Get the low and high threshold values for the parameter
    low_threshold = criteria[parameter_name]["Value Low"]
    high_threshold = criteria[parameter_name]["Value High"]

    # Check if the actual value is within the threshold range
    if low_threshold <= actual_value <= high_threshold:
        status = "Pass"
    else:
        status = "Fail"

    return status
```

3. **Summary Generation**: The MSS generates a summary report indicating whether the processed images meet the safety criteria. If any criteria are not met, the report highlights these discrepancies.
4. **Alerts and Notifications**: Depending on the configuration, the MSS can trigger alerts or notifications if the analysis results do not comply with the safety standards.

### MSS Configuration Parameters

- `INCLUDE_MSS_ANALYSIS`: Enable or disable MSS analysis.
- `CRITERIA_CSV_PATH`: Path to the CSV file containing MSS criteria.

The MSS ensures that the image analysis process not only identifies and fits ellipses accurately but also adheres to safety standards critical for the application's context.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Michael C.M. Varney
