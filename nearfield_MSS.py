"""
Nearfield Morphological Ellipse Discriminator.

Version: 2.1.0
Author: Michael C.M. Varney
Revision Notes: 
- Replaced the "failed" subfolder mechanism with a consolidated mss_summary.csv file.
- The summary CSV includes overall pass/fail status and individual metric evaluations in binary form.
- Enhanced code modularity and clarity for better maintainability.
Changelog:
- Removed failed subfolder creation.
- Added generation of mss_summary.csv with detailed pass/fail information.
- Updated versioning and revision history.
"""

import os
import math
import csv
import logging
from datetime import datetime
from tkinter import Tk, messagebox
from tkinter.filedialog import askopenfilenames, askopenfilename
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import matplotlib
from skimage.filters import threshold_yen
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration Parameters

# Image processing parameters
IMAGE_PATH = "temp_image.tiff"
APPLY_GAUSSIAN_BLUR = False  # Flag to apply Gaussian blur
GAUSSIAN_KERNEL_SIZE = (3, 3)  # Kernel size for Gaussian blur

# Thresholding parameters
THRESH_METHOD = "half_max"
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C = 2

# Morphological operations parameters
MORPH_ITERATIONS_CLOSE = 2
MORPH_ITERATIONS_DILATE = 2
KERNEL_SIZE = (2, 2)

# Ellipse fitting parameters
SCALE_FACTOR = 1
ELLIPSE_THICKNESS = 2  # Reduced thickness for better visualization

# Monte Carlo simulation parameters
EPOCH_SIZE = 50  # Number of samples in an epoch
MAX_NO_CHANGE_SAMPLES = 10  # Max consecutive epochs without significant score change
SCORE_CHANGE_THRESHOLD = 50  # Threshold for significant score change
MAX_SAMPLES = 1000  # Maximum number of samples to run through
SAMPLES_PER_EPOCH = 100  # Samples per epoch
INITIAL_PERTURBATION_RANGES = {
    "center_x": (-50, 50),
    "center_y": (-50, 50),
    "axes_major": (-200, 200),
    "axes_minor": (-200, 200),
    "angle": (0, 90),
}

# Area penalty parameters
AREA_WEIGHT = 0.1  # Weight for the area penalty in the combined score

# Output parameters
DEFAULT_FILE_TYPE = "png"  # Default file type for saving output images ('tiff', 'png', 'jpg')
DISPLAY_FIGURE = False  # Flag to control displaying the figure

# Machine Safety System (MSS) parameters
INCLUDE_MSS_ANALYSIS = True  # Flag to include MSS analysis
CRITERIA_CSV_PATH = ""  # To be set by user input if MSS is enabled

# Set matplotlib backend based on DISPLAY_FIGURE
if DISPLAY_FIGURE:
    matplotlib.use("TkAgg")  # Interactive backend
else:
    matplotlib.use("Agg")  # Non-interactive backend


def ellipse_circumference(a, b):
    """Calculate the circumference of an ellipse using Ramanujan's approximation."""
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def threshold_image(image, method, half_max_value):
    """Apply thresholding to the image based on the specified method."""
    if method == "otsu":
        _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive_mean":
        binary_mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C,
        )
    elif method == "adaptive_gaussian":
        binary_mask = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C,
        )
    elif method == "triangle":
        _, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    elif method == "tozero":
        _, binary_mask = cv2.threshold(image, half_max_value, 255, cv2.THRESH_TOZERO)
    elif method == "entropy":
        thresh_value = threshold_yen(image)
        _, binary_mask = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    elif method == "half_max":
        _, binary_mask = cv2.threshold(image, half_max_value, 255, cv2.THRESH_BINARY)
    else:
        _, binary_mask = cv2.threshold(image, half_max_value, 255, cv2.THRESH_BINARY)
    return binary_mask


def morphological_operations(binary_mask):
    """Perform morphological operations on the binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, KERNEL_SIZE)
    morph_binary = cv2.morphologyEx(
        binary_mask, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS_CLOSE
    )
    morph_binary = cv2.dilate(morph_binary, kernel, iterations=MORPH_ITERATIONS_DILATE)
    return morph_binary


def find_largest_contour(morph_binary):
    """Find the largest contour in the morphologically processed binary image."""
    contours, _ = cv2.findContours(morph_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour
    return None


def compute_score(params, best_params, morph_binary, mask_area, area_weight):
    """
    Compute the combined score for a given set of ellipse parameters.

    Args:
        params (tuple): Perturbation parameters (delta_center_x, delta_center_y, delta_axes_major, delta_axes_minor, delta_angle).
        best_params (dict): Current best parameters of the ellipse.
        morph_binary (numpy.ndarray): Morphologically processed binary mask.
        mask_area (int): Area of the morphological mask.
        area_weight (float): Weight for the area penalty.

    Returns:
        tuple: (combined_score, exclusion_score, area_penalty, new_ellipse, params)
    """
    (
        delta_center_x,
        delta_center_y,
        delta_axes_major,
        delta_axes_minor,
        delta_angle,
    ) = params
    new_center = (
        best_params["center_x"] + delta_center_x,
        best_params["center_y"] + delta_center_y,
    )
    new_axes = (
        max(best_params["axes_major"] + delta_axes_major, 1),
        max(best_params["axes_minor"] + delta_axes_minor, 1),
    )
    new_angle = best_params["angle"] + delta_angle
    new_ellipse = (new_center, new_axes, new_angle)
    ellipse_mask = np.zeros_like(morph_binary)
    cv2.ellipse(ellipse_mask, new_ellipse, 255, -1)

    # Calculate mask pixels outside the ellipse
    outside_pixels = cv2.bitwise_and(morph_binary, cv2.bitwise_not(ellipse_mask))
    exclusion_score = np.count_nonzero(outside_pixels == 255)

    # Compute area penalty
    semi_major = new_axes[0] / 2
    semi_minor = new_axes[1] / 2
    ellipse_area = math.pi * semi_major * semi_minor
    area_penalty = max(0, ellipse_area - mask_area)

    # Combine scores
    combined_score = exclusion_score + area_weight * area_penalty

    return (combined_score, exclusion_score, area_penalty, new_ellipse, params)


def calculate_statistics(image_array):
    """Calculate statistical metrics of the image."""
    mean_intensity = np.mean(image_array)
    median_intensity = np.median(image_array)
    intensity_variance = np.var(image_array)
    standard_deviation = np.std(image_array)
    histogram, _ = np.histogram(image_array, bins=256, range=(0, 255))
    histogram_data = {str(i): int(count) for i, count in enumerate(histogram)}

    return {
        "MeanIntensity": mean_intensity,
        "MedianIntensity": median_intensity,
        "IntensityVariance": intensity_variance,
        "StandardDeviation": standard_deviation,
        "HistogramData": histogram_data,
    }


def create_and_save_figure(
    overlay_binary,
    overlay_morph,
    ellipse_data,
    ellipticity,
    scaled_major_axis_length,
    scaled_minor_axis_length,
    scaled_angle,
    area,
    mask_area,
    area_ratio_morph,
    morph_perimeter,
    ellipse_perimeter,
    binary_area,
    binary_perimeter,
    results_dir,
    image_path,
):
    """Create and save the figure without GUI operations."""
    try:
        print("\nCreating figure...")

        # Create figure with specific DPI for better resolution
        plt.switch_backend("Agg")
        fig = plt.figure(figsize=(18, 12), dpi=100)
        gs = plt.GridSpec(2, 3, figure=fig)

        # Top row (3 panels)
        ax1 = fig.add_subplot(gs[0, 0])  # Original Binary
        ax2 = fig.add_subplot(gs[0, 1])  # Morphological Binary
        ax3 = fig.add_subplot(gs[0, 2])  # Monte Carlo Progress

        # Ensure proper image display by normalizing if needed
        if overlay_binary.dtype != np.uint8:
            overlay_binary = (overlay_binary * 255).astype(np.uint8)
        if overlay_morph.dtype != np.uint8:
            overlay_morph = (overlay_morph * 255).astype(np.uint8)

        # Display images with proper color conversion
        ax1.imshow(overlay_binary)
        ax1.set_title("Original Binary Image with Scaled Ellipse Overlay", pad=20)
        ax1.axis("off")

        ax2.imshow(overlay_morph)
        ax2.set_title("Morphological Binary Image with Scaled Ellipse Overlay", pad=20)
        ax2.axis("off")

        # Plot Monte Carlo progress
        combined_scores = [d["score"] for d in ellipse_data]
        exclusion_scores = [d["exclusion_score"] for d in ellipse_data]
        area_penalties = [d["area_penalty"] for d in ellipse_data]

        if combined_scores:
            ax3.plot(
                range(1, len(combined_scores) + 1),
                combined_scores,
                "r-",
                linewidth=1,
                marker="o",
                markersize=2,
                label="Combined Score",
            )
            ax3.plot(
                range(1, len(exclusion_scores) + 1),
                exclusion_scores,
                "b-",
                linewidth=1,
                marker="x",
                markersize=2,
                label="Exclusion Score",
            )
            ax3.plot(
                range(1, len(area_penalties) + 1),
                area_penalties,
                "g-",
                linewidth=1,
                marker="s",
                markersize=2,
                label="Area Penalty",
            )
            ax3.set_title("Monte Carlo Simulation Progress", pad=20)
            ax3.set_xlabel("Sample")
            ax3.set_ylabel("Score")
            ax3.legend()
            ax3.grid(True, linestyle="--", alpha=0.7)
        else:
            print("Warning: No score data available for plotting")

        # Bottom row content
        ax4 = fig.add_subplot(gs[1, 0])  # Original Binary Stats
        ax5 = fig.add_subplot(gs[1, 1])  # Morphological Stats
        ax6 = fig.add_subplot(gs[1, 2])  # Parameters

        # Original Binary Stats
        original_stats_text = (
            f"Ellipticity: {ellipticity:.2f}\n"
            f"Major Axis Length: {scaled_major_axis_length:.2f}\n"
            f"Minor Axis Length: {scaled_minor_axis_length:.2f}\n"
            f"Angle of Rotation: {scaled_angle:.2f} degrees\n"
            f"Area of Scaled Ellipse: {area:.2f}\n"
            f"Area of Original Binary Mask: {binary_area}\n"
            f"Ratio of Mask Area to Ellipse Area: {binary_area / area:.2f}\n"
            f"Perimeter of Ellipse: {ellipse_perimeter:.2f}\n"
            f"Perimeter of Original Binary Mask: {binary_perimeter:.2f}\n"
            f"Perimeter of Morphological Binary Mask: {morph_perimeter:.2f}\n"
            f"Perimeter Ratio (Ellipse/Original Mask): {ellipse_perimeter / binary_perimeter:.2f}"
        )
        ax4.text(
            0.5,
            0.5,
            original_stats_text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax4.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        ax4.set_title("Ellipse vs Original Binary Mask Statistics", pad=20)
        ax4.axis("off")

        # Morphological Stats
        morph_stats_text = (
            f"Ellipticity: {ellipticity:.2f}\n"
            f"Major Axis Length: {scaled_major_axis_length:.2f}\n"
            f"Minor Axis Length: {scaled_minor_axis_length:.2f}\n"
            f"Angle of Rotation: {scaled_angle:.2f} degrees\n"
            f"Area of Scaled Ellipse: {area:.2f}\n"
            f"Area of Morphological Binary Mask: {mask_area}\n"
            f"Ratio of Mask Area to Ellipse Area: {area_ratio_morph:.2f}\n"
            f"Perimeter of Ellipse: {ellipse_perimeter:.2f}\n"
            f"Perimeter of Morphological Binary Mask: {morph_perimeter:.2f}\n"
            f"Perimeter Ratio (Ellipse/Morph Mask): {ellipse_perimeter / morph_perimeter:.2f}"
        )
        ax5.text(
            0.5,
            0.5,
            morph_stats_text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax5.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        ax5.set_title("Ellipse vs Morphological Binary Mask Statistics", pad=20)
        ax5.axis("off")

        # Parameters
        params_text = (
            f"Thresholding Method: {THRESH_METHOD}\n"
            f"Threshold Value: Adaptive\n\n"
            f"Morphological Parameters:\n"
            f"  Kernel Size: {KERNEL_SIZE}\n"
            f"  Close Iterations: {MORPH_ITERATIONS_CLOSE}\n"
            f"  Dilate Iterations: {MORPH_ITERATIONS_DILATE}"
        )
        ax6.text(
            0.5,
            0.5,
            params_text,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax6.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.8),
        )
        ax6.set_title("Thresholding and Morphological Parameters", pad=20)
        ax6.axis("off")

        # Adjust layout and save
        plt.tight_layout(pad=3.0)
        save_path = os.path.join(
            results_dir, f"{os.path.basename(image_path)}_figure_1.png"
        )

        # Save with high quality
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Figure saved to: {save_path}")

        plt.close(fig)
        print("Figure creation completed")

        return save_path

    except Exception as e:
        print(f"Error in create_and_save_figure: {str(e)}")
        raise


def setup_logging(results_dir):
    """Setup logging configuration."""
    log_file = os.path.join(results_dir, "processing.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def save_stats_summary(data, results_dir, image_path):
    """Save statistics summary to CSV file."""
    base_name = os.path.basename(image_path)
    summary_csv_path = os.path.join(results_dir, f"{base_name}_stats_summary.csv")

    with open(summary_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Parameter", "Value"])

        def write_stat(parameter, value):
            if isinstance(value, float):
                value = f"{value:.3f}"
            writer.writerow([parameter, value])

        write_stat("Ellipticity", data["ellipticity"])
        write_stat("Major Axis Length", data["scaled_major_axis_length"])
        write_stat("Minor Axis Length", data["scaled_minor_axis_length"])
        write_stat("Angle of Rotation", data["scaled_angle"])
        write_stat("Area of Scaled Ellipse", data["area"])
        write_stat("Area of Original Binary Mask", data["binary_area"])
        write_stat(
            "Ratio of Mask Area to Ellipse Area",
            data["binary_area"] / data["area"] if data["area"] != 0 else 0,
        )
        write_stat("Perimeter of Ellipse", data["ellipse_perimeter"])
        write_stat("Perimeter of Original Binary Mask", data["binary_perimeter"])
        write_stat("Perimeter of Morphological Binary Mask", data["morph_perimeter"])
        write_stat(
            "Perimeter Ratio (Ellipse/Original Mask)",
            data["ellipse_perimeter"] / data["binary_perimeter"] if data["binary_perimeter"] != 0 else 0,
        )
        write_stat(
            "Perimeter Ratio (Ellipse/Morph Mask)",
            data["ellipse_perimeter"] / data["morph_perimeter"] if data["morph_perimeter"] != 0 else 0,
        )

    print(f"Stats summary saved to: {summary_csv_path}")


def evaluate_parameters(parameters, criteria):
    """
    Evaluate each parameter against the criteria.

    Args:
        parameters (dict): Dictionary of parameter names and their actual values.
        criteria (dict): Dictionary of parameter names and their (low, high) thresholds.

    Returns:
        dict: Dictionary with parameter evaluation results.
    """
    evaluation_results = {}
    for param, value in parameters.items():
        if param in criteria:
            low, high = criteria[param]
            if low <= value <= high:
                status = "Pass"
                binary = 1
            else:
                status = "Fail"
                binary = 0
            evaluation_results[param] = {
                "Value": value,
                "Value Low": low,
                "Value High": high,
                "Status": status,
                "Binary": binary,
            }
        else:
            evaluation_results[param] = {
                "Value": value,
                "Value Low": "N/A",
                "Value High": "N/A",
                "Status": "No Criteria",
                "Binary": "N/A",
            }
    return evaluation_results


def save_evaluation_results(evaluation_results, results_dir, image_path):
    """
    Save the evaluation results to a CSV file.

    Args:
        evaluation_results (dict): Dictionary with parameter evaluation results.
        results_dir (str): Directory to save the evaluation CSV.
        image_path (str): Path of the processed image.
    """
    base_name = os.path.basename(image_path)
    evaluation_csv_path = os.path.join(results_dir, f"{base_name}_evaluation.csv")

    with open(evaluation_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Parameter", "Value", "Value Low", "Value High", "Status", "Binary"])

        for param, result in evaluation_results.items():
            writer.writerow([
                param,
                result["Value"],
                result["Value Low"],
                result["Value High"],
                result["Status"],
                result["Binary"]
            ])

    print(f"Evaluation results saved to: {evaluation_csv_path}")


def load_mss_criteria(csv_path):
    """
    Load MSS pass/fail criteria from a CSV file.

    Args:
        csv_path (str): Path to the criteria CSV file.

    Returns:
        dict: Dictionary with parameter names as keys and (low, high) tuples as values.
    """
    criteria = {}
    try:
        with open(csv_path, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                param = row["Parameter"].strip()
                low = float(row["Value Low"].strip())
                high = float(row["Value High"].strip())
                criteria[param] = (low, high)
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading MSS criteria CSV: {str(e)}")
    return criteria


def generate_mss_summary(mss_summary_data, results_dir):
    """
    Generate the mss_summary.csv file in the top-level run directory.

    Args:
        mss_summary_data (list): List of dictionaries containing per-image evaluation data.
        results_dir (str): Directory to save the mss_summary.csv.
    """
    summary_csv_path = os.path.join(results_dir, "mss_summary.csv")

    # Determine all metric names from the first image's evaluation
    if not mss_summary_data:
        print("No data available to generate mss_summary.csv.")
        return

    first_evaluation = mss_summary_data[0]["evaluation"]
    metric_names = list(first_evaluation.keys())

    with open(summary_csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Header
        header = ["Image Name", "Pass/Fail", "Pass/Fail (Binary)"] + metric_names
        writer.writerow(header)

        for entry in mss_summary_data:
            image_name = os.path.basename(entry["image_path"])
            evaluations = entry["evaluation"]

            # Determine overall Pass/Fail
            overall_pass = all(
                detail["Status"] == "Pass" for detail in evaluations.values() if detail["Status"] != "No Criteria"
            )
            overall_status = "Pass" if overall_pass else "Fail"
            overall_binary = 1 if overall_pass else 0

            # Collect binary pass/fail for each metric
            metric_binaries = []
            for metric in metric_names:
                binary = evaluations[metric]["Binary"]
                metric_binaries.append(binary)

            # Write row
            writer.writerow([image_name, overall_status, overall_binary] + metric_binaries)

    print(f"MSS summary saved to: {summary_csv_path}")


def process_image(image_path, results_dir):
    """Process image without GUI operations."""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Started processing image: {image_path}")

        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Load image with validation
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            logger.error(f"Unable to load image at {image_path}")
            return None

        # Check if image is blank or invalid
        if image.size == 0 or np.all(image == 0) or np.all(image == 255):
            logger.warning(
                f"Warning: Image {image_path} appears to be blank or invalid"
            )
            return None

        # Calculate statistics
        statistics = calculate_statistics(image)

        # Print image statistics for debugging
        logger.info(f"\nProcessing image: {image_path}")
        logger.info(f"Image shape: {image.shape}")
        logger.info(f"Image intensity range: [{np.min(image)}, {np.max(image)}]")

        # Apply Gaussian blur if the flag is set
        if APPLY_GAUSSIAN_BLUR:
            image = cv2.GaussianBlur(image, GAUSSIAN_KERNEL_SIZE, 0)

        max_value = image.max()
        half_max_value = max_value / 2

        # Thresholding
        binary_mask = threshold_image(image, THRESH_METHOD, half_max_value)

        # Save original binary mask
        binary_mask_path = os.path.join(
            results_dir, f"{os.path.basename(image_path)}_original_binary_mask.png"
        )
        cv2.imwrite(binary_mask_path, binary_mask)
        logger.info(f"Saved binary mask: {binary_mask_path}")

        # Morphological operations
        morph_binary = morphological_operations(binary_mask)

        # Save morphological binary mask
        morph_mask_path = os.path.join(
            results_dir,
            f"{os.path.basename(image_path)}_morphological_binary_mask.png",
        )
        cv2.imwrite(morph_mask_path, morph_binary)
        logger.info(f"Saved morphological mask: {morph_mask_path}")

        # Find contours
        largest_contour = find_largest_contour(morph_binary)
        if largest_contour is None or len(largest_contour) < 5:
            logger.warning(
                "Not enough points in the largest contour to fit an ellipse."
            )
            return None

        # Initial ellipse fitting
        ellipse = cv2.fitEllipse(largest_contour)
        (center, axes, angle) = ellipse
        major_axis_length = max(axes)
        minor_axis_length = min(axes)

        scaled_major_axis = major_axis_length * SCALE_FACTOR
        scaled_minor_axis = minor_axis_length * SCALE_FACTOR
        scaled_axes = (scaled_major_axis, scaled_minor_axis)
        ellipse_scaled = (center, scaled_axes, angle)

        best_ellipse = ellipse_scaled
        best_params = {
            "center_x": center[0],
            "center_y": center[1],
            "axes_major": scaled_major_axis,
            "axes_minor": scaled_minor_axis,
            "angle": angle,
        }

        # Compute initial score
        mask_area = np.count_nonzero(morph_binary == 255)
        initial_params = (0, 0, 0, 0, 0)  # Zero perturbations
        initial_score, exclusion_score, area_penalty, _, _ = compute_score(
            initial_params, best_params, morph_binary, mask_area, AREA_WEIGHT
        )
        best_score = initial_score

        # Initialize data storage with initial guess
        ellipse_data = [
            {
                "center_x": best_params["center_x"],
                "center_y": best_params["center_y"],
                "axes_major": best_params["axes_major"],
                "axes_minor": best_params["axes_minor"],
                "angle": best_params["angle"],
                "score": best_score,  # Combined score
                "exclusion_score": exclusion_score,
                "area_penalty": area_penalty,
            }
        ]
        total_samples = 0  # Track total samples

        perturbation_ranges = INITIAL_PERTURBATION_RANGES.copy()
        no_change_samples_counter = 0  # Counter for epochs without significant score change

        # Adaptive epochs
        while total_samples < MAX_SAMPLES:
            # Generate perturbations for this epoch
            perturbations = []
            for _ in range(EPOCH_SIZE):
                delta_center_x = np.random.uniform(*perturbation_ranges["center_x"])
                delta_center_y = np.random.uniform(*perturbation_ranges["center_y"])
                delta_axes_major = np.random.uniform(*perturbation_ranges["axes_major"])
                delta_axes_minor = np.random.uniform(*perturbation_ranges["axes_minor"])
                delta_angle = np.random.uniform(*perturbation_ranges["angle"])
                perturbations.append(
                    (
                        delta_center_x,
                        delta_center_y,
                        delta_axes_major,
                        delta_axes_minor,
                        delta_angle,
                    )
                )

            # Compute scores in parallel
            args = (
                (params, best_params, morph_binary, mask_area, AREA_WEIGHT)
                for params in perturbations
            )
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                results = list(
                    tqdm(
                        executor.map(lambda p: compute_score(*p), args),
                        total=EPOCH_SIZE,
                        desc="Computing Scores",
                    )
                )

            # Find the best result in this epoch
            epoch_best_score = float("inf")
            epoch_best_ellipse = None
            epoch_best_params = None

            for result in results:
                combined_score, exclusion_score, area_penalty, new_ellipse, params = result
                new_params = {
                    "center_x": best_params["center_x"] + params[0],
                    "center_y": best_params["center_y"] + params[1],
                    "axes_major": best_params["axes_major"] + params[2],
                    "axes_minor": best_params["axes_minor"] + params[3],
                    "angle": best_params["angle"] + params[4],
                }

                # Store data for each sample
                ellipse_data.append(
                    {
                        "center_x": new_params["center_x"],
                        "center_y": new_params["center_y"],
                        "axes_major": new_params["axes_major"],
                        "axes_minor": new_params["axes_minor"],
                        "angle": new_params["angle"],
                        "score": combined_score,
                        "exclusion_score": exclusion_score,
                        "area_penalty": area_penalty,
                    }
                )

                total_samples += 1  # Increment total samples

                if combined_score < epoch_best_score:
                    epoch_best_score = combined_score
                    epoch_best_ellipse = new_ellipse
                    epoch_best_params = new_params

            # Calculate score improvement
            score_improvement = best_score - epoch_best_score

            # Update the global best if significant improvement
            if score_improvement >= SCORE_CHANGE_THRESHOLD:
                best_score = epoch_best_score
                best_ellipse = epoch_best_ellipse
                best_params = epoch_best_params
                no_change_samples_counter = 0  # Reset counter
            else:
                no_change_samples_counter += 1  # Increment counter

            # Check for early stopping condition
            if no_change_samples_counter >= MAX_NO_CHANGE_SAMPLES:
                logger.info(
                    f"Stopping early due to less than {SCORE_CHANGE_THRESHOLD} score change in {MAX_NO_CHANGE_SAMPLES} consecutive epochs."
                )
                break

            # Check if maximum samples reached
            if total_samples >= MAX_SAMPLES:
                logger.info(
                    f"Stopping early due to reaching the maximum number of samples: {MAX_SAMPLES}."
                )
                break

            # Narrow down the perturbation ranges around the best parameters
            for key in perturbation_ranges:
                range_min, range_max = perturbation_ranges[key]
                perturbation_ranges[key] = (range_min * 0.5, range_max * 0.5)

        # Original binary mask statistics
        binary_area = np.count_nonzero(binary_mask == 255)
        contours_binary, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_binary:
            largest_contour_binary = max(contours_binary, key=cv2.contourArea)
            binary_perimeter = cv2.arcLength(largest_contour_binary, True)
        else:
            binary_perimeter = 0  # Ensure binary_perimeter is defined
            logger.warning("Warning: No contours found in the original binary mask.")

        # Save score versus sample data to CSV
        csv_path = os.path.join(
            results_dir, f"{os.path.basename(image_path)}_score_vs_sample.csv"
        )
        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                "Sample",
                "Combined Score",
                "Exclusion Score",
                "Area Penalty",
                "Major Axis Length",
                "Minor Axis Length",
                "Angle",
                "Center X",
                "Center Y",
                "Ellipticity",
                "Area",
                "Mask Area",
                "Area Ratio Morph",
                "Perimeter Morph",
                "Perimeter Ellipse",
                "Perimeter Ratio Morph",
                "Perimeter Binary",
                "Perimeter Ratio Binary",
            ])
            for i, data in enumerate(ellipse_data):
                major_axis_length = data["axes_major"]
                minor_axis_length = data["axes_minor"]
                angle = data["angle"]
                center_x = data["center_x"]
                center_y = data["center_y"]
                ellipticity = minor_axis_length / major_axis_length if major_axis_length != 0 else 0
                semi_major = major_axis_length / 2
                semi_minor = minor_axis_length / 2
                area = math.pi * semi_major * semi_minor
                area_ratio_morph = mask_area / area if area != 0 else 0
                perimeter_morph = cv2.arcLength(largest_contour, True) if largest_contour is not None else 0
                perimeter_ellipse = ellipse_circumference(semi_major, semi_minor)
                perimeter_ratio_morph = perimeter_ellipse / perimeter_morph if perimeter_morph != 0 else 0
                perimeter_binary = binary_perimeter
                perimeter_ratio_binary = perimeter_ellipse / perimeter_binary if perimeter_binary != 0 else 0
                writer.writerow([
                    i + 1,
                    data["score"],
                    data["exclusion_score"],
                    data["area_penalty"],
                    major_axis_length,
                    minor_axis_length,
                    angle,
                    center_x,
                    center_y,
                    ellipticity,
                    area,
                    mask_area,
                    area_ratio_morph,
                    perimeter_morph,
                    perimeter_ellipse,
                    perimeter_ratio_morph,
                    perimeter_binary,
                    perimeter_ratio_binary,
                ])
        logger.info(f"Saved score data: {csv_path}")

        # Use the best ellipse found
        ellipse_scaled = best_ellipse
        (best_center, best_axes, best_angle) = ellipse_scaled
        scaled_major_axis_length = best_axes[0]
        scaled_minor_axis_length = best_axes[1]
        scaled_angle = best_angle

        semi_major = scaled_major_axis_length / 2
        semi_minor = scaled_minor_axis_length / 2
        ellipse_perimeter = ellipse_circumference(semi_major, semi_minor)

        # Visualization
        overlay_morph = cv2.cvtColor(morph_binary, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(overlay_morph, ellipse_scaled, (255, 0, 0), ELLIPSE_THICKNESS)

        overlay_binary = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
        cv2.ellipse(overlay_binary, ellipse_scaled, (255, 0, 0), ELLIPSE_THICKNESS)

        ellipticity = scaled_minor_axis_length / scaled_major_axis_length if scaled_major_axis_length != 0 else 0
        area = math.pi * semi_major * semi_minor
        mask_area = np.count_nonzero(morph_binary == 255)
        area_ratio_morph = mask_area / area if area != 0 else 0

        morph_perimeter = cv2.arcLength(largest_contour, True) if largest_contour is not None else 0

        # Original binary mask statistics
        binary_area = np.count_nonzero(binary_mask == 255)
        contours_binary, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours_binary:
            largest_contour_binary = max(contours_binary, key=cv2.contourArea)
            binary_perimeter = cv2.arcLength(largest_contour_binary, True)
        else:
            binary_perimeter = 0
            logger.warning("Warning: No contours found in the original binary mask.")

        # Check if the ellipse encloses all values of 255
        mask = np.zeros_like(morph_binary)
        cv2.ellipse(mask, ellipse_scaled, 255, -1)
        enclosed = cv2.bitwise_and(morph_binary, mask)
        if np.count_nonzero(enclosed == 255) != mask_area:
            logger.warning("Warning: The scaled ellipse does not enclose all values of 255")
        else:
            logger.info("The scaled ellipse encloses all values of 255 in the morphologically processed binary mask.")

        # Return the data needed for figure creation and evaluation
        return {
            "overlay_binary": overlay_binary,
            "overlay_morph": overlay_morph,
            "ellipse_data": ellipse_data,
            "ellipticity": ellipticity,
            "scaled_major_axis_length": scaled_major_axis_length,
            "scaled_minor_axis_length": scaled_minor_axis_length,
            "scaled_angle": scaled_angle,
            "area": area,
            "mask_area": mask_area,
            "area_ratio_morph": area_ratio_morph,
            "morph_perimeter": morph_perimeter,
            "ellipse_perimeter": ellipse_perimeter,
            "binary_area": binary_area,
            "binary_perimeter": binary_perimeter,
            "results_dir": results_dir,
            "image_path": image_path,
            "statistics": statistics,
        }

    except Exception as e:
        logger.error(f"Error processing {image_path}: {str(e)}")
        return None


def main():
    """Main function to execute the image processing workflow."""
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("analysis_results", f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir)
    logger.info(f"Starting new processing run in {results_dir}")

    # Select multiple input files
    Tk().withdraw()  # Hide the root window
    file_paths = askopenfilenames(
        title="Select Image Files",
        filetypes=[("TIFF files", "*.tiff"), ("All files", "*.*")],
    )

    if not file_paths:
        logger.info("No files selected.")
        return

    # If MSS is enabled, prompt user to select criteria CSV
    criteria = {}
    if INCLUDE_MSS_ANALYSIS:
        messagebox.showinfo("MSS Criteria Selection", "Please select the MSS criteria CSV file.")
        criteria_csv_path = askopenfilename(
            title="Select MSS Criteria CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not criteria_csv_path:
            logger.error("MSS analysis enabled but no criteria CSV file selected. Exiting.")
            return
        criteria = load_mss_criteria(criteria_csv_path)
        if not criteria:
            logger.error("Failed to load MSS criteria. Exiting.")
            return
        logger.info(f"Loaded MSS criteria from: {criteria_csv_path}")

    if len(file_paths) > 1 and DISPLAY_FIGURE:
        logger.warning(
            "\nWARNING: Display figure enabled - only the last image's figure will be displayed.\n"
        )

    # Collect data for figure creation and MSS summary
    figure_data_list = []
    evaluation_results_list = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            executor.submit(process_image, file_path, results_dir)
            for file_path in file_paths
        ]

        for future in tqdm(futures, desc="Processing Images"):
            data = future.result()
            if data is not None:
                figure_data_list.append(data)
            else:
                logger.warning("Processing returned no data.")

    # Now create and save figures and evaluate parameters in the main thread
    for data in figure_data_list:
        figure_path = create_and_save_figure(
            data["overlay_binary"],
            data["overlay_morph"],
            data["ellipse_data"],
            data["ellipticity"],
            data["scaled_major_axis_length"],
            data["scaled_minor_axis_length"],
            data["scaled_angle"],
            data["area"],
            data["mask_area"],
            data["area_ratio_morph"],
            data["morph_perimeter"],
            data["ellipse_perimeter"],
            data["binary_area"],
            data["binary_perimeter"],
            data["results_dir"],
            data["image_path"],
        )
        if figure_path:
            logger.info(f"Saved figure: {figure_path}")
        save_stats_summary(data, data["results_dir"], data["image_path"])

        # If MSS is enabled, evaluate parameters
        if INCLUDE_MSS_ANALYSIS:
            parameters = {
                "Ellipticity": data["ellipticity"],
                "Major Axis Length": data["scaled_major_axis_length"],
                "Minor Axis Length": data["scaled_minor_axis_length"],
                "Angle of Rotation": data["scaled_angle"],
                "Area of Scaled Ellipse": data["area"],
                "Area of Original Binary Mask": data["binary_area"],
                "Ratio of Mask Area to Ellipse Area": data["binary_area"] / data["area"] if data["area"] != 0 else 0,
                "Perimeter of Ellipse": data["ellipse_perimeter"],
                "Perimeter of Original Binary Mask": data["binary_perimeter"],
                "Perimeter of Morphological Binary Mask": data["morph_perimeter"],
                "Perimeter Ratio (Ellipse/Original Mask)": data["ellipse_perimeter"] / data["binary_perimeter"] if data["binary_perimeter"] != 0 else 0,
                "Perimeter Ratio (Ellipse/Morph Mask)": data["ellipse_perimeter"] / data["morph_perimeter"] if data["morph_perimeter"] != 0 else 0,
            }
            evaluation_results = evaluate_parameters(parameters, criteria)
            save_evaluation_results(evaluation_results, data["results_dir"], data["image_path"])
            evaluation_results_list.append({
                "image_path": data["image_path"],
                "evaluation": evaluation_results
            })

    # Generate mss_summary.csv
    if INCLUDE_MSS_ANALYSIS:
        generate_mss_summary(evaluation_results_list, results_dir)

    # Handle figure display in the main thread
    if DISPLAY_FIGURE and figure_data_list:
        matplotlib.use("TkAgg")
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        for data in figure_data_list:
            figure_path = create_and_save_figure(
                data["overlay_binary"],
                data["overlay_morph"],
                data["ellipse_data"],
                data["ellipticity"],
                data["scaled_major_axis_length"],
                data["scaled_minor_axis_length"],
                data["scaled_angle"],
                data["area"],
                data["mask_area"],
                data["area_ratio_morph"],
                data["morph_perimeter"],
                data["ellipse_perimeter"],
                data["binary_area"],
                data["binary_perimeter"],
                data["results_dir"],
                data["image_path"],
            )
            if figure_path:
                logger.info("Saved figure: %s", figure_path)
            save_stats_summary(data, data["results_dir"], data["image_path"])


if __name__ == "__main__":
    main()
