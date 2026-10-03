from pathlib import Path

import cv2
import numpy as np

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

#==========================
# Load image function
#==========================
def load_image(image_path: str | Path) -> np.ndarray:
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    if not image_path.is_file():
        raise ValueError(f"The provided path is not a file: {image_path}")
    
    if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {image_path.suffix}")
    
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"OpenCV could not read the image: {image_path}")
    
    return image

#============================
# Resize small imagesfor OCR
#============================

def resize_if_needed(image: np.ndarray, min_width: int=760) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError(f"Input image is empty.")
    if min_width <= 0:
        raise ValueError("Minimum width must be greater than zero.")
    
    height, width = image.shape[:2]
    
    if width >= min_width:
        return image.copy()
    
    scale_factor = min_width / width
    new_height = round(height * scale_factor)
    
    resized_image = cv2.resize(image, (min_width, new_height), interpolation=cv2.INTER_CUBIC)

    return resized_image
#==========================
#
#==========================
def convert_to_grayscale(image):
    if image is None or image.size == 0:
        raise ValueError(f"Input image is empty.")
    
    if image.ndim == 2:
        return image.copy()
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #OpenCV read image in BGR not RGB
    
    return grayscaled_image

#==========================
# Reduce Noise Image
#==========================
def reduce_noise(image: np.ndarray, strength: float=5.0) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    if image.ndim != 2:
        raise ValueError("Noise reduction requires a grayscale image.")
    if strength <= 0:
        raise ValueError("Denoising strength must be greater than zero.")
    
    denoised_image = cv2.fastNlMeansDenoising(image, None, h=strength, templateWindowSize=7, searchWindowSize=21)
    
    return denoised_image

#==========================
#
#==========================
def clahe_enhance_contrast(image, clip_limit, tile_grid_size: tuple[int, int]=(8,8)):
    if image is None or image.size == 0:
        raise ValueError("Contrast enhancement requires a grayscale image.")
    if image.ndim != 2:
        raise ValueError("CLAHE requires a grayscale image (dimensions = 2)")
    if clip_limit <= 0:
        raise ValueError("Clip limit must be greater than zero.")
    if (len(tile_grid_size) != 2 or tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0):
        raise ValueError("Tile grid size must contain two positive integers.")
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)    
    
    enhanced_image = clahe.apply(image)
    
    return enhanced_image

# ===============================
# Apply adaptive threshold
# ===============================
def apply_adaptive_threshold(image, block_size=31, constant=15.0):
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim != 2:
        raise ValueError("Adaptive thresholding requires a grayscale image.")
    if image.dtype != np.uint8:
        raise ValueError("Adative thresholding requires an unit8 image.")
    if block_size <= 1:
        raise ValueError("Block size must be grater than 1.")
    if block_size % 2 == 0:
        raise ValueError("Block size must be an odd number.")
    
    thresholded_image = cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=block_size,
        C=constant
    )
    
    return thresholded_image


# ==========================
# Apply morphological processing
# ==========================
def apply_morphology(image: np.ndarray, operation: str="closing", kernel_size: tuple[int, int]=(2, 2), iterations: int=1,) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if image.ndim != 2:
        raise ValueError("Morphological processing requires a single-channel image.")
    if (len(kernel_size) != 2 or kernel_size[0] <= 0 or kernel_size[1] <= 0):
        raise ValueError("Kernel size must contain two positive integers.")
    if iterations <= 0:
        raise ValueError("Iterations must be greater than zero.")

    supported_operations = {"opening", "closing"}

    if operation not in supported_operations:
        raise ValueError(f"Unsupported operation: {operation}. Supported operations: {supported_operations}")

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,kernel_size,)

    operation_type = (cv2.MORPH_OPEN
        if operation == "opening"
        else cv2.MORPH_CLOSE
    )

    # Threshold image contains black text on a white background.
    # OpenCV morphology treats white pixels as foreground,
    # so invert the image before applying morphology.
    inverted_image = cv2.bitwise_not(image)

    processed_inverted_image = cv2.morphologyEx(
        inverted_image,
        operation_type,
        kernel,
        iterations=iterations,
    )

    processed_image = cv2.bitwise_not(processed_inverted_image)

    return processed_image

# ==========================
# Estimate skew angle
# ==========================
def estimate_skew_angle(
    binary_image: np.ndarray,
) -> float:
    
    if binary_image is None or binary_image.size == 0:
        raise ValueError("Input image is empty.")
    if binary_image.ndim != 2:
        raise ValueError("Skew estimation requires a single-channel binary image.")

    # Black text becomes white foreground.
    inverted_image = cv2.bitwise_not(binary_image)

    coordinates = cv2.findNonZero(inverted_image)

    if coordinates is None or len(coordinates) < 10:
        return 0.0

    rectangle = cv2.minAreaRect(coordinates)
    angle = rectangle[-1]

    # Normalize OpenCV's angle to approximately [-45, 45].
    if angle > 45.0:
        angle -= 90.0

    return float(angle)

# ==========================
# Rotate image
# ==========================
def rotate_image(
    image: np.ndarray,
    angle: float,
    border_value: int | tuple[int, int, int] = 255,
) -> np.ndarray:
    
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")

    height, width = image.shape[:2]

    center = (width / 2.0, height / 2.0,)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated_image

# ==========================
# Deskew image
# ==========================
def deskew_image(
    image: np.ndarray,
    binary_reference: np.ndarray,
    minimum_angle: float = 0.1,
    maximum_angle: float = 15.0,
) -> tuple[np.ndarray, float]:
    
    if image is None or image.size == 0:
        raise ValueError("Input image is empty.")
    if binary_reference is None or binary_reference.size == 0:
        raise ValueError("Binary reference image is empty.")
    if image.shape[:2] != binary_reference.shape[:2]:
        raise ValueError("Image and binary reference must have the same size.")
    if minimum_angle < 0:
        raise ValueError("Minimum angle must be greater than or equal to zero.")
    if maximum_angle <= 0:
        raise ValueError("Maximum angle must be greater than zero.")

    skew_angle = estimate_skew_angle(binary_reference)

    if abs(skew_angle) < minimum_angle:
        return image.copy(), 0.0

    if abs(skew_angle) > maximum_angle:
        return image.copy(), 0.0

    deskewed_image = rotate_image(image, angle=skew_angle, border_value=255)

    return deskewed_image, skew_angle


# ==========================
# Save processed image
# ==========================
def save_processed_image(image: np.ndarray, output_path: str | Path,) -> Path:
    
    if image is None or image.size == 0:
        raise ValueError("Cannot save an empty image.")

    output_path = Path(output_path)

    supported_output_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    }

    if output_path.suffix.lower() not in supported_output_extensions:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

    output_path.parent.mkdir(parents=True,exist_ok=True)

    save_successful = cv2.imwrite(str(output_path),image)

    if not save_successful:
        raise IOError(f"OpenCV could not save the image: {output_path}")

    return output_path

# ==========================
# Complete preprocessing pipeline
# ==========================
def preprocess_image(
    image: np.ndarray,
    min_width: int = 720,
    denoising_strength: float = 7.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
    threshold_block_size: int = 31,
    threshold_constant: float = 15.0,
    morphology_operation: str = "closing",
    morphology_kernel_size: tuple[int, int] = (2, 2),
) -> dict[str, np.ndarray | float]:
    
    resized_image = resize_if_needed(image, min_width=min_width)
    grayscale_image = convert_to_grayscale(resized_image)
    denoised_image = reduce_noise(grayscale_image, strength=denoising_strength)
    enhanced_image = clahe_enhance_contrast(denoised_image, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
    thresholded_image = apply_adaptive_threshold(enhanced_image, block_size=threshold_block_size, constant=threshold_constant)
    morphology_image = apply_morphology(thresholded_image, operation=morphology_operation, kernel_size=morphology_kernel_size,iterations=1)
    deskewed_image, skew_angle = deskew_image(morphology_image,binary_reference=morphology_image,)

    return {
        "original": image.copy(),
        "resized": resized_image,
        "grayscale": grayscale_image,
        "denoised": denoised_image,
        "enhanced": enhanced_image,
        "thresholded": thresholded_image,
        "morphology": morphology_image,
        "deskewed": deskewed_image,
        "skew_angle": skew_angle,
    }

# ==========================
# Display preprocessing stages
# ==========================
def show_preprocessing_results(results: dict[str, np.ndarray | float],) -> None:
    image_stages = [
        ("1 - Original", "original"),
        ("2 - Resized", "resized"),
        ("3 - Grayscale", "grayscale"),
        ("4 - Denoised", "denoised"),
        ("5 - CLAHE enhanced", "enhanced"),
        ("6 - Adaptive threshold", "thresholded"),
        ("7 - Morphology", "morphology"),
        ("8 - Deskewed", "deskewed"),
    ]

    for window_title, result_key in image_stages:
        image = results[result_key]

        if isinstance(image, np.ndarray):
            cv2.imshow(window_title, image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


#==========================
# Main function to check
#==========================
def main():
    project_root = Path(__file__).resolve().parents[2]
        
    image_path = (project_root/"data/raw_receipts/synthetic/train-000067.jpg")
    
    original_image = load_image(image_path)
    
    output_path = (project_root/ "data"/ "processed_images"/ f"{image_path.stem}_processed.png")

    original_image = load_image(image_path)

    results = preprocess_image(original_image)

    for stage_name, result in results.items():
        if isinstance(result, np.ndarray):
            print(f"{stage_name}: shape={result.shape}, dtype={result.dtype}")
        else:
            print(f"{stage_name}: {result:.2f}")

    processed_image = results["deskewed"]

    if not isinstance(processed_image, np.ndarray):
        raise TypeError("The processed result is not an image.")

    saved_path = save_processed_image(processed_image,output_path,)

    print(f"Processed image saved to: {saved_path}")

    show_preprocessing_results(results)

if __name__ == "__main__":
    main()