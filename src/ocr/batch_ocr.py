from pathlib import Path
from typing import Any
import numpy as np

from .receipt_ocr import ReceiptOCR
from ..evaluation.ground_truth_loader import (load_ground_truth,)
from ..evaluation.ocr_metrics import (calculate_ocr_metrics,)
from ..image_processing.preprocess import load_image, preprocess_image


# ==================================================
# Evaluate original receipt images
# ==================================================
def evaluate_original_receipts(records: list[dict[str, Any]], ocr_engine: ReceiptOCR) -> tuple[list[dict[str, Any]],list[dict[str, str]],]:
    evaluation_results: list[dict[str, Any]] = []
    failed_results: list[dict[str, str]] = []

    total_records = len(records)

    for record_index, record in enumerate(records,start=1,):
        receipt_id = record["id"]
        image_path = record["resolved_image_path"]

        print(f"\n[{record_index}/{total_records}] Processing {receipt_id}...")

        try:
            ocr_result = ocr_engine.recognize(
                image=image_path,
                image_name=image_path.name,
                variant="original",
            )

            metrics = calculate_ocr_metrics(
                reference_text=record["full_text"],
                predicted_text=ocr_result.raw_text,
            )

            evaluation_record = {
                "receipt_id": receipt_id,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "variant": ocr_result.variant,
                "degradations": record["degradations"],
                "runtime_seconds": (ocr_result.runtime_seconds),
                "number_of_lines": (ocr_result.number_of_lines),
                "mean_confidence": (ocr_result.mean_confidence),
                "raw_cer": metrics["raw_cer"],
                "raw_wer": metrics["raw_wer"],
                "normalized_cer": (metrics["normalized_cer"]),
                "normalized_wer": (metrics["normalized_wer"]),
                "ground_truth_text": (record["full_text"]),
                "predicted_text": (ocr_result.raw_text),
                "lines": [
                    line.line_to_dict()
                    for line in ocr_result.lines
                ],
            }

            evaluation_results.append(
                evaluation_record
            )

            print(
                f"Completed: "
                f"CER={metrics['normalized_cer']:.4f}, "
                f"WER={metrics['normalized_wer']:.4f}, "
                f"confidence="
                f"{ocr_result.mean_confidence:.4f}, "
                f"runtime="
                f"{ocr_result.runtime_seconds:.2f}s"
            )

        except Exception as error:
            failed_record = {
                "receipt_id": receipt_id,
                "image_name": image_path.name,
                "variant": "original",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            failed_results.append(failed_record)

            print(f"Failed: {receipt_id} — {type(error).__name__}: {error}")

    return evaluation_results, failed_results


# ==================================================
# Print evaluation summary
# ==================================================
def print_evaluation_summary(evaluation_results: list[dict[str, Any]],failed_results: list[dict[str, str]],) -> None:
    
    print("\n" + "=" * 70)
    print("ORIGINAL OCR EVALUATION SUMMARY")
    print("=" * 70)

    successful_count = len(evaluation_results)
    failed_count = len(failed_results)

    print(f"Successful receipts: {successful_count}")
    print(f"Failed receipts: {failed_count}")

    if not evaluation_results:
        print("No successful OCR result available.")
        return

    mean_normalized_cer = sum(
        result["normalized_cer"]
        for result in evaluation_results
    ) / successful_count

    mean_normalized_wer = sum(
        result["normalized_wer"]
        for result in evaluation_results
    ) / successful_count

    mean_confidence = sum(
        result["mean_confidence"]
        for result in evaluation_results
    ) / successful_count

    mean_runtime = sum(
        result["runtime_seconds"]
        for result in evaluation_results
    ) / successful_count

    print(f"Mean normalized CER: {mean_normalized_cer:.4f}")
    print(f"Mean normalized WER: {mean_normalized_wer:.4f}")
    print(f"Mean OCR confidence: {mean_confidence:.4f}")
    print(f"Mean runtime: {mean_runtime:.4f} seconds")

    if failed_results:
        print("\nFailed receipt IDs:")

        for failed_record in failed_results:
            print(
                f"- {failed_record['receipt_id']}: "
                f"{failed_record['error_type']} — "
                f"{failed_record['error_message']}"
            )

# ==================================================
# Evaluate processed receipt images
# ==================================================
def evaluate_processed_receipts(
    records: list[dict[str, Any]],
    ocr_engine: ReceiptOCR,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:

    evaluation_results = []
    failed_results = []

    total_records = len(records)

    for index, record in enumerate(records, start=1):
        receipt_id = record["id"]
        image_path = record["resolved_image_path"]

        print(f"\n[{index}/{total_records}] Processing {receipt_id}...")

        try:
            # 1. Đọc ảnh gốc
            original_image = load_image(image_path)

            # 2. Chạy toàn bộ pipeline preprocessing
            preprocessing_results = preprocess_image(original_image)

            # 3. Lấy ảnh cuối cùng sau deskew
            processed_image = preprocessing_results["deskewed"]

            if not isinstance(processed_image, np.ndarray):
                raise TypeError("The preprocessing result 'deskewed' is not an image.")

            # 4. Chạy OCR trực tiếp trên NumPy array
            ocr_result = ocr_engine.recognize(
                image=processed_image,
                image_name=image_path.name,
                variant="processed",
            )

            # 5. So sánh OCR text với ground truth
            metrics = calculate_ocr_metrics(reference_text=record["full_text"],predicted_text=ocr_result.raw_text,)

            evaluation_record = {
                "receipt_id": receipt_id,
                "image_name": image_path.name,
                "image_path": str(image_path),
                "variant": "processed",
                "degradations": record["degradations"],
                "runtime_seconds": (
                    ocr_result.runtime_seconds
                ),
                "number_of_lines": (
                    ocr_result.number_of_lines
                ),
                "mean_confidence": (
                    ocr_result.mean_confidence
                ),
                "raw_cer": metrics["raw_cer"],
                "raw_wer": metrics["raw_wer"],
                "normalized_cer": (
                    metrics["normalized_cer"]
                ),
                "normalized_wer": (
                    metrics["normalized_wer"]
                ),
                "ground_truth_text": record["full_text"],
                "predicted_text": ocr_result.raw_text,
                "lines": [
                    line.line_to_dict()
                    for line in ocr_result.lines
                ],
            }

            evaluation_results.append(evaluation_record)

            print(
                "Completed: "
                f"CER={metrics['normalized_cer']:.4f}, "
                f"WER={metrics['normalized_wer']:.4f}, "
                f"confidence="
                f"{ocr_result.mean_confidence:.4f}, "
                f"runtime="
                f"{ocr_result.runtime_seconds:.2f}s"
            )

        except Exception as error:
            failed_record = {
                "receipt_id": receipt_id,
                "image_name": image_path.name,
                "variant": "processed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

            failed_results.append(failed_record)

            print(f"Failed: {receipt_id} — {type(error).__name__}: {error}")

    return evaluation_results, failed_results


# ==================================================
# Main function for testing
# ==================================================
def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    metadata_path = (
        project_root
        / "data"
        / "raw_receipts"
        / "synthetic_de"
        / "metadata.jsonl"
    )

    records = load_ground_truth(
        metadata_path=metadata_path,
        project_root=project_root,
    )

    print(f"Loaded {len(records)} ground-truth records.")

    ocr_engine = ReceiptOCR()

    processed_results, processed_failures = (evaluate_processed_receipts(records=records, ocr_engine=ocr_engine))

    print_evaluation_summary(
        evaluation_results=processed_results,
        failed_results=processed_failures,
        variant="processed",
    )


if __name__ == "__main__":
    main()
