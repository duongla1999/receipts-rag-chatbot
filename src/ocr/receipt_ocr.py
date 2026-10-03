from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from paddleocr import PaddleOCR

from .ocr_result import OCRLine, OCRResult


# ==========================
# Receipt OCR service
# ==========================
class ReceiptOCR:
    def __init__(self) -> None:
        print(" ================== Initializing PaddleOCR... ================== ")

        self.ocr_engine = PaddleOCR(
            lang="de",
            ocr_version="PP-OCRv6",
            device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print(" ================== PaddleOCR initialized successfully. ================== ")
        print(f"PaddlePaddle version: {version('paddlepaddle')}")
        print(f"PaddleOCR version: {version('paddleocr')}")
        print(f"PaddleX version: {version('paddlex')}")
        print(f"==========================================================================")

    @staticmethod
    def _convert_to_list(value: Any) -> list:
        if hasattr(value, "tolist"):
            return value.tolist()

        return list(value)

    def _parse_lines(self, paddle_result: Any) -> list[OCRLine]:
        result_data = paddle_result.json

        if "res" not in result_data:
            raise KeyError("OCR result does not contain the 'res' field.")

        ocr_data = result_data["res"]

        required_fields = {
            "rec_texts",
            "rec_scores",
            "rec_boxes",
            "rec_polys",
        }

        missing_fields = required_fields - ocr_data.keys()

        if missing_fields:
            raise KeyError(f"Missing OCR fields: {missing_fields}")

        rec_texts = ocr_data["rec_texts"]
        rec_scores = ocr_data["rec_scores"]
        rec_boxes = ocr_data["rec_boxes"]
        rec_polys = ocr_data["rec_polys"]

        number_of_texts = len(rec_texts)

        if not (
            number_of_texts
            == len(rec_scores)
            == len(rec_boxes)
            == len(rec_polys)
        ):
            raise ValueError("OCR texts, scores, boxes and polygons must have the same length.")

        lines = []

        for index in range(number_of_texts):
            line = OCRLine(
                text=str(rec_texts[index]).strip(),
                confidence=float(rec_scores[index]),
                bbox=self._convert_to_list(rec_boxes[index]),
                polygon=self._convert_to_list(rec_polys[index])
            )

            lines.append(line)

        return lines

    def recognize(self, image: str | Path | np.ndarray, image_name: str, variant: str) -> OCRResult:
        supported_variants = {"original", "processed"}

        if variant not in supported_variants:
            raise ValueError(f"Unsupported image variant: {variant}. Supported variants: {supported_variants}")

        if isinstance(image, Path):
            if not image.exists():
                raise FileNotFoundError(f"Image file not found: {image}")
            
            if not image.is_file():
                raise ValueError(f"The provided path is not a file: {image}")
            
            image = str(image)
        
        if isinstance(image, np.ndarray) and image.size ==0:
            raise ValueError(f"Input image is empty.")

        print(f"Running OCR for {image_name} ({variant})...")

        start_time = perf_counter()

        paddle_results = list(self.ocr_engine.predict(input=image))

        runtime_seconds = perf_counter() - start_time

        if not paddle_results:
            raise RuntimeError(f"PaddleOCR returned no result for: {image_name}")

        if len(paddle_results) > 1:
            raise ValueError(f"ReceiptOCR.recognize() expects one image, but received {len(paddle_results)} results.")

        lines = self._parse_lines(paddle_results[0])

        return OCRResult(
            image_name=image_name,
            variant=variant,
            runtime_seconds=runtime_seconds,
            lines=lines,
        )


# ==========================
# Main function for testing
# ==========================
def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    image_path = (project_root/ "data"/ "raw_receipts"/ "synthetic"/ "train-000067.jpg")

    receipt_ocr = ReceiptOCR()

    result = receipt_ocr.recognize(
        image=image_path,
        image_name=image_path.name,
        variant="original",
    )

    print("\nOCR result:")
    print(result)

    print("\nRaw text:")
    print(result.raw_text)

    print("\nNumber of lines:")
    print(result.number_of_lines)

    print("\nMean confidence:")
    print(f"{result.mean_confidence:.4f}")

    if result.lines:
        print("\nFirst OCR line:")
        print(result.lines[0])

        print("\nFirst OCR line as dictionary:")
        print(result.lines[0].line_to_dict())
    else:
        print("\nNo OCR lines were recognized.")

    print("\n" + "=" * 100)
    print("All OCR lines:")

    for line_index, line in enumerate(result.lines, start=1):
        print(f"\nLine {line_index}:")
        print(line.line_to_dict())


if __name__ == "__main__":
    main()