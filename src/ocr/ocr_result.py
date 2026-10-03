# ==================================================
# One recognized text region
# ==================================================

class OCRLine:
    def __init__(self, text: str, confidence: float, bbox: list[int], polygon: list[list[int]]):
        self.text = text
        self.confidence = confidence
        self.bbox = bbox
        self.polygon = polygon
    
    def line_to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "polygon": self.polygon
        }
        
    def __repr__(self) -> str:
        return (
            "OCRLine("
            f"text={self.text!r}, "
            f"confidence={self.confidence!r}, "
            f"bbox={self.bbox!r}, "
            f"polygon={self.polygon!r}"
            ")"
        )
        
# ==================================================
# Complete OCR result (include objects of OCRLine)
# ==================================================

class OCRResult:
    def __init__(self, image_name: str, variant: str, runtime_seconds: float, lines: list[OCRLine]):
        self.image_name = image_name
        self.variant = variant
        self.runtime_seconds = runtime_seconds
        self.lines = lines
        
    @property
    def raw_text(self) -> str:
        return "\n".join(
            line.text
            for line in self.lines
            if line.text
        )

    @property
    def mean_confidence(self) -> float:
        if not self.lines:
            return 0.0

        total_confidence = sum(line.confidence for line in self.lines)
        mean_confidence = total_confidence / len(self.lines)
        
        return mean_confidence

    @property
    def number_of_lines(self) -> int:
        return len(self.lines)

    def result_to_dict(self) -> dict:
        return {
            "image_name": self.image_name,
            "variant": self.variant,
            "runtime_seconds": self.runtime_seconds,
            "number_of_lines": self.number_of_lines,
            "mean_confidence": self.mean_confidence,
            "raw_text": self.raw_text,
            "lines": [
                line.line_to_dict()
                for line in self.lines
            ],
        }

    def __repr__(self) -> str:
        return (
            "OCRResult("
            f"image_name={self.image_name!r}, "
            f"variant={self.variant!r}, "
            f"runtime_seconds={self.runtime_seconds:.4f}, "
            f"number_of_lines={self.number_of_lines}"
            ")"
        )