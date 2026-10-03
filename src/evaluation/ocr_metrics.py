from jiwer import cer, wer

from .text_normalizer import normalize_text


# ==================================================
# Calculate OCR evaluation metrics
# ==================================================
def calculate_ocr_metrics(reference_text: str, predicted_text: str) -> dict[str, float]:
    
    if not isinstance(reference_text, str):
        raise TypeError("Reference text must be a string.")

    if not isinstance(predicted_text, str):
        raise TypeError("Predicted text must be a string.")

    raw_reference = reference_text.strip()
    raw_prediction = predicted_text.strip()

    if not raw_reference:
        raise ValueError("Reference text cannot be empty.")

    normalized_reference = normalize_text(raw_reference)
    normalized_prediction = normalize_text(raw_prediction)

    metrics = {
        "raw_cer": float(cer(raw_reference, raw_prediction)),
        "raw_wer": float(wer(raw_reference, raw_prediction)),
        
        "normalized_cer": float(cer(normalized_reference, normalized_prediction)),
        "normalized_wer": float(wer(normalized_reference, normalized_prediction,))
    }

    return metrics


# ==================================================
# Main function for testing
# ==================================================
def main() -> None:
    reference_text = (
        "SUMME 41,44 EUR\n"
        "2 × 5,99 EUR"
    )

    predicted_text = (
        "Summe   41,44 EUR\n\n"
        "2 x 5,99 eur"
    )

    metrics = calculate_ocr_metrics(reference_text=reference_text, predicted_text=predicted_text)

    print("Reference text:")
    print(reference_text)

    print("\nPredicted text:")
    print(predicted_text)

    print("\nNormalized reference:")
    print(normalize_text(reference_text))

    print("\nNormalized prediction:")
    print(normalize_text(predicted_text))

    print("\nOCR metrics:")

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()