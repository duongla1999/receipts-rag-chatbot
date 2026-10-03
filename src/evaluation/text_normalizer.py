import unicodedata


# ==================================================
# Normalize OCR text before evaluation
# ==================================================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    # Normalize different Unicode representations.
    normalized_text = unicodedata.normalize("NFKC", text)

    # Make the comparison case-insensitive.
    normalized_text = normalized_text.upper()

    # Treat multiplication symbols equally.
    normalized_text = normalized_text.replace("×", "X")

    # Collapse spaces, tabs and line breaks into
    # a single space.
    normalized_text = " ".join(normalized_text.split())

    return normalized_text


# ==================================================
# Main function for testing
# ==================================================
def main() -> None:
    sample_text = (
        "  Summe     41,44 EUR  \n"
        "\n"
        "  2 × 5,99 eur  "
    )

    normalized_text = normalize_text(sample_text)

    print("Original text:")
    print(repr(sample_text))

    print("\nNormalized text:")
    print(repr(normalized_text))


if __name__ == "__main__":
    main()