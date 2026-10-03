import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "image_path",
    "full_text",
    "degradations",
}


# ==================================================
# Resolve image path from metadata
# ==================================================
def resolve_image_path(image_path_value: str, project_root: Path) -> Path:
    if not isinstance(image_path_value, str):
        raise TypeError("The 'image_path' field must be a string.")

    if not image_path_value.strip():
        raise ValueError("The 'image_path' field cannot be empty.")

    # metadata.jsonl uses Windows backslashes.
    # Replacing them makes the path usable on both
    # Windows and Linux.
    normalized_path = image_path_value.replace("\\", "/")

    image_path = Path(normalized_path)

    if not image_path.is_absolute():
        image_path = project_root / image_path

    image_path = image_path.resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Ground-truth image not found: {image_path}")

    if not image_path.is_file():
        raise ValueError(f"The image path is not a file: {image_path}")

    return image_path

# ==================================================
# Validate one ground-truth record
# ==================================================
def validate_ground_truth_record(record: dict[str, Any], line_number: int, project_root: Path) -> dict[str, Any]:
    missing_fields = REQUIRED_FIELDS - record.keys()

    if missing_fields:
        raise KeyError(f"Missing fields at JSONL line {line_number}: {missing_fields}")

    receipt_id = record["id"]
    full_text = record["full_text"]
    degradations = record["degradations"]

    if not isinstance(receipt_id, str):
        raise TypeError(f"The 'id' field at line {line_number} must be a string.")

    if not receipt_id.strip():
        raise ValueError(f"The 'id' field at line {line_number} cannot be empty.")

    if not isinstance(full_text, str):
        raise TypeError(f"The 'full_text' field at line {line_number} must be a string.")

    if not full_text.strip():
        raise ValueError(f"The 'full_text' field at line {line_number} cannot be empty.")

    if not isinstance(degradations, str):
        raise TypeError(f"The 'degradations' field at line {line_number} must be a string.")

    resolved_image_path = resolve_image_path(
        image_path_value=record["image_path"],
        project_root=project_root,
    )

    validated_record = record.copy()

    validated_record["id"] = receipt_id.strip()
    validated_record["full_text"] = full_text.strip()
    validated_record["resolved_image_path"] = (resolved_image_path)

    return validated_record

# ==================================================
# Load metadata.jsonl
# ==================================================
def load_ground_truth(metadata_path: str | Path, project_root: str | Path) -> list[dict[str, Any]]:
    metadata_path = Path(metadata_path)
    project_root = Path(project_root).resolve()

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not metadata_path.is_file():
        raise ValueError(f"The metadata path is not a file: {metadata_path}")

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    with metadata_path.open(mode="r",encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file,start=1):
            stripped_line = line.strip()

            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error

            if not isinstance(record, dict):
                raise TypeError(f"JSONL line {line_number} must contain a JSON object.")

            validated_record = validate_ground_truth_record(
                record=record,
                line_number=line_number,
                project_root=project_root,
            )

            receipt_id = validated_record["id"]

            if receipt_id in seen_ids:
                raise ValueError(f"Duplicate receipt ID at line {line_number}: {receipt_id}")

            seen_ids.add(receipt_id)
            records.append(validated_record)

    if not records:
        raise ValueError(f"No ground-truth records found in: {metadata_path}")

    return records


# ==================================================
# Main function for testing
# ==================================================
def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    metadata_path = (project_root/ "data"/ "raw_receipts"/ "synthetic_de"/ "metadata.jsonl")

    records = load_ground_truth(metadata_path=metadata_path,project_root=project_root)
    
    print(f"Loaded {len(records)} ground-truth records.")

    first_record = records[0]

    print("\nFirst record:")
    print(f"ID: {first_record['id']}")
    print(f"Image path: {first_record['resolved_image_path']}")
    print(f"Degradations: {first_record['degradations']}")

    print("\nGround-truth text:")
    print(first_record["full_text"])

    print("\nAll receipt IDs:")

    for index, record in enumerate(records, start=1):
        print(
            f"{index:02d}. "
            f"{record['id']} -> "
            f"{record['resolved_image_path'].name}"
        )


if __name__ == "__main__":
    main()