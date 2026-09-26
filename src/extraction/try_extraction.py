from receipt_extractor import extract_receipt
from validator import validate_receipt

fake_raw_text = """REWE Markt GmbH
Hannover
------------------------------
H-MILCH 3,5%          1,29 A
BANANEN   1,10 kg x 1,79   1,97 A
VOLLKORNBROT           2,49 A
------------------------------
SUMME EUR              7,04
MWST A  7%              0,46
Kartenzahlung
19.09.2026  18:22"""

receipt = extract_receipt(fake_raw_text)

if receipt is None:
    print("Extraction that bai.")
else:
    print(receipt.model_dump_json(indent=2))
    errors = validate_receipt(receipt)
    if errors:
        print("\nLOI NGHIEP VU:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nHop le, san sang insert vao DB.")