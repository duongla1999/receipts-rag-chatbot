from pydantic import ValidationError

from schemas import ReceiptSchema

def validate_receipt(receipt: ReceiptSchema, tolerance: float = 0.05) -> list[str]:
    errors = []

    items_sum = sum(item.total_price for item in receipt.items)
    if receipt.subtotal is not None:
        diff = abs(items_sum - receipt.subtotal)
        if diff > tolerance:
            errors.append(
                f"Items sum: ({items_sum:.2f}) big diff with subtotal"
                f"({receipt.subtotal:.2f}), diff: {diff:.2f}"
            )

    if receipt.total < 0:
        errors.append(f"Total < 0: {receipt.total}")
    if not receipt.items:
        errors.append("No item found!")

    for i, item in enumerate(receipt.items):
        if item.total_price < 0:
            errors.append(f"Item: #{i} ({item.raw_name}) has negative price {item.total_price}")

    return errors

def parse_and_validate(raw: dict) -> tuple[ReceiptSchema | None, list[str]]:
    try:
        receipt = ReceiptSchema.model_validate(raw)
    except ValidationError as e:
        return None, [f"Wrong format: {err['loc']} - {err['msg']}" for err in e.errors()]
    return receipt, validate_receipt(receipt)