"""Four whole-dollar digits predicted in parallel.

Marketplace prices in this pilot are non-negative and at most $1100.
Scientific notation left most slots constant. These four slots cover $0 to
$9999: thousands, hundreds, tens, and ones. Values are rounded half-up to the
nearest dollar before encoding.
"""

from decimal import Decimal, ROUND_HALF_UP, localcontext
import math

SLOT_SIZES = (10, 10, 10, 10)
SLOT_NAMES = ("thousands", "hundreds", "tens", "ones")
MAX_DOLLARS = 9999


def encode(value: float) -> tuple[int, ...]:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("The target must be a finite non-negative price")
    with localcontext() as context:
        context.prec = 32
        dollars = int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if dollars > MAX_DOLLARS:
        raise ValueError("Rounded price exceeds four dollar digits")
    return (dollars // 1000, (dollars // 100) % 10, (dollars // 10) % 10, dollars % 10)


def validate(slots) -> tuple[int, ...]:
    slots = tuple(slots)
    if len(slots) != len(SLOT_SIZES):
        raise ValueError("Expected four dollar-digit choices")
    for choice, size in zip(slots, SLOT_SIZES):
        if not isinstance(choice, int) or isinstance(choice, bool) or not 0 <= choice < size:
            raise ValueError("Invalid categorical slot choice")
    return slots


def format_slots(slots) -> str:
    return "".join(str(slot) for slot in validate(slots))


def decode(slots) -> float:
    thousands, hundreds, tens, ones = validate(slots)
    return float(1000 * thousands + 100 * hundreds + 10 * tens + ones)


def quantization_error(value: float) -> float:
    return abs(decode(encode(value)) - value)
