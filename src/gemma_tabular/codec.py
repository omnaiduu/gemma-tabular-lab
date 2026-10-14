"""Canonical four-significant-digit scientific notation with eight choices.

The first mantissa digit includes zero, giving 64 logits across the eight heads.
Zero is decoded canonically even if the other predicted slots disagree.
"""

from decimal import Decimal, localcontext
import math

SLOT_SIZES = (2, 10, 10, 10, 10, 2, 10, 10)
SLOT_NAMES = ("sign", "digit_1", "digit_2", "digit_3", "digit_4", "exponent_sign", "exponent_1", "exponent_2")


def encode(value: float) -> tuple[int, ...]:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("The target must be finite")
    if value == 0:
        return (0,) * 8
    with localcontext() as context:
        context.prec = 32
        text = format(Decimal(str(value)), "+.3e")
    mantissa, exponent_text = text.split("e")
    exponent = int(exponent_text)
    if abs(exponent) > 99:
        raise ValueError("Rounded target exceeds the two-digit exponent range")
    digits = mantissa[1:].replace(".", "")
    return (int(mantissa[0] == "-"), *(int(d) for d in digits),
            int(exponent < 0), abs(exponent) // 10, abs(exponent) % 10)


def validate(slots) -> tuple[int, ...]:
    slots = tuple(slots)
    if len(slots) != 8:
        raise ValueError("Expected eight slot choices")
    for choice, size in zip(slots, SLOT_SIZES):
        if not isinstance(choice, int) or isinstance(choice, bool) or not 0 <= choice < size:
            raise ValueError("Invalid categorical slot choice")
    return slots


def format_slots(slots) -> str:
    slots = validate(slots)
    if slots[1] == 0:
        return "+0.000e+00"
    sign = "-" if slots[0] else "+"
    exponent_sign = "-" if slots[5] else "+"
    return f"{sign}{slots[1]}.{slots[2]}{slots[3]}{slots[4]}e{exponent_sign}{slots[6]}{slots[7]}"


def decode(slots) -> float:
    return float(format_slots(slots))


def quantization_error(value: float) -> float:
    return abs(decode(encode(value)) - value)
