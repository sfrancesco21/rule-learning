from __future__ import annotations


def parse_int_spec(spec: str) -> list[int]:
    values: list[int] = []
    for chunk in spec.split(","):
        item = chunk.strip()
        if not item:
            continue
        if "-" in item:
            start_str, end_str = item.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(item))
    if not values:
        raise ValueError("Expected at least one integer in the specification.")
    return values


def parse_csv_spec(spec: str) -> list[str]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one value in the specification.")
    return values
