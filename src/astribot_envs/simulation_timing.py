"""Small wall-clock scheduling helpers for simulation loops."""


def advance_periodic_deadline(deadline_ns: int, now_ns: int, period_ns: int) -> int:
    """Advance one period, resynchronizing to now after a missed deadline."""
    if period_ns <= 0:
        raise ValueError("period_ns must be positive")
    next_deadline_ns = deadline_ns + period_ns
    if next_deadline_ns <= now_ns:
        next_deadline_ns = now_ns
    return next_deadline_ns
