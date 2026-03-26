def format_timestamp(seconds: float) -> str:
    """Format *seconds* as ``H:MM:SS`` or ``MM:SS`` (no milliseconds)."""
    total = int(seconds)
    h, remainder = divmod(total, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"
