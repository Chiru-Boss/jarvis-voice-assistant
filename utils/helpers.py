import re


def strip_punctuation_edges(text):
    """Remove leading/trailing punctuation from *text*."""
    return text.strip().strip('.,!?;:')


def truncate(text, max_chars=500):
    """Return *text* truncated to *max_chars* characters."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + '…'
