def listen_for_wake_word(recognized_text, wake_word='jarvis'):
    """Return True if *wake_word* appears in *recognized_text*.

    The check is case-insensitive so "Hey Jarvis", "JARVIS!", etc. all match.

    Parameters
    ----------
    recognized_text : str
        Text transcribed from the user's speech.
    wake_word : str
        The trigger word to listen for (default ``'jarvis'``).

    Returns
    -------
    bool
        ``True`` when the wake word is detected, ``False`` otherwise.
    """
    if not recognized_text:
        return False
    return wake_word.lower() in recognized_text.lower()


def strip_wake_word(text, wake_word='jarvis'):
    """Remove the wake word (and surrounding whitespace) from *text*.

    This lets the assistant process only the command portion of a
    sentence such as "Jarvis, what time is it?" → "what time is it?".

    Parameters
    ----------
    text : str
        Full transcribed text that may contain the wake word.
    wake_word : str
        The trigger word to strip out.

    Returns
    -------
    str
        The text with the wake word removed and whitespace normalised.
    """
    import re
    pattern = re.compile(re.escape(wake_word), re.IGNORECASE)
    cleaned = pattern.sub('', text).strip().lstrip(',').strip()
    return cleaned
