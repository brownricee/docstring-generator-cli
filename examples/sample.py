"""A small partially-documented module, for trying the CLI end to end.

Two of these functions have docstrings and two do not -- which is the case the
tool is built for: a real repo with a handful of gaps, not an undocumented one.
"""


def slugify(text, separator="-"):
    """Convert text into a lowercase URL-safe slug.

    Args:
        text: The string to convert.
        separator: Character placed between words.

    Returns:
        str: The slugified string.
    """
    words = [w for w in text.lower().split() if w.isalnum()]
    return separator.join(words)


def word_count(text):
    counts = {}
    for word in text.split():
        counts[word] = counts.get(word, 0) + 1
    return counts


class Accumulator:
    """Running total that can be reset."""

    def __init__(self):
        self.total = 0

    def add(self, value):
        self.total += value
        return self.total
