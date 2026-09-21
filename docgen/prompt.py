def build_prompt(code: str) -> str:
    """Turn docstring-stripped code into the exact text fed to the model.

    Training and the CLI must build this string the same way, character for
    character. If the two differ, the model meets a format at inference that
    it never saw in training. Defined once here; training/train.py and
    docgen/generator.py both import it.
    """
    return (
        "Write a Google-style docstring for the following Python function. "
        "Respond with only the docstring text -- no code, no quotes.\n\n"
        f"{code}\n\nDocstring:\n"
    )
