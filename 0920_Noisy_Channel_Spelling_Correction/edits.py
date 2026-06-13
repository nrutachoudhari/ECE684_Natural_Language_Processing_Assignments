def get_edits(original: str, characters: list[str]) -> list[tuple[str, str]]:
    edits = []

    # generate deletions
    for idx, char in enumerate(original):
        previous_char = original[idx - 1] if idx > 0 else "#"
        edits.append((f"d:{previous_char}{char}", original[:idx] + original[idx + 1 :]))

    # generate substitutions
    for idx, old_char in enumerate(original):
        for new_char in characters:
            edits.append(
                (
                    f"s:{old_char}{new_char}",
                    original[:idx] + new_char + original[idx + 1 :],
                )
            )

    # generate additions
    for idx, char in enumerate("#" + original):
        for new_char in characters:
            edits.append(
                (
                    f"a:{char}{new_char}",
                    original[:idx] + new_char + original[idx:],
                )
            )

    return edits


if __name__ == "__main__":
    edits = get_edits("hello", ["a", "b", "c"])
    print(edits)
