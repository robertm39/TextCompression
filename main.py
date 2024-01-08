import lzma

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
OK_CHARS = ALPHABET + ALPHABET.upper() + "0123456789 "


def main_1():
    in_file = r"texts\wiki_chocolate_trimmed.txt"
    out_file = r"texts\wiki_chocolate_trimmed.lzma"

    with open(in_file, "r") as file:
        text = file.read()

    with lzma.open(out_file, "w") as file:
        file.write(text.encode(encoding="ascii"))


def main_2():
    in_file = r"texts\wiki_chocolate_trimmed.txt"
    out_file = r"texts\wiki_chocolate_trimmed.txt"
    lines = list[str]()
    with open(in_file, "r", encoding="utf8") as file:
        for line in file:
            out_line = "".join([c for c in line if c in OK_CHARS])  # type: ignore
            lines.append(out_line)

    with open(out_file, "w") as file:
        for line in lines:
            if line.strip():
                line = line.strip().lower()
                file.write(f"{line}\n")


def main():
    main_1()
    # main_2()


if __name__ == "__main__":
    main()
