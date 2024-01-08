import os
import lzma
import shutil

from consts import *

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

def extract_txts_from_anc():
    corpus_dir = r"Corpora\OANC_GrAF\OANC-GrAF\data"
    out_dir = r"Corpora\OANC_GrAF\OANC_Text_Files"
    i = 0
    for root, _, files in os.walk(corpus_dir):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext != ".txt":
                continue
            # print(os.path.join(root, file))
            out_name = f"{i}_{file}"
            # out_name = os.path.join(root, file).replace(r"Corpora\OANC_GrAF\OANC-GrAF\data", "")
            # out_name = out_name[1:]
            # out_name = out_name.replace("-", "_").replace("\\", "-")
            # print(out_name)
            shutil.copy(src=os.path.join(root, file), dst=os.path.join(out_dir, out_name))
            
            i += 1



def main():
    # main_1()
    # main_2()
    extract_txts_from_anc()


if __name__ == "__main__":
    main()
