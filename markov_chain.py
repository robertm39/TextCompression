import os
import json

from consts import *

PrefixInfo = dict[str, dict[str, int]]

# # Information about the frequency of characters after certain prefixes.
# class PrefixInfo:
#     def __init__(self, prefix_info: Mapping[str, Mapping[str, int]]):
#         self.prefix_info = frozendict(
#             {prefix: frozendict(freqs) for prefix, freqs in prefix_info.items()}
#         )

#     # Return the frequencies after the given prefix.
#     def get_frequencies(self, prefix: str) -> Mapping[str, int] | None:
#         if not prefix in self.prefix_info:
#             return None
#         return self.prefix_info[prefix]


# Get the prefix-frequency info for the given file.
def get_prefix_info(filename: str) -> PrefixInfo:
    # Read the file.
    with open(filename, "r", encoding="utf8") as file:
        text = file.read()

    # Remove unacceptable characters and clean up whitespace.
    text = "".join([c for c in text if c in OK_CHARS])
    text = text.replace("\n", " ").lower()
    while "  " in text:
        text = text.replace("  ", " ")

    # Gather the prefix information.
    prefix_len = 5
    result = PrefixInfo()
    current_prefix = ""
    for c in text:
        # Check that the prefix is the correct length.
        if len(current_prefix) == prefix_len:
            # Take note that this character occured after this prefix.
            if not current_prefix in result:
                result[current_prefix] = dict[str, int]()
            freqs = result[current_prefix]
            if not c in freqs:
                freqs[c] = 0
            freqs[c] += 1

        # Update the prefix.
        current_prefix = current_prefix + c
        while len(current_prefix) > prefix_len:
            current_prefix = current_prefix[1:]

    return result


# Add the second prefix-info to the first.
def add_prefix_info(
    p1: PrefixInfo,
    p2: PrefixInfo,
) -> None:
    for prefix, freqs in p2.items():
        if not prefix in p1:
            p1[prefix] = dict[str, int]()
        p1_freqs = p1[prefix]
        for c, freq in freqs.items():
            if not c in p1_freqs:
                p1_freqs[c] = 0
            p1_freqs[c] += freq


def make_prefix_info():
    files_dir = r"Corpora\OANC_GrAF\OANC_Text_Files"
    out_filepath = r"Prefix_Info\prefix_info.json"
    prefix_info = PrefixInfo()
    for filename in os.listdir(files_dir):
        filepath = os.path.join(files_dir, filename)
        file_prefix_info = get_prefix_info(filename=filepath)
        add_prefix_info(prefix_info, file_prefix_info)

    with open(out_filepath, "w") as out_file:
        json.dump(prefix_info, out_file)

def markov_make_text():
    in_file = r""

def main():
    make_prefix_info()


if __name__ == "__main__":
    main()
