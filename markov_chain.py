from typing import cast, Mapping
from abc import ABC, abstractmethod
import os
import json
import random

from bitlist import bitlist
import torch

from consts import *
import models
import text_nn_utils

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
def get_prefix_info_from_text(filename: str) -> PrefixInfo:
    # Read the file.
    with open(filename, "r", encoding="utf8") as file:
        text = file.read()

    # Remove unacceptable characters and clean up whitespace.
    text = "".join([c for c in text if c in OK_ALPHA_CHARS])
    text = text.replace("\n", " ").lower()
    while "  " in text:
        text = text.replace("  ", " ")

    # Gather the prefix information.
    prefix_len = 4
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
        file_prefix_info = get_prefix_info_from_text(filename=filepath)
        add_prefix_info(prefix_info, file_prefix_info)

    with open(out_filepath, "w") as out_file:
        json.dump(prefix_info, out_file)


# Return a random choice with the given frequencies.
def choose_from_freqs(freqs: dict[str, int]) -> str:
    total_freq = sum(freqs.values())
    num = random.randint(a=1, b=total_freq)
    for c, freq in freqs.items():
        if num <= freq:
            return c
        num -= freq
    return " "


# Make a random text using the frequencies extracted.
def markov_make_text():
    in_file = r"Prefix_Info\prefix_info_oanc_len4_lowercase.json"
    with open(in_file) as file:
        prefix_info = json.load(file)

    prefix = "and "
    print(prefix)
    for _ in range(10):
        for _ in range(80):
            freqs = prefix_info.get(prefix, None)
            if freqs is None:
                next_char = random.choice(OUT_CHARS)
            else:
                next_char = choose_from_freqs(freqs=freqs)
            print(next_char, end="")
            prefix = prefix[1:] + next_char
        print("")


def get_chars_sorted_by_frequency_descending(freqs: Mapping[str, int]) -> list[str]:
    chars = list(freqs)
    chars.sort(key=lambda c: freqs.get(c, 0), reverse=True)
    return chars


def get_prefix_info() -> dict[str, dict[str, int]]:
    in_file = r"Prefix_Info\prefix_info_oanc_len4_lowercase.json"
    with open(in_file) as file:
        prefix_info = json.load(file)
    return prefix_info


# Return the given text, cleaned up.
def clean_up_text(text: str) -> str:
    text = text.replace("\n", " ").replace("\t", " ").lower()
    text = "".join([c for c in text if c in OUT_CHARS])
    while "  " in text:
        text = text.replace("  ", " ")
    return text


# Predicts the next character in a text.
class CharPredictor(ABC):
    # Return the possible next characters in decreasing order of likeliness.
    @abstractmethod
    def predict_char(self, text: str) -> list[str] | None:
        ...


# A letter-predictor using fixed-length prefixes.
class FixedLengthPredixMarkovPredictor(CharPredictor):
    def __init__(self, prefix_info: PrefixInfo, prefix_len: int) -> None:
        self.prefix_info = prefix_info
        self.prefix_len = prefix_len

    def predict_char(self, text: str) -> list[str] | None:
        # If the text isn't long enough, we have no prediction.
        if len(text) < self.prefix_len:
            return None

        # Get the prefix.
        prefix = text[-self.prefix_len :]
        freqs = self.prefix_info.get(prefix, None)
        if freqs is None:
            return None

        sorted_chars = get_chars_sorted_by_frequency_descending(freqs=freqs)
        return sorted_chars


PADDING_HEADER_LENGTH = 3


# Compress the given text using a markov chain.
def markov_chain_compress(text: str, prefix_len=4) -> bytes:
    prefix_info = get_prefix_info()

    # Clean up the text.
    text = text.replace("\n", " ").replace("\t", " ").lower()
    text = "".join([c for c in text if c in OUT_CHARS])
    while "  " in text:
        text = text.replace("  ", " ")

    bits = bitlist(length=0)
    current_prefix = ""
    for c in text:
        # If the character isn't one of the ones we use, ignore it.
        if not c in OUT_CHARS:
            continue

        # Get the frequencies for the current prefix.
        if len(current_prefix) == prefix_len:
            freqs = prefix_info.get(current_prefix, None)
        else:
            freqs = None

        # If we have the frequencies, see if they match the current letter.
        found_in_freqs = False
        if freqs is not None:
            chars_in_order = get_chars_sorted_by_frequency_descending(freqs=freqs)
            if c == chars_in_order[0]:
                bits += bitlist("1")
                found_in_freqs = True
            elif c == chars_in_order[1]:
                bits += bitlist("01")
                found_in_freqs = True

        # If we couldn't use the frequencies, specify the character.
        if not found_in_freqs:
            char_num = OUT_CHARS.find(c)
            if char_num == -1:  # Should never happen.
                continue
            letter_bits = bitlist(char_num, 5)
            bits += bitlist("00")
            bits += letter_bits

        # Update the current prefix.
        current_prefix = current_prefix + c
        while len(current_prefix) > prefix_len:
            current_prefix = current_prefix[1:]

    # Pad the bits so they make up a whole number of bytes.
    num_bits = len(bits) + PADDING_HEADER_LENGTH
    padding_needed = 8 - (num_bits % 8)
    padding = bitlist(0, length=padding_needed)
    padding_header = bitlist(padding_needed, length=PADDING_HEADER_LENGTH)
    bits = padding_header + bits + padding

    return bits.to_bytes()


def general_compress(text: str, char_predictor: CharPredictor, prefix_len=4) -> bytes:
    # Clean up the text.
    text = text.replace("\n", " ").replace("\t", " ").lower()
    text = "".join([c for c in text if c in OUT_CHARS])
    while "  " in text:
        text = text.replace("  ", " ")

    bits = bitlist(length=0)
    current_prefix = ""
    for c in text:
        # If the character isn't one of the ones we use, ignore it.
        if not c in OUT_CHARS:
            continue

        predicted_chars = char_predictor.predict_char(text=current_prefix)
        found_in_freqs = False
        if predicted_chars is not None:
            if c == predicted_chars[0]:
                bits += bitlist("1")
                found_in_freqs = True
            elif c == predicted_chars[1]:
                bits += bitlist("01")
                found_in_freqs = True

        # If we couldn't use the frequencies, specify the character.
        if not found_in_freqs:
            char_num = OUT_CHARS.find(c)
            if char_num == -1:  # Should never happen.
                continue
            letter_bits = bitlist(char_num, 5)
            bits += bitlist("00")
            bits += letter_bits

        # Update the current prefix.
        current_prefix = current_prefix + c
        while len(current_prefix) > prefix_len:
            current_prefix = current_prefix[1:]

    # Pad the bits so they make up a whole number of bytes.
    num_bits = len(bits) + PADDING_HEADER_LENGTH
    padding_needed = 8 - (num_bits % 8)
    padding = bitlist(0, length=padding_needed)
    padding_header = bitlist(padding_needed, length=PADDING_HEADER_LENGTH)
    bits = padding_header + bits + padding

    return bits.to_bytes()


# Decompress the given bytes using a markov chain.
def markov_chain_decompress(msg: bytes, prefix_len=4) -> str | None:
    prefix_info = get_prefix_info()

    # Remove the padding.
    bits = bitlist(msg)
    padding_header, bits = bits[:PADDING_HEADER_LENGTH], bits[PADDING_HEADER_LENGTH:]
    padding_len = int(padding_header)
    bits = bits[:-padding_len]  # type: ignore
    bits = cast(bitlist, bits)

    current_prefix = ""
    result = list[str]()
    b_iter = iter(bits)
    try:
        while True:
            # Read one character.
            b = next(b_iter)
            b = cast(int, b)

            # Get the frequency information, if any.
            if len(current_prefix) == prefix_len:
                freqs = prefix_info.get(current_prefix, None)
            else:
                freqs = None

            # The next character is the most frequenct character.
            next_char = None
            if b == 1:
                # print("First char")
                if freqs is None:
                    # We don't have the frequency information!
                    print("No frequency!")
                    return None
                chars_in_order = get_chars_sorted_by_frequency_descending(freqs=freqs)
                next_char = chars_in_order[0]
            elif b == 0:
                b2 = next(b_iter)
                if b2 == 1:
                    # print("Second char")
                    if freqs is None:
                        print("No frequency!")
                        # We don't have the frequency information!
                        return None
                    chars_in_order = get_chars_sorted_by_frequency_descending(
                        freqs=freqs
                    )
                    if len(chars_in_order) == 1:
                        # There is no second-most-frequent character!
                        print("No second-most-frequent character!")
                        return None
                    next_char = chars_in_order[1]
                elif b2 == 0:
                    # Read the character from the next five bits.
                    bs = (
                        next(b_iter),
                        next(b_iter),
                        next(b_iter),
                        next(b_iter),
                        next(b_iter),
                    )
                    bs = cast(tuple[int, ...], bs)
                    bs = bitlist(bs)
                    char_index = int(bs)
                    try:
                        next_char = OUT_CHARS[char_index]
                    except IndexError:
                        print(f"Index error. i={char_index}")
                        return None

            if next_char is None:
                print("No character found.")
                return None

            # Add this character to the result.
            result.append(next_char)

            # Update the prefix.
            current_prefix = current_prefix + next_char
            while len(current_prefix) > prefix_len:
                current_prefix = current_prefix[1:]
    except StopIteration:
        pass
    return "".join(result)


def general_decompress(
    msg: bytes, char_predictor: CharPredictor, prefix_len=4
) -> str | None:
    # prefix_info = get_prefix_info()

    # Remove the padding.
    bits = bitlist(msg)
    padding_header, bits = bits[:PADDING_HEADER_LENGTH], bits[PADDING_HEADER_LENGTH:]
    padding_len = int(padding_header)
    bits = bits[:-padding_len]  # type: ignore
    bits = cast(bitlist, bits)

    current_prefix = ""
    result = list[str]()
    b_iter = iter(bits)
    try:
        while True:
            # Read one character.
            b = next(b_iter)
            b = cast(int, b)

            # # Get the frequency information, if any.
            # if len(current_prefix) == prefix_len:
            #     freqs = prefix_info.get(current_prefix, None)
            # else:
            #     freqs = None

            # Get the predicted next chars.
            predicted_chars = char_predictor.predict_char(text=current_prefix)

            # The next character is the most frequenct character.
            next_char = None
            if b == 1:
                # print("First char")
                # if freqs is None:
                #     # We don't have the frequency information!
                #     print("No frequency!")
                #     return None
                # chars_in_order = get_chars_sorted_by_frequency_descending(freqs=freqs)
                if predicted_chars is None:
                    return None
                if len(predicted_chars) <= 0:
                    return None
                next_char = predicted_chars[0]
            elif b == 0:
                b2 = next(b_iter)
                if b2 == 1:
                    # print("Second char")
                    # if freqs is None:
                    #     print("No frequency!")
                    #     # We don't have the frequency information!
                    #     return None
                    # chars_in_order = get_chars_sorted_by_frequency_descending(
                    #     freqs=freqs
                    # )

                    if predicted_chars is None:
                        return None
                    if len(predicted_chars) <= 1:
                        # There is no second-most-frequent character!
                        print("No second-most-frequent character!")
                        return None
                    next_char = predicted_chars[1]
                elif b2 == 0:
                    # Read the character from the next five bits.
                    bs = (
                        next(b_iter),
                        next(b_iter),
                        next(b_iter),
                        next(b_iter),
                        next(b_iter),
                    )
                    bs = cast(tuple[int, ...], bs)
                    bs = bitlist(bs)
                    char_index = int(bs)
                    try:
                        next_char = OUT_CHARS[char_index]
                    except IndexError:
                        print(f"Index error. i={char_index}")
                        return None

            if next_char is None:
                print("No character found.")
                return None

            # Add this character to the result.
            result.append(next_char)

            # Update the prefix.
            current_prefix = current_prefix + next_char
            while len(current_prefix) > prefix_len:
                current_prefix = current_prefix[1:]
    except StopIteration:
        pass
    return "".join(result)


TEXT = (
    "Polands war-torn and almost incomprehensibly fractured history plays "
    "out like an epic novel — occasionally triumphant, frequently sad and tragic."
    " Over a millennium, Poland evolved from a huge and imposing, economically po"
    "werful kingdom to a partitioned nation that ceased to exist on world maps fo"
    "r over 120 years, and finally to a people and land at the center of the 20th"
    " centurys greatest wars and most horrific human tragedies. But Poland has su"
    "rvived, with its culture, language and most of its territory intact, and tod"
    "ay Poles look forward with optimism to taking their place at the forefront o"
    "f the new, post-Communist Central Europe."
)


def test_compress():
    compressed = markov_chain_compress(text=TEXT)
    num_chars = len(TEXT)
    num_bytes = len(compressed)
    print(f"{num_chars} chars.")
    print(f"{num_bytes} bytes.")
    bits_per_char = num_bytes * 8 / num_chars
    print(f"{bits_per_char:.02f} bits per char.")
    print("Compressed:")
    print(compressed)

    decompressed = markov_chain_decompress(compressed)
    print("")
    print("Decompressed:")
    print(decompressed)


def test_general_compress(predictor: CharPredictor):
    prefix_len = 4
    # predictor = FixedLengthPredixMarkovPredictor(
    #     prefix_info=get_prefix_info(), prefix_len=prefix_len
    # )
    compressed = general_compress(
        text=TEXT, char_predictor=predictor, prefix_len=prefix_len
    )
    num_chars = len(TEXT)
    num_bytes = len(compressed)
    print(f"{num_chars} chars.")
    print(f"{num_bytes} bytes.")
    bits_per_char = num_bytes * 8 / num_chars
    print(f"{bits_per_char:.02f} bits per char.")
    print("Compressed:")
    print(compressed)

    decompressed = general_decompress(
        msg=compressed, char_predictor=predictor, prefix_len=prefix_len
    )
    print("")
    print("Decompressed:")
    print(decompressed)


# Test compression using a Markov-chain based predictor.
def test_markov_compress():
    prefix_len = 4
    test_general_compress(
        predictor=FixedLengthPredixMarkovPredictor(
            prefix_info=get_prefix_info(), prefix_len=prefix_len
        )
    )


# A char-predictor that uses a Pytorch model.
class ModelPredictor(CharPredictor):
    def __init__(self, model, prefix_len: int, eps=1e-8):
        self.model = model
        self.prefix_len = prefix_len
        self.eps = eps

    def predict_char(self, text: str) -> list[str] | None:
        if len(text) < self.prefix_len:
            padding = EMPTY_CHAR * (self.prefix_len - len(text))
            text = padding + text
        elif len(text) > self.prefix_len:
            text = text[-self.prefix_len :]

        # Turn the text into an array of one-hot vectors.
        in_array = text_nn_utils.snippet_to_array(snippet=text)

        # Unsqueeze the batch dimension.
        in_array = torch.unsqueeze(in_array, dim=0)

        predictions = self.model(in_array)
        predictions = torch.squeeze(predictions, 0)


        char_to_score = dict[str, float]()
        for i, char in enumerate(OUT_CHARS):
            score = predictions[i].item()
            char_to_score[char] = score

        # Make sure that no two scores are too close.
        # If they are, results might be nondeterministic.
        all_scores = list(char_to_score.values())
        all_scores.sort()
        for prev_score, score in zip(all_scores[:-1], all_scores[1:]):
            if abs(score - prev_score) < self.eps:
                return None

        low_score = min(char_to_score.values()) - 100

        chars = list(OUT_CHARS)
        chars.sort(key=lambda c: char_to_score.get(c, low_score), reverse=True)
        return chars


# Test compression using an NN.
def test_nn_compress():
    model = models.SimpleLetterModel()
    filepath = r"Model_Saves\Fully_Connected\1_0_0_During_624.model"
    model.load_state_dict(torch.load(filepath))
    model.eval()

    prefix_len = 4
    predictor = ModelPredictor(model=model, prefix_len=prefix_len)
    test_general_compress(predictor=predictor)


# Return the snippets contained in the given text.
def get_snippets_from_text(text: str, snippet_len: int) -> list[str]:
    result = list[str]()

    for i in range(len(text)):
        snippet = text[max(0, i - snippet_len) : i]
        filler = EMPTY_CHAR * (snippet_len - len(snippet))
        snippet = filler + snippet
        result.append(snippet)

    return result


def make_snippets_lists():
    snippet_len = 5
    files_dir = r"Corpora\OANC_GrAF\OANC_Text_Files"
    out_dir = rf"Datasets\Oanc_Snippets_Len{snippet_len}"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    out_template = os.path.join(out_dir, "snippets_{}.txt")

    snippets = list[str]()
    chunk_num = 0

    # Write the snippets to a file.
    def write_chunk(chunk: list[str]):
        # if len(snippets) == 0:  # type: ignore
        #     return
        # if len(snippets) < SNIPPETS_PER_FILE and not force:  # type: ignore
        #     return

        # chunk, snippets = snippets[:SNIPPETS_PER_FILE], snippets[SNIPPETS_PER_FILE:]  # type: ignore
        out_filename = out_template.format(chunk_num)
        with open(out_filename, "w") as file:
            for snippet in chunk[:-1]:
                file.write(f"{snippet}\n")
            file.write(chunk[-1])

    # Get the snippets in each file and save them.
    for filename in os.listdir(files_dir):
        filepath = os.path.join(files_dir, filename)
        with open(filepath, encoding="utf8") as file:
            text = clean_up_text(file.read())
        file_snippets = get_snippets_from_text(text=text, snippet_len=snippet_len)
        snippets.extend(file_snippets)

        if len(snippets) >= SNIPPETS_PER_FILE:
            chunk, snippets = snippets[:SNIPPETS_PER_FILE], snippets[SNIPPETS_PER_FILE:]
            write_chunk(chunk)
            chunk_num += 1
        # flush_snippets()

    if len(snippets) > 0:
        write_chunk(snippets)
    # flush_snippets(force=True)


# def shuffle_snippets_lists():
#     snippet_len = 5
#     in_dir = rf"Datasets\Oanc_Snippets_Len{snippet_len}"
#     # if not os.path.exists(in_dir):
#     #     os.mkdir(in_dir)
#     in_template = os.path.join(in_dir, "snippets_{}.txt")

#     out_dir = rf"Datasets\Oanc_Snippets_Len{snippet_len}_Shuffled"
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
#     out_template = os.path.join(out_dir, "snippets_{}.txt")


def main():
    # make_prefix_info()
    # markov_make_text()
    # test_compress()
    # test_markov_compress()
    test_nn_compress()
    # make_snippets_lists()


if __name__ == "__main__":
    main()
