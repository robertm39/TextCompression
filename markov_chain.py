from typing import cast, Mapping
from abc import ABC, abstractmethod
import os
import json
import random
from unidecode import unidecode

from bitlist import bitlist
import torch

from consts import *
import models
import text_nn_utils
import huffman_coding

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
def get_prefix_info_from_text(filename: str, prefix_len: int) -> PrefixInfo:
    # Read the file.
    with open(filename, "r", encoding="utf8") as file:
        text = file.read()

    # Remove unacceptable characters and clean up whitespace.
    text = "".join([c for c in text if c in OK_ALPHA_CHARS])
    text = text.replace("\n", " ").lower()
    while "  " in text:
        text = text.replace("  ", " ")

    # Gather the prefix information.
    # prefix_len = 4
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


# Make the prefix-info for the specified file and save it in the specified file.
def make_prefix_info_for_file(
    in_filepath: str, out_filepath: str, max_prefix_len: int
) -> None:
    # If the file already exists, don't overwrite it.
    if os.path.exists(out_filepath):
        return

    text = ""
    # Read the text from the file. If we throw an exception, just skip this file.
    try:
        with open(in_filepath, "r", encoding="utf8") as file:
            text = file.read()
            text = unidecode(text)
            text = clean_up_text(text)
    except KeyboardInterrupt:
        raise
    except:
        pass

    # Get the prefix-info.
    current_prefix = ""
    prefix_to_freqs = PrefixInfo()
    for c in text:
        # Take note of this character.
        # For each sub-prefix (including the whole prefix),
        # take note that that sub-prefix led to this character.
        for sub_prefix_len in range(1, len(current_prefix) + 1):
            sub_prefix = current_prefix[-sub_prefix_len:]
            if not sub_prefix in prefix_to_freqs:
                # prefix_to_freqs[sub_prefix] = {c: 0 for c in OUT_CHARS}
                prefix_to_freqs[sub_prefix] = dict[str, int]()
            freqs = prefix_to_freqs[sub_prefix]
            if not c in freqs:
                freqs[c] = 0
            # prefix_to_freqs[sub_prefix][c] += 1
            freqs[c] += 1

        # Update the current prefix.
        current_prefix = current_prefix + c
        current_prefix = current_prefix[-max_prefix_len:]

    # Save the data to file.
    with open(out_filepath, "w") as file:
        json.dump(prefix_to_freqs, file)


# Make the prefix-info, split into individual files.
def make_prefix_info_filewise() -> None:
    in_dir = r"Corpora\Wikipedia\Wikipedia_Smaller"
    out_dir = r"Filewise_Prefix_Info\Wikipedia_Smaller"
    out_filename_template = "freqs_{}.json"

    max_prefix_len = 10
    for i, in_filename in enumerate(os.listdir(in_dir)):
        # Get the name of the output file.
        out_filename = out_filename_template.format(i)
        out_filepath = os.path.join(out_dir, out_filename)
        in_filepath = os.path.join(in_dir, in_filename)
        print(in_filename)
        make_prefix_info_for_file(
            in_filepath=in_filepath,
            out_filepath=out_filepath,
            max_prefix_len=max_prefix_len,
        )


def make_prefix_info():
    files_dir = r"Corpora\OANC_GrAF\OANC_Text_Files"
    out_filepath = r"Prefix_Info\prefix_info.json"
    prefix_info = PrefixInfo()

    # Only use one character to predict the next.
    prefix_len = 5
    for filename in os.listdir(files_dir):
        print(filename)
        filepath = os.path.join(files_dir, filename)
        file_prefix_info = get_prefix_info_from_text(
            filename=filepath, prefix_len=prefix_len
        )
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

    # Put the characters that are most frequent in general first.
    chars.sort(key=lambda c: CHARS_BY_FREQUENCY.index(c))

    # Then sort by the frequencies given.
    chars.sort(key=lambda c: freqs.get(c, 0), reverse=True)
    return chars


def get_prefix_info() -> dict[str, dict[str, int]]:
    # in_file = r"Prefix_Info\prefix_info_oanc_len1_lowercase.json"
    # in_file = r"Prefix_Info\prefix_info_oanc_len2_lowercase.json"
    # in_file = r"Prefix_Info\prefix_info_oanc_len3_lowercase.json"
    # in_file = r"Prefix_Info\prefix_info_oanc_len4_lowercase.json"
    in_file = r"Prefix_Info\prefix_info_oanc_len5_lowercase.json"
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
            # return None
            return list(CHARS_BY_FREQUENCY)

        # Get the prefix.
        prefix = text[-self.prefix_len :]
        freqs = self.prefix_info.get(prefix, None)
        if freqs is None:
            # return None
            freqs = {c: 0 for c in OUT_CHARS}
        else:
            freqs = freqs.copy()
            for c in OUT_CHARS:
                if not c in freqs:
                    freqs[c] = 0

        sorted_chars = get_chars_sorted_by_frequency_descending(freqs=freqs)
        return sorted_chars


PADDING_HEADER_LENGTH = 3


# Return the given bits, with a padding header and padding to be a whole number of bytes.
def get_padded(bits: bitlist) -> bitlist:
    # Pad the bits so they make up a whole number of bytes.
    num_bits = len(bits) + PADDING_HEADER_LENGTH
    padding_needed = 8 - (num_bits % 8)
    padding = bitlist(0, length=padding_needed)
    padding_header = bitlist(padding_needed, length=PADDING_HEADER_LENGTH)
    bits = padding_header + bits + padding
    return bits


# Return the given bits, with the padding header and padding removed.
def get_unpadded(bits: bitlist) -> bitlist:
    # Remove the padding.
    padding_header, bits = bits[:PADDING_HEADER_LENGTH], bits[PADDING_HEADER_LENGTH:]  # type: ignore
    padding_len = int(padding_header)
    bits = bits[:-padding_len]  # type: ignore
    # bits = cast(bitlist, bits)
    return bits


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

    # # Pad the bits so they make up a whole number of bytes.
    # num_bits = len(bits) + PADDING_HEADER_LENGTH
    # padding_needed = 8 - (num_bits % 8)
    # padding = bitlist(0, length=padding_needed)
    # padding_header = bitlist(padding_needed, length=PADDING_HEADER_LENGTH)
    # bits = padding_header + bits + padding
    bits = get_padded(bits=bits)

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
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.model = model.to(device)
        self.model = model
        self.prefix_len = prefix_len
        self.eps = eps

    def predict_char(self, text: str) -> list[str]:
        if len(text) < self.prefix_len:
            padding = EMPTY_CHAR * (self.prefix_len - len(text))
            text = padding + text
        elif len(text) > self.prefix_len:
            text = text[-self.prefix_len :]

        # Turn the text into an array of one-hot vectors.
        in_array = text_nn_utils.snippet_to_array(snippet=text)

        # Unsqueeze the batch dimension.
        in_array = torch.unsqueeze(in_array, dim=0)

        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # in_array = in_array.to(device)
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
                # return None
                return list(CHARS_BY_FREQUENCY)

        low_score = min(char_to_score.values()) - 100

        chars = list(OUT_CHARS)
        chars.sort(key=lambda c: char_to_score.get(c, low_score), reverse=True)
        return chars


def get_model():
    # model = models.SimpleLetterModel()
    model = models.ConvLetterModel()
    # filepath = r"Model_Saves\Fully_Connected\1_0_0_During_624.model"
    # filepath = r"Model_Saves\Fully_Connected_4\1_Epoch_315_1_After.model"
    # filepath = r"Model_Saves\Conv_6\1_Epoch_2_1_After.model"
    # filepath = r"Model_Saves\Conv_7\1_Epoch_0_0_During_5000.model"
    filepath = r"Model_Saves\Conv_7\1_Epoch_7_1_After.model"
    model.load_state_dict(torch.load(filepath))
    model.eval()
    return model


# Test compression using an NN.
def test_nn_compress():
    # model = models.SimpleLetterModel()
    # # filepath = r"Model_Saves\Fully_Connected\1_0_0_During_624.model"
    # filepath = r"Model_Saves\Fully_Connected_4\1_Epoch_315_1_After.model"
    # model.load_state_dict(torch.load(filepath))
    # model.eval()
    model = get_model()

    prefix_len = 10
    predictor = ModelPredictor(model=model, prefix_len=prefix_len)
    test_general_compress(predictor=predictor)


def add_token(tok_to_count: dict[str, int], tok: str) -> None:
    if not tok in tok_to_count:
        tok_to_count[tok] = 0
    tok_to_count[tok] += 1


def add_tokens(
    tok_to_count: dict[str, int], add_tok_to_count: Mapping[str, int]
) -> None:
    for tok, count in add_tok_to_count.items():
        if not tok in tok_to_count:
            tok_to_count[tok] = 0
        tok_to_count[tok] += count


# Return the token frequencies in the given text for the given predictor.
def get_token_frequencies(
    predictor: CharPredictor, text: str, prefix_len: int
) -> dict[str, int]:
    result = dict[str, int]()

    current_prefix = ""
    num_first = 0
    for c in text:
        # Determine what token to use.
        pred_chars = predictor.predict_char(current_prefix)
        is_first = False
        token = None
        if pred_chars is None:
            print("No prediction!!!")
            token = c
        elif c in pred_chars:
            pred_index = pred_chars.index(c)
            if pred_index == 0:
                num_first += 1
                is_first = True
            else:
                token = PRED_CHAR_TOKENS[pred_chars.index(c)]
        else:
            token = c

        # Add the token.
        if not is_first:
            if num_first > 0:
                max_first_token = len(MULTIPLE_FIRST_TOKENS) - 1
                first_tokens = list[str]()
                while num_first > max_first_token:
                    first_tokens.append(MULTIPLE_FIRST_TOKENS[-1])
                    num_first -= max_first_token
                if num_first > 0:
                    first_tokens.append(MULTIPLE_FIRST_TOKENS[num_first])
                # first_token = MULTIPLE_FIRST_TOKENS[num_first]
                for tok in first_tokens:
                    add_token(tok_to_count=result, tok=tok)
                num_first = 0

            if token is not None:
                if not token in result:
                    result[token] = 0
                result[token] += 1

        # Update the prefix.
        current_prefix = current_prefix + c
        if len(current_prefix) > prefix_len:
            current_prefix = current_prefix[-prefix_len:]
        # while len(current_prefix) > prefix_len:
        #     current_prefix = current_prefix[1:]

    if num_first > 0:
        max_first_token = len(MULTIPLE_FIRST_TOKENS) - 1
        first_tokens = list[str]()
        while num_first > max_first_token:
            first_tokens.append(MULTIPLE_FIRST_TOKENS[-1])
            num_first -= max_first_token
        if num_first > 0:
            first_tokens.append(MULTIPLE_FIRST_TOKENS[num_first])
        # first_token = MULTIPLE_FIRST_TOKENS[num_first]
        for tok in first_tokens:
            add_token(tok_to_count=result, tok=tok)
        num_first = 0

    return result


# Return the char-predictor and the prefix length.
def get_predictor() -> tuple[CharPredictor, int]:
    prefix_len = 5
    predictor = FixedLengthPredixMarkovPredictor(
        prefix_info=get_prefix_info(), prefix_len=prefix_len
    )
    return predictor, prefix_len


# Determine the frequencies of the different tokens we will use.
def determine_token_frequencies():
    # model = models.SimpleLetterModel()
    # filepath = r"Model_Saves\Fully_Connected_4\1_Epoch_315_1_After.model"
    # model.load_state_dict(torch.load(filepath))
    # model.eval()
    # model = get_model()

    # prefix_len = 10
    # predictor = ModelPredictor(model=model, prefix_len=prefix_len)
    predictor, prefix_len = get_predictor()

    # # A for the 1st predicted char, B for the second, etc.
    # pred_char_tokens = ALPHABET.upper() + "_"
    # tok_to_count = dict[str, int]()

    out_dir = r"Token_Counts"
    # Determine which files we've already made.
    already_made_files = set[str]()
    for filename in os.listdir(out_dir):
        already_made_files.add(os.path.join(out_dir, filename))

    out_template = "{}_token_counts.json"
    out_filepath_template = os.path.join(out_dir, out_template)
    texts_dir = r"Corpora\OANC_GrAF\OANC_Text_Files"
    for filename in os.listdir(texts_dir):
        file_num = int(filename.split("_")[0])

        out_filepath = out_filepath_template.format(file_num)
        if out_filepath in already_made_files:
            continue

        print(filename)
        # Get the text of the file.
        filepath = os.path.join(texts_dir, filename)
        with open(filepath, encoding="utf8") as file:
            text = file.read()
        text = clean_up_text(text)

        # Determine the token-frequencies in the file.
        file_tok_to_count = get_token_frequencies(
            predictor=predictor, text=text, prefix_len=prefix_len
        )

        # Save the frequencies to file.
        with open(out_filepath, "w") as file:
            json.dump(file_tok_to_count, file)

        # # Update the overall frequencies.
        # for tok, count in file_tok_to_count.items():
        #     if not tok in tok_to_count:
        #         tok_to_count[tok] = 0
        #     tok_to_count[tok] += count

    # out_filename = "token_counts.json"
    # with open(out_filename, "w") as file:
    #     json.dump(tok_to_count, file)


def combine_token_counts():
    freqs_dir = r"Token_Counts"
    tok_to_count = dict[str, int]()

    for filename in os.listdir(freqs_dir):
        filepath = os.path.join(freqs_dir, filename)
        with open(filepath) as file:
            file_tok_to_count = json.load(file)
        add_tokens(tok_to_count=tok_to_count, add_tok_to_count=file_tok_to_count)

    out_filename = "token_counts.json"
    with open(out_filename, "w") as file:
        json.dump(tok_to_count, file)


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


# Convert the given message into compression tokens.
def get_compression_tokens(
    msg: str, predictor: CharPredictor, prefix_len: int
) -> list[str]:
    result = list[str]()

    num_first = 0
    current_prefix = ""
    for c in msg:
        pred_chars = predictor.predict_char(current_prefix)
        if pred_chars is None:
            raise ValueError()

        # C is the first predicted character.
        is_first = False
        if c == pred_chars[0]:
            num_first += 1
            is_first = True

        if not is_first:
            # Add the pending first-char token.
            if num_first > 0:
                first_tokens = list[str]()
                max_first_token = len(MULTIPLE_FIRST_TOKENS) - 1
                while num_first > max_first_token:
                    num_first -= max_first_token
                    first_tokens.append(MULTIPLE_FIRST_TOKENS[-1])

                if num_first > 0:
                    first_tokens.append(MULTIPLE_FIRST_TOKENS[num_first])

                result.extend(first_tokens)
                num_first = 0

            char_index = pred_chars.index(c)
            token = PRED_CHAR_TOKENS[char_index]
            result.append(token)

        # Update the prefix.
        current_prefix = current_prefix + c
        if len(current_prefix) > prefix_len:
            current_prefix = current_prefix[-prefix_len:]

    # Add the final first-tokens, if any.
    if num_first > 0:
        first_tokens = list[str]()
        max_first_token = len(MULTIPLE_FIRST_TOKENS) - 1
        while num_first > max_first_token:
            num_first -= max_first_token
            first_tokens.append(MULTIPLE_FIRST_TOKENS[-1])

        if num_first > 0:
            first_tokens.append(MULTIPLE_FIRST_TOKENS[num_first])

        result.extend(first_tokens)
        num_first = 0

    return result


def get_huffman_tree() -> huffman_coding.HuffmanTree:
    # Read the token count information.
    tok_to_count_filename = r"token_counts.json"
    with open(tok_to_count_filename) as file:
        tok_to_count = cast(dict[str, int], json.load(file))

    # Get the huffman tree.
    huffman_tree = huffman_coding.get_huffman_tree(tok_to_count=tok_to_count)
    return huffman_tree


# Encode the given message using a character predictor and huffman-tree compression.
def huffman_encode(msg: str, predictor: CharPredictor, prefix_len: int) -> bytes:
    # # Read the token count information.
    # tok_to_count_filename = r"token_counts.json"
    # with open(tok_to_count_filename) as file:
    #     tok_to_count = cast(dict[str, int], json.load(file))

    # # Get the huffman tree.
    # huffman_tree = huffman_coding.get_huffman_tree(tok_to_count=tok_to_count)
    huffman_tree = get_huffman_tree()

    # # Print the number of bits for each token.
    # for token in tok_to_count:
    #     print(f"{token}: {len(huffman_tree.get_bits(tok=token))}")  # type: ignore

    # Turn the message into a list of tokens.
    tokens = get_compression_tokens(msg=msg, predictor=predictor, prefix_len=prefix_len)
    tokens = "".join(tokens)

    # # TODO compress the tokens and return the result.
    bits = huffman_tree.encode(toks=tokens)
    if bits is None:
        raise ValueError()

    bits = get_padded(bits)
    return bits.to_bytes()


# Manages a text and the current prefix.
class TextPrefixManager:
    def __init__(self, prefix_len: int) -> None:
        self.prefix_len = prefix_len
        # self.current_prefix = ""
        self._text = list[str]()

    # Add one character to the text.
    def add_str(self, s: str) -> None:
        # if len(char) > 1:
        #     raise ValueError()
        self._text.extend(s)

    @property
    def text(self) -> str:
        return "".join(self._text)

    @property
    def current_prefix(self) -> str:
        return "".join(self._text[-self.prefix_len :])


# Decode the given message using a character predictor and huffman-tree decompression.
def huffman_decode(encoded: bytes, predictor: CharPredictor, prefix_len: int) -> str:
    huffman_tree = get_huffman_tree()

    # Get the bits and remove the padding.
    bits = bitlist(encoded)
    bits = get_unpadded(bits=bits)

    # Get the compression tokens.
    toks = huffman_tree.decode(bits=bits)
    if toks is None:
        raise ValueError()

    text_mng = TextPrefixManager(prefix_len=prefix_len)
    for tok in toks:
        if tok in MULTIPLE_FIRST_TOKENS:
            num_toks = int(tok)
            for _ in range(num_toks):
                pred_chars = predictor.predict_char(text=text_mng.current_prefix)
                if pred_chars is None:
                    raise ValueError()
                text_mng.add_str(pred_chars[0])
        else:
            pred_chars = predictor.predict_char(text=text_mng.current_prefix)
            if pred_chars is None:
                raise ValueError()
            char_index = PRED_CHAR_TOKENS.find(tok)
            char = pred_chars[char_index]
            text_mng.add_str(char)

    return text_mng.text


def test_huffman_encode():
    msg = clean_up_text(TEXT)
    # model = models.SimpleLetterModel()
    # # filepath = r"Model_Saves\Fully_Connected\1_0_0_During_624.model"
    # filepath = r"Model_Saves\Fully_Connected_4\1_Epoch_315_1_After.model"
    # model.load_state_dict(torch.load(filepath))
    # model.eval()
    model = get_model()
    prefix_len = 10
    predictor = ModelPredictor(model=model, prefix_len=prefix_len)

    encoded = huffman_encode(msg=msg, predictor=predictor, prefix_len=prefix_len)
    print(f"Text:\n{msg}")
    print("")
    print(f"Num chars: {len(TEXT)}")
    print(f"Bits per char: {len(encoded) * 8 / len(msg):.02f}")
    print(f"Encoded:\n{encoded}")
    print("")

    decoded = huffman_decode(
        encoded=encoded, predictor=predictor, prefix_len=prefix_len
    )
    print(f"Same: {msg == decoded}")
    print(f"Decoded:\n{decoded}")


# Print the expected number of bits per character.
def get_expected_bits_per_character():
    # Read the token count information.
    tok_to_count_filename = r"token_counts_prefix_5.json"
    with open(tok_to_count_filename) as file:
        tok_to_count = cast(dict[str, int], json.load(file))

    # Get the huffman tree.
    huffman_tree = huffman_coding.get_huffman_tree(tok_to_count=tok_to_count)

    # Print the number of bits for each token.
    total_num_bits = 0
    total_num_chars = 0
    for token, count in tok_to_count.items():
        if token.isnumeric():
            num_chars = int(token)
        else:
            num_chars = 1
        total_num_bits += len(huffman_tree.get_bits(tok=token)) * count  # type: ignore
        total_num_chars += num_chars * count

    bits_per_char = total_num_bits / total_num_chars
    print(f"Expected bits per char: {bits_per_char:.02f}")


def main():
    # make_prefix_info()
    make_prefix_info_filewise()
    # markov_make_text()
    # test_compress()
    # test_markov_compress()
    # test_nn_compress()
    # make_snippets_lists()
    # determine_token_frequencies()
    # combine_token_counts()
    # test_huffman_encode()
    # get_expected_bits_per_character()


if __name__ == "__main__":
    main()
