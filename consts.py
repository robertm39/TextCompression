ALPHABET = "abcdefghijklmnopqrstuvwxyz"
OK_CHARS = ALPHABET + ALPHABET.upper() + "0123456789 "
OK_ALPHA_CHARS = ALPHABET + ALPHABET.upper() + " "

# A for the 1st predicted char, B for the second, etc.
PRED_CHAR_TOKENS = ALPHABET.upper() + "_"
MULTIPLE_FIRST_TOKENS = "0123456789"

OUT_CHARS = ALPHABET + " "
# SNIPPETS_PER_FILE = 1024 * 1024
SNIPPETS_PER_FILE = 1024 * 128
EMPTY_CHAR = "~"

CHARS_BY_FREQUENCY = " etaoinshrdlucmfwygpbvkqjxz"