# must same as the ASR symbols
_pause = ["~", "sos", "eos", "unk", "<blank>", "sp", "sil", "#0", "#1", "#2", "#3", "#4"]

_initials = [
    "b",
    "c",
    "ch",
    "d",
    "f",
    "g",
    "h",
    "j",
    "k",
    "l",
    "m",
    "n",
    "p",
    "q",
    "r",
    "s",
    "sh",
    "t",
    "w",
    "x",
    "y",
    "z",
    "zh",
]

_finals = [
    "a",
    "ai",
    "an",
    "ang",
    "ao",
    "e",
    "ei",
    "en",
    "eng",
    "er",
    "i",
    "ia",
    "ian",
    "iang",
    "iao",
    "ie",
    "ii",
    "iii",
    "in",
    "ing",
    "iong",
    "iou",
    "o",
    "ong",
    "ou",
    "u",
    "ua",
    "uai",
    "uan",
    "uang",
    "uei",
    "uen",
    "ueng",
    "uo",
    "v",
    "van",
    "ve",
    "vn",
    "xr"
]

_cmu = [
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
    "P",
    "B",
    "CH",
    "D",
    "DH",
    "F",
    "G",
    "HH",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

_punc = [
    "?",
    "!",
    ",",
    ".",
    ";",
    ":",
    "？",
    "！",
    "，",
    "。",
    "；",
    "：",
    "、",
]

symbols = _pause + _initials + _finals + _cmu + _punc
# print(len(symbols))   # 126
# print(symbols[74])   # 126
# print(symbols[113])   # 126
# for i in range(len(symbols)):
#     print(symbols[i])

tone_symbols = ['~',
                '0',
                '1',
                '2',
                '3',
                '4',
                '5',
                '6', # english no stress
                '7', # english stress 0
                '8', # english stress 1
                '9', # english stress 2
]

language_symbols = ['~', '1', '2', '3']
# print(len(symbols))   # 121
# for i in range(len(symbols)):
#     print(i, symbols[i])