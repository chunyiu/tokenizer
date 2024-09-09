import regex as re
import unicodedata

def get_paircounts(ids, counts=None):
    counts = {} if counts is None else counts
    for pair in zip(ids,ids[1:]):
        counts[pair] = counts.get(pair,0)+1
    return counts

def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            newids.append(idx)
            i+=2
        else:
            newids.append(ids[i])
            i+=1
    return newids

def replace_control_characters(s: str) -> str:
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch)
        else:
            chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

def render_token(t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
SPECIAL_TOKENS = {
    '<|endoftext|>': 100257,
    '<|fim_prefix|>': 100258,
    '<|fim_middle|>': 100259,
    '<|fim_suffix|>': 100260,
    '<|endofprompt|>': 100261
}

class RegexTokenizer():
    
    def __init__(self, pattern=None):
        self.merges = {}
        self.pattern = SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = {}
        self.register_special_tokens(SPECIAL_TOKENS)
    
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size-256

        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            paircounts = {}
            for chunk_ids in ids:
                get_paircounts(chunk_ids, paircounts)
            pair = max(paircounts,key=paircounts.get)
            idx = 256+i
            ids = [merge(chunk_ids,pair,idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]]+vocab[pair[1]]
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {paircounts[pair]} occurrences")
        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v:k for k,v in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"Invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8",errors="replace")
        return text

    def encode_chunk(self, text_bytes):
        ids = list(text_bytes)
        while len(ids) >= 2:
            paircounts = get_paircounts(ids)
            pair = min(paircounts, key=lambda p: self.merges.get(p,float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8")
            chunk_ids = self.encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k:v for k,v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        ids = []
        for chunk in special_chunks:
            if chunk in special:
                ids.append(special[chunk])
            else:
                ids.extend(self.encode_ordinary(chunk))
        return ids
    
    def save(self, file_prefix):
        model_file = file_prefix+".model"
        with open(model_file, "w") as file:
            file.write(f"RegexTokenizer v1")
            file.write(f"{self.pattern}\n")
            file.write(f"{len(self.special_tokens)}\n")
            for special_token,idx in self.special_tokens.items():
                file.write(f"{special_token} {idx}\n")
            for idx1,idx2 in self.merges:
                file.write(f"{idx1} {idx2}\n")
        
        vocab_file = file_prefix+".vocab"
        inverted_merges = {idx: pair for pair,idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf=8") as file:
            for idx,token in self.vocab.items():
                s = render_token(token)
                if idx in inverted_merges:
                    idx0,idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    file.write(f"[{s0}][{s0}] -> [{s}] {idx}\n")
                else:
                    file.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, encoding="utf-8") as file:
            version = file.readlline().strip()
            assert version == "Regex Tokenizer v1"
            self.pattern = file.readline().strip()
            num_special = int(file.readline().strip())
            for _ in range(num_special):
                special,special_idx = file.readline().strip().split()
                special_tokens[special] = int(special_idx)
            for line in file:
                idx1,idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens