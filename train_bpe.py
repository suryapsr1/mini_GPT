import json
import os
from collections import defaultdict

def get_stats(vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, str], int]:
    """
    Counts how often every adjacent pair of symbols appears across the vocabulary.

    Parameters:
        vocab: Dictionary of {word_tuple: frequency}
               e.g., {('u', 'n', 'd', 'e', 'r', '</w>'): 5}

    Returns:
        pairs: Dictionary of {(symbol_a, symbol_b): total_count}
    """
    pairs = defaultdict(int)

    for word, freq in vocab.items():
        # Iterate over consecutive neighbors: (0, 1), (1, 2), ..., (len-2, len-1)
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq

    return dict(pairs)




'''After finding the highest occuring pairs we also need a way to merge them'''

def merge_vocab(pair: tuple[str, str], vocab: dict[tuple[str, ...], int]) -> dict[tuple[str, ...], int]:
    """
    Finds all occurrences of `pair` (first, second) in the vocabulary tuples
    and merges them into a single token 'firstsecond'.
    
    Parameters:
        pair: The (symbol_a, symbol_b) pair to merge, e.g. ('d', 'e')
        vocab: The current vocabulary dictionary of {word_tuple: count}
        
    Returns:
        new_vocab: Updated vocabulary dictionary with the merged pair
    """
    first, second = pair
    merged_token = first + second
    new_vocab = {}

    for word, freq in vocab.items():
        new_word = []
        i = 0
        while i < len(word):
            # Check if current and next symbol match the target pair
            if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                new_word.append(merged_token)
                i += 2  # Skip both merged symbols
            else:
                new_word.append(word[i])
                i += 1
                
        new_vocab[tuple(new_word)] = freq

    return new_vocab


'''now a functional code which will first find the freaquencies pick the 
most frequently occuring character and merge them'''


def train_bpe(corpus_words: list[str], target_vocab_size: int):
    """
    Trains a Byte-Pair Encoding (BPE) vocabulary and merge table.
    
    Args:
        corpus_words: List of raw strings (e.g., words from our dataset).
        target_vocab_size: Total tokens desired in final vocabulary.
        
    Returns:
        vocab: Dictionary of words mapped to their current segmented tuples.
        merges: Ordered dictionary/list of (pair -> merged_token).
    """
    # Step 1: Initialize base vocabulary with end-of-word marker </w>
    vocab = {}
    for word in corpus_words:
        # Split word into characters + append </w>
        token_tuple = tuple(list(word) + ["</w>"])
        vocab[token_tuple] = vocab.get(token_tuple, 0) + 1

    # Base vocabulary: every unique individual character seen
    initial_tokens = set()
    #get the keys and then parse each character in the tuple and add it to the initial_tokens set
    for word_tuple in vocab.keys():
        for char in word_tuple:
            initial_tokens.add(char)
            
    print(f"Base character count: {len(initial_tokens)}")

    #we decide how much token info we need

    num_merges = target_vocab_size - len(initial_tokens)

    if num_merges <= 0:
        print("Target vocab size is already smaller than base characters.")
        return vocab, []

    merges = []

    # Step 2: The Training Loop
    for step in range(num_merges):
        pair_stats = get_stats(vocab)
        if not pair_stats:
            print("No more pairs to merge.")
            break

        # Pick the most frequent pair
        #pair_stats is a dictionary where keys are pairs and values are their counts. max(pair_stats, key=pair_stats.get) returns the pair with the highest count.
        best_pair = max(pair_stats, key=pair_stats.get)
        
        # Merge it across the vocabulary
        vocab = merge_vocab(best_pair, vocab)
        
        # Save the rule this later tells tokeniser how to build tokens [('d', 'e'), ('de', 'r')]
        merges.append(best_pair)
        print(f"Merge {step + 1}/{num_merges}: {best_pair} -> '{best_pair[0] + best_pair[1]}'")

    return vocab, merges

'''BPE starts from characters.
It repeatedly merges the most common adjacent pair.
This creates larger tokens like de, ing, tion, etc.
The merges list is what ultimately defines the learned merge rules'''

'''From these learned merges we can build a tokenizer that will segment 
new words into the learned tokens.'''



'''if __name__ == "__main__":
    sample_corpus = [
        "low", "low", "low", "low", "low",
        "lowest", "lowest",
        "newer", "newer", "newer", "newer", "newer", "newer",
        "wider", "wider", "wider"
    ]

    # Let's run until we add 5 new merged subwords
    target_size = 11 + 5  # Base letters + 5 merges
    trained_vocab, merge_rules = train_bpe(sample_corpus, target_vocab_size=target_size)

    print("\n--- Final Learned Merge Order ---")
    for idx, rule in enumerate(merge_rules, 1):
        print(f"{idx}. {rule}")

    print("\n--- Final Corpus Representation ---")
    for segmented_word, freq in trained_vocab.items():
        print(f"  {segmented_word} (count: {freq})")'''


def bpe_tokenize_word(word: str, merges: list[tuple[str,str]]) -> list[str]:
    """
    Tokenizes a single word using the learned BPE merges.
    
    Args:
        word: The input word to tokenize.
        merges: List of learned merge pairs in order of application.
        
    Returns:
        tokens: List of tokens after applying BPE merges.
    """
    #stage 1: split the word into characters and append </w>
    tokens = list(word) + ["</w>"]

    #stage 2: apply merges in order
    for first, second in merges:
        merged_token= first + second

        new_tokens=[]
        i=0
        while i < len(tokens):

            if i<len(tokens)-1 and tokens[i]==first and tokens[i+1]==second:
                new_tokens.append(merged_token)
                i+=2
            else: 
                new_tokens.append(tokens[i])
                i+=1
        tokens = new_tokens
    return tokens


'''if __name__ == "__main__":
    # Suppose we only learned two merges:
    test_merges = [('t', 'h'), ('th', 'e')]
    
    result = bpe_tokenize_word("the", test_merges)
    print("Tokens for 'the':", result)'''


def encode(text: str , merges: list[tuple[str, str]], token_to_id:dict[str,int]) -> list[int]:
    """
    Converts a raw sentence into a list of integer token IDs.
    """

    words= text.strip().split()
    token_ids = []

    for word in words: 
        # Step 1: Subword segmentation using the helper we just verified
        subwords = bpe_tokenize_word(word, merges)

        #Step 2: convert in integre id

        for tok in subwords:
            idx= token_to_id.get(tok, token_to_id.get("<UNK>", 0))
            token_ids.append(idx)

    return token_ids


'''
if __name__ == "__main__":
    test_merges = [('t', 'h'), ('th', 'e')]

    # Toy token-to-ID dictionary
    test_token_to_id = {
        "<UNK>": 0,
        "<PAD>": 1,
        "</w>": 2,
        "the": 3,
        "u": 4,
        "n": 5
    }

    sentence = "the un"
    encoded_output = encode(sentence, test_merges, test_token_to_id)
    print("Encoded token IDs:", encoded_output)'''



def decode(token_ids: list[int], id_to_token: dict[int, str]) -> str:
    """
    Reconstructs clean, human-readable text from a list of integer token IDs.
    """
    # 1. Map each integer ID back to its string subword token
    subwords = [id_to_token.get(idx, "<UNK>") for idx in token_ids]

    # 2. Join all subwords into a single continuous string
    raw_text = "".join(subwords)

    # 3. Replace the end-of-word marker </w> with spaces and strip padding
    clean_text = raw_text.replace("</w>", " ").strip()

    return clean_text


'''if __name__ == "__main__":
    # Inverted dictionary matching our earlier encode test
    test_token_to_id = {
        "<UNK>": 0,
        "<PAD>": 1,
        "</w>": 2,
        "the": 3,
        "u": 4,
        "n": 5
    }
    test_id_to_token = {idx: tok for tok, idx in test_token_to_id.items()}

    # Simulated output from model inference
    sample_generated_ids = [3, 2, 4, 5, 2]

    # Run decoding
    reconstructed_text = decode(sample_generated_ids, test_id_to_token)
    print("Generated IDs:     ", sample_generated_ids)
    print("Reconstructed Text:", f"'{reconstructed_text}'")'''




def save_tokenizer(
    merges: list[tuple[str, str]],
    token_to_id: dict[str, int],
    export_json: str
) -> None:
    """Save merge rules and token IDs as a JSON tokenizer artifact."""
    with open(export_json, "w", encoding="utf-8") as f:
        json.dump(
            {"merges": merges, "token_to_id": token_to_id},
            f,
            ensure_ascii=False,
            indent=2
        )


def load_tokenizer(
    tokenizer_json: str
) -> tuple[list[tuple[str, str]], dict[str, int], dict[int, str]]:
    """Load merge rules, token IDs, and the inverse token-ID mapping."""
    with open(tokenizer_json, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    if not isinstance(tokenizer, dict):
        raise ValueError("Tokenizer file must contain a JSON object.")

    merges_data = tokenizer.get("merges")
    token_to_id = tokenizer.get("token_to_id")
    if not isinstance(merges_data, list) or not isinstance(token_to_id, dict):
        raise ValueError("Tokenizer file must contain 'merges' and 'token_to_id'.")

    if any(
        not isinstance(pair, list)
        or len(pair) != 2
        or not all(isinstance(token, str) for token in pair)
        for pair in merges_data
    ):
        raise ValueError("Tokenizer 'merges' must be a list of two-token pairs.")

    if any(
        not isinstance(token, str) or not isinstance(token_id, int)
        for token, token_id in token_to_id.items()
    ):
        raise ValueError("Tokenizer 'token_to_id' must map strings to integers.")

    merges = [(pair[0], pair[1]) for pair in merges_data]
    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    return merges, token_to_id, id_to_token



def train_and_export(
    corpus_path: str = "data/input.txt",
    export_json: str = "tokenizer/bpe_rules.json",
    num_merges: int = 300
):
    """
    Learns BPE merge pairs on the raw text file and exports the tokenizer artifact.
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Corpus not found at {corpus_path}")

    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print(f"Loaded corpus: {len(raw_text):,} characters")

    # 1. Base vocabulary initialization
    # Extract unique characters and create word representations ending in </w>
    words = raw_text.strip().split()
    vocab: dict[tuple[str, ...], int] = {}
    for word in words:
        tokenized_word = tuple(list(word) + ["</w>"])
        vocab[tokenized_word] = vocab.get(tokenized_word, 0) + 1

    # Track initial characters
    unique_chars = set()
    for word in vocab:
        unique_chars.update(word)

    # Build base token ID dictionary
    special_tokens = ["<UNK>", "<PAD>"]
    token_to_id = {tok: idx for idx, tok in enumerate(special_tokens)}
    for ch in sorted(unique_chars):
        if ch not in token_to_id:
            token_to_id[ch] = len(token_to_id)

    # 2. Iterative BPE pair counting and merging
    merges = []
    current_vocab = vocab
    print(f"Starting BPE training for {num_merges} merges...")

    for i in range(num_merges):
        pairs = get_stats(current_vocab)
        if not pairs:
            print(f"Early stop at merge {i}: No further mergeable pairs found.")
            break

        best_pair = max(pairs, key=pairs.get)
        current_vocab = merge_vocab(best_pair, current_vocab)
        merges.append(best_pair)

        # The merged entity becomes a new token in our vocabulary
        merged_token = "".join(best_pair)
        if merged_token not in token_to_id:
            token_to_id[merged_token] = len(token_to_id)

        if (i + 1) % 50 == 0 or (i + 1) == num_merges:
            print(f"Merge {i + 1:03d}/{num_merges} completed: {best_pair} -> '{merged_token}'")

    # 3. Save tokenizer rules
    export_dir = os.path.dirname(export_json)
    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
    save_tokenizer(merges, token_to_id, export_json)
    print(f"Final Vocabulary Size: {len(token_to_id)} tokens\n")
    return merges, token_to_id

if __name__ == "__main__":
    train_and_export(
        corpus_path="data/input.txt",
        export_json="tokenizer/bpe_rules.json",
        num_merges=300
    )
