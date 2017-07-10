import nltk
import json
import argparse
from collections import Counter


def build_word2id(seq_path, min_word_count):
    """Creates word2id dictionary.
    
    Args:
        seq_path: String; text file path
        min_word_count: Integer; minimum word count threshold
        
    Returns:
        word2id: Dictionary; word-to-id dictionary
    """
    sequences = open(seq_path).readlines()
    num_seqs = len(sequences)
    counter = Counter()
    
    for i, sequence in enumerate(sequences):
        tokens = nltk.tokenize.word_tokenize(sequence.lower())
        counter.update(tokens)

        if i % 1000 == 0:
            print("[{}/{}] Tokenized the sequences.".format(i, num_seqs))

    # create a dictionary and add special tokens
    word2id = {}
    word2id['<pad>'] = 0
    word2id['<start>'] = 1
    word2id['<end>'] = 2
    word2id['<unk>'] = 3
    
    # if word frequency is less than 'min_word_count', then the word is discarded
    words = [word for word, count in counter.items() if count >= min_word_count]
    
    # add the words to the word2id dictionary
    for i, word in enumerate(words):
        word2id[word] = i + 4
    
    return word2id


def main(config):
    
    # build word2id dictionaries for source and target sequences
    src_word2id = build_word2id(config.src_path, config.min_word_count)
    trg_word2id = build_word2id(config.trg_path, config.min_word_count)
    
    # save word2id dictionaries
    with open(config.src_word2id_path, 'w') as f:
        json.dump(src_word2id, f)
    with open(config.trg_word2id_path, 'w') as f:
        json.dump(trg_word2id, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/src_train.txt')
    parser.add_argument('--trg_path', type=str, default='./data/trg_train.txt')
    parser.add_argument('--src_word2id_path', type=str, default='./data/src_word2id.json')
    parser.add_argument('--trg_word2id_path', type=str, default='./data/trg_word2id.json')
    parser.add_argument('--min_word_count', type=int, default=4)
    config = parser.parse_args()
    print (config)
    main(config)