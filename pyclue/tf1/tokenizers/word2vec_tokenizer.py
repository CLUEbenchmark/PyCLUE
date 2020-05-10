#!/usr/bin/python3

"""
@Author: Liu Shaoweihua
@Site: https://github.com/liushaoweihua
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import unicodedata

from pyclue.tf1.tokenizers.utils import convert_by_vocab, convert_to_unicode, load_vocab, whitespace_tokenize, \
    _is_control, _is_punctuation, _is_whitespace

"""Word2vec tokenizer classes."""


class Word2VecTokenizer(object):
    """Runs word2vec tokenization."""

    def __init__(self, vocab_file, norm_file=None, do_lower_case=True):
        self.vocab = load_vocab(vocab_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.norm_file = norm_file
        if norm_file:
            with open(norm_file, 'r') as f:
                norm_dict = f.readlines()
            self.norm_dict = [item.strip().split('\t') for item in norm_dict]
        else:
            self.norm_dict = None
        self.regex = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
        self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    def tokenize(self, text):
        if self.norm_dict:
            for item in self.norm_dict:
                text = text.replace(item[0], item[1])
        text = self.regex.sub('', text)
        split_tokens = []
        for token in self.basic_tokenizer.tokenize(text):
            split_tokens.append(token)

        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens, unknown=self.vocab.get('[UNK]'))

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids, unknown='[UNK]')


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True):
        """Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        text = self._tokenize_chars(text)

        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chars(self, text):
        output = []
        for char in text:
            output.append(" ")
            output.append(char)
            output.append(" ")
        return "".join(output)

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
