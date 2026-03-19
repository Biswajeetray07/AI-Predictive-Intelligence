"""
Tokenizer wrapper for the NLP model.

Wraps HuggingFace's DeBERTa-v3-base tokenizer with project-specific
defaults and helper methods for batch encoding.
"""

import torch
from transformers import AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256  # Financial text is usually short (headlines, titles, descriptions)


class NLPTokenizer:
    """Thin wrapper around HuggingFace tokenizer with project defaults."""

    def __init__(self, model_name: str = MODEL_NAME, max_length: int = MAX_LENGTH):
        self.model_name = model_name
        self.max_length = max_length
        logging.info(f"Loading tokenizer: {model_name}")
        # Try fast tokenizer first (more compatible), then fall back to slow
        self.tokenizer = None
        for model_to_try in [model_name, "bert-base-uncased"]:
            for use_fast in [True, False]:
                try:
                    logging.info(f"Attempting to load tokenizer: {model_to_try} (use_fast={use_fast})")
                    tok = AutoTokenizer.from_pretrained(model_to_try, use_fast=use_fast)
                    if tok is not None:
                        self.tokenizer = tok
                        self.model_name = model_to_try
                        logging.info(f"✅ Successfully loaded tokenizer: {model_to_try} (use_fast={use_fast})")
                        return
                except Exception as e:
                    logging.warning(f"⚠️ Failed to load {model_to_try} (use_fast={use_fast}): {e}")
        
        if self.tokenizer is None:
            raise RuntimeError(
                f"Failed to load any tokenizer (tried {model_name} and bert-base-uncased). "
                f"Please run: pip install sentencepiece protobuf transformers --upgrade"
            )

    def encode_batch(self, texts: list, return_tensors: str = 'pt') -> dict:
        """
        Tokenize a batch of texts.
        Returns dict with input_ids, attention_mask (and token_type_ids if applicable).
        """
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        return encoding

    def encode(self, text: str, return_tensors: str = 'pt') -> dict:
        """
        Tokenize a single text string.
        Returns dict with input_ids, attention_mask (and token_type_ids if applicable).
        """
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=return_tensors,
        )
        return encoding

    def decode(self, token_ids) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


if __name__ == "__main__":
    tok = NLPTokenizer()
    sample = ["Stock market rallied after Fed announcement", "AI startup raises $100M in Series B"]
    encoded = tok.encode_batch(sample)
    logging.info(f"Input IDs shape: {encoded['input_ids'].shape}")
    logging.info(f"Decoded back: {tok.decode(encoded['input_ids'][0])}")
