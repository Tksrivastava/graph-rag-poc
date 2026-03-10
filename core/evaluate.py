import re
import numpy as np
import sentencepiece
from typing import List
from pathlib import Path
from collections import Counter
from core.logging import LoggerFactory

logger = LoggerFactory().get_logger(__name__)


class EvaluateTokenize:
    def __init__(self, model_path: Path, input_text_list: List[str]):
        self.model = sentencepiece.SentencePieceProcessor(model_file=model_path)
        self.input_text_list = input_text_list
        logger.info("Tokenizer model loaded")

    def unk_token_ratio(self):
        total_tokens, unk_tokens = 0, 0
        for text in self.input_text_list:
            ids = self.model.encode(text, out_type=int)
            total_tokens += len(ids)
            unk_tokens += ids.count(0)
        logger.info("Unknown token rate calculated")
        return unk_tokens / total_tokens

    def fragmentation_rate(self):
        total_words, total_tokens = 0, 0
        for text in self.input_text_list:
            words = re.findall(r"\w+", text)
            total_words += len(words)
            total_tokens += len(self.model.encode(text))
        logger.info("Fragmentation rate calculated")
        return total_tokens / total_words

    def compression_ratio(self):
        ratio = []
        for text in self.input_text_list:
            ratio.append(len(self.model.encode(text)) / len(text))
        logger.info("Compression ratio calculated")
        return np.mean(ratio)

    def vocab_utilization(self):
        counter = Counter()
        for text in self.input_text_list:
            counter.update(self.model.encode(text))
        logger.info("Vocabulary utilization calculated")
        return len(counter)
