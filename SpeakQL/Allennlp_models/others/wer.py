from collections import Counter
import math
from typing import Iterable, Tuple, List, Dict, Set, Optional

import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric
# from allennlp.nn.util import dist_reduce_sum

import editdistance


@Metric.register("wer")
class WordErrorRate(Metric):
    """
    WER

    (Modified from allennlp BLEU)    
    """

    def __init__(
        self,
        exclude_tokens: Set[str] = None,
    ) -> None:
        self._exclude_tokens = set(exclude_tokens) or set()
        # self._precision_matches: Dict[int, int] = Counter()
        # self._precision_totals: Dict[int, int] = Counter()
        # self._prediction_lengths = 0
        # self._reference_lengths = 0
        self._wer_numer = 0
        self._wer_denom = 0

    def reset(self) -> None:
        self._wer_numer = 0
        self._wer_denom = 0

    def __call__(
        self,  # type: ignore
        predictions: List[List],
        gold_targets: List[List],
        mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        """
        Update precision counts.

        # Parameters

        predictions : `List[List]`, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : `List[List]`, required
            Batched reference (gold) translations with shape `(batch_size, max_gold_sequence_length)`.

        # Returns

        None
        """
        if mask is not None:
            raise NotImplementedError("This metric does not support a mask.")

        predictions, gold_targets = self.detach_tensors(predictions, gold_targets)
        for p, g in zip(predictions, gold_targets):
            _p = [str(t) for t in p if str(t) not in self._exclude_tokens]
            _g = [str(t) for t in g if str(t) not in self._exclude_tokens]

            n_ops = editdistance.eval(_p, _g)
            self._wer_numer += n_ops
            self._wer_denom += len(_g)


    def get_metric(self, reset: bool = False) -> Dict[str, float]:

        wer = 1.0 * self._wer_numer / self._wer_denom

        if reset:
            self.reset()
        return {"WER": wer}
