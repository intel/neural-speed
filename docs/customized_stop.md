You can customize the stopping criteria according to your own needs by processing the input_ids to determine if text generation needs to be stopped.
Here is a simple example, which requires a minimum generation length of 80 tokens. Once the `min_length` is met, encountering a terminator `eos_token_id` will end the generation.

```python
import torch
from typing import List
from transformers import StoppingCriteria, StoppingCriteriaList

class StopOnTokens(StoppingCriteria):
    def __init__(self, min_length: int, start_length: int, stop_token_id: List[int]):
        self.min_length = min_length
        self.start_length = start_length
        self.stop_token_id = stop_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids.shape[-1] - self.start_length > self.min_length:
            for stop_id in self.stop_token_id:
                if input_ids[0][input_ids.shape[-1] - 1] == stop_id:
                    return True
        return False

stopping_criteria = StoppingCriteriaList(
    [
        StopOnTokens(
            min_length=80,
            start_length=inputs.shape[1],
            stop_token_id=[tokenizer.eos_token_id],
        )
    ]
)

outputs = model.generate(inputs, streamer=streamer, stopping_criteria=stopping_criteria)
```
