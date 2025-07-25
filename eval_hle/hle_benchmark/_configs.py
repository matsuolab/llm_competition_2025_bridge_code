from dataclasses import dataclass

@dataclass
class Config:
    dataset: str
    provider: str
    base_url: str
    model: str
    max_completion_tokens: int
    reasoning: bool
    num_workers: int
    max_samples: int
    judge: str