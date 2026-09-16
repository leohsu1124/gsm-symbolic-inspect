"""GSM-Symbolic: templated variants of GSM8K for testing mathematical reasoning.

Mirzadeh et al., 2024. https://arxiv.org/abs/2410.05229

Prompt, decoding, and answer extraction follow the authors' README:
https://github.com/apple/ml-gsm-symbolic
"""

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import (
    match,
)
from inspect_ai.solver import generate

DATASET_PATH = "apple/GSM-Symbolic"
DATASET_REVISION = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
VARIANTS = ("main", "p1", "p2")


# translates a dataset record into an inspect sample
def record_to_sample(record: dict[str, Any]) -> Sample:
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    id = f"{record['id']}_{record['instance']}"
    metadata = {
        "template_id": record["id"],
        "instance": record["instance"],
        "original_id": record["original_id"],
    }
    return Sample(id=id, input=input, target=target, metadata=metadata)


@task
def gsm_symbolic(
    variant: str = "main",
    fewshot: int = 8,
) -> Task:
    """Q&A evaluation with match-based scoring.

    Args:
        variant: either main, p1, or p2
        fewshot: original paper uses fewshot of 8
    """
    # error check
    if variant not in VARIANTS:
        raise ValueError(f"Invalid variant {variant!r}, must be one of {VARIANTS}")

    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            revision=DATASET_REVISION,
            name=variant,
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=generate(),
        scorer=match(),
    )
