
"""GSM-Symbolic: templated variants of GSM8K for testing mathematical reasoning.
 
Mirzadeh et al., 2024. https://arxiv.org/abs/2410.05229
 
Prompt, decoding, and answer extraction follow the authors' README:
https://github.com/apple/ml-gsm-symbolic
"""
 
import math
import re
from typing import Any
 
from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, prompt_template


DATASET_PATH = "apple/GSM-Symbolic"
DATASET_REVISION = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
VARIANTS = ("main", "p1", "p2")


# translates a dataset record into an inspect sample
def record_to_sample(record: dict[str, Any]) -> Sample:
    DELIM = '####'
    input = record['question']
    answer = record['answer'].split(DELIM)
    target = answer.pop().strip()
    reasoning=DELIM.join(answer).strip()
    id=f'{record['id']}_{record['instance']}'
    metadata = {'reasoning': reasoning,'original_question': record['original_question'], 'original_answer': record['original_answer'], 'original_id': record['original_id']}
    return Sample(
        id=id,
        input=input,
        target=target,
        metadata=metadata
    )
    


@task
def gsm_symbolic(
    fewshot: int = DEFAULT_FEWSHOT,
    epochs: int = DEFAULT_EPOCHS,
) -> Task:
    """Q&A evaluation with match-based scoring.

    Args:
        fewshot: Number of few-shot examples (0 to disable).
        epochs: Number of evaluation epochs.
    """
    solver = [prompt_template(PROMPT_TEMPLATE), generate()]
    if fewshot:
        solver.insert(0, system_message(FEWSHOT_PROMPT))

    return Task(
        dataset=[record_to_sample(r) for r in DATASET],
        solver=solver,
        scorer=match(),
        epochs=epochs,
    )
