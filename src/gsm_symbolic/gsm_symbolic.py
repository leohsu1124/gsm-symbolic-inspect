"""GSM-Symbolic: templated variants of GSM8K for testing mathematical reasoning.

Mirzadeh et al., 2024. https://arxiv.org/abs/2410.05229

Prompt, decoding, and answer extraction follow the authors' README:
https://github.com/apple/ml-gsm-symbolic
"""

import statistics
from collections import defaultdict
from typing import Any
import re, math

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    ValueToFloat,
    accuracy,
    metric,
    scorer,
    stderr,
    value_to_float,
)
from inspect_ai.solver import TaskState, generate, prompt_template

DATASET_PATH = "apple/GSM-Symbolic"
DATASET_REVISION = "93b5b3758d9d9841ffe81d6cd2ae2b030685b078"
VARIANTS = ("main", "p1", "p2")

PROMPT_INSTRUCTION = """ As an expert problem solver, solve step by step the following mathematical questions."""

FEWSHOT_EXAMPLES: list[tuple[str, str, str]] = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
    "There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.", "6"), 
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?", "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.","5"), 
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?","Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.","39"), 
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?","Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.","8"),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?","Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.","9"),
    ("There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?","There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.","29"),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?","Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.","33"),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she have left?","Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.","8")]


ANSWER_CUE = re.compile(
    r"(?:the\s+)?final\s+answer(?:\s+to[^:.\n]*)?\s*(?:is|:)", re.IGNORECASE)

def build_prompt_template(fewshot: int) -> str:
    if not 0 <= fewshot <= len(FEWSHOT_EXAMPLES):
        raise ValueError(f"fewshot must be between 0 and {len(FEWSHOT_EXAMPLES)}")
    shots = [
        f"Q: {q}\nA: Let's think step by step. {reasoning} The final answer is {final}."
        for q, reasoning, final in FEWSHOT_EXAMPLES[:fewshot]
    ]

    body = "\n\n".join([PROMPT_INSTRUCTION, *shots]).replace("{", "{{").replace("}", "}}")
    return body + "\n\nQ: {prompt}\nA: Let's think step by step."

# translates a dataset record into an inspect sample
def record_to_sample(record: dict[str, Any]) -> Sample:
    DELIM = "####"
    sample_input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    sample_id = f"{record['id']}_{record['instance']}"
    metadata = {
        "template_id": record["id"],
        "instance": record["instance"],
        "original_id": record["original_id"],
    }
    return Sample(id=sample_id, input=sample_input, target=target, metadata=metadata)

def _numbers(text: str) -> list[float]:
    """Every number in `text`, in order (commas removed, as the authors do)."""
    return [float(n) for n in re.findall(r"-?\d+\.?\d*", text.replace(",", ""))]

def extract_final_answer(response: str, question: str | None = None) -> float | None:
    """Return the last number in a response, per the authors' heuristic.
 
    Returns None (instead of raising, as the reference code does) when the
    response contains no number.
    """
    # Drop anything after the model starts writing a new question.
    response = response.split("\nQ:")[0]
    # Markdown emphasis around the answer ("**$51**", "### Final Answer").
    response = re.sub(r"[*#]+", " ", response)
 
    numbers = _numbers(response)
    if not numbers:
        return None
 
    cues = list(ANSWER_CUE.finditer(response))
    if not cues:
        return numbers[-1]
 
    # Last cue: if the model states a conclusion twice, the final one stands.
    line_end = response.find("\n", cues[-1].end())
    segment = response[cues[-1].end() : line_end if line_end != -1 else len(response)]
    on_line = _numbers(segment)
    if not on_line:
        return numbers[-1]
 
    if question is not None:
        asked = set(_numbers(question))
        novel = [n for n in on_line if n not in asked]
        if novel:
            return novel[0]
        # Every candidate also appears in the question: the answer legitimately
        # restates a given quantity, so take the first one as stated.
 
    return on_line[0]

@metric
def per_instance_accuracy(to_float: ValueToFloat = value_to_float()) -> Metric:
    """Spread of accuracy across the dataset's generated instance sets."""

    def metric_fn(scores: list[SampleScore]) -> Value:
        by_instance: dict[Any, list[float]] = defaultdict(list)
        for sample_score in scores:
            metadata = sample_score.sample_metadata
            if metadata is None or "instance" not in metadata:
                raise ValueError(
                    f"Sample {sample_score.sample_id!r} has no 'instance' metadata; "
                    "per_instance_accuracy needs it to group samples into sets."
                )
            by_instance[metadata["instance"]].append(to_float(sample_score.score.value))

        if not by_instance:
            return {"instance_mean": 0.0, "instance_std": 0.0,
                    "instance_min": 0.0, "instance_max": 0.0}

        accuracies = [sum(values) / len(values) for values in by_instance.values()]
        return {
            "instance_mean": statistics.fmean(accuracies),
            # pstdev, not stdev: one value gives 0.0 rather than raising.
            "instance_std": statistics.pstdev(accuracies),
            "instance_min": min(accuracies),
            "instance_max": max(accuracies),
        }

    return metric_fn

 
@scorer(metrics=[accuracy(), stderr(cluster="template_id"), per_instance_accuracy()])
def gsm_symbolic_scorer() -> Scorer:
    """Score by numeric equality of the last number in the response."""
 
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        predicted = extract_final_answer(response, state.input_text)
        expected = float(target.text.replace(",", ""))
        correct = predicted is not None and math.isclose(
            predicted, expected, rel_tol=1e-6, abs_tol=1e-6
        )
        return Score(
            value=CORRECT if correct else INCORRECT,
            answer=None if predicted is None else str(predicted),
            explanation=response,
        )
 
    return score

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
 
    solver = [prompt_template(build_prompt_template(fewshot)), generate()]
 
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            revision=DATASET_REVISION,
            name=variant,
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=solver,
        scorer=gsm_symbolic_scorer(),
        config=GenerateConfig(temperature=0.0, top_p=1.0, stop_seqs=["\nQ:"]),
    )
