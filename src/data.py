# Galaxy Zoo 2 decision tree flattened to (question, answers) pairs.
# The order of pairs defines the layout of the 30-element output vector
QUESTIONS: list[tuple[str, list[str]]] = [
    ("smooth-or-featured", ["smooth", "featured-or-disk", "artifact"]),
    ("disk-edge-on", ["yes", "no"]),
    ("has-spiral-arms", ["yes", "no"]),
    ("bar", ["yes", "no"]),
    ("bulge-size", ["dominant", "obvious", "just-noticeable", "no"]),
    ("something-odd", ["yes", "no"]),
    ("how-rounded", ["round", "in-between", "cigar"]),
    ("bulge-shape", ["round", "boxy", "no-bulge"]),
    ("spiral-winding", ["tight", "medium", "loose"]),
    ("spiral-arm-count", ["1", "2", "3", "4", "more-than-4", "cant-tell"]),
]

CAMPAIGN = "gz2"

ANSWER_KEYS: list[str] = [
    f"{question}_{answer}" for question, answers in QUESTIONS for answer in answers
]
QUESTION_NAMES: list[str] = [question for question, _ in QUESTIONS]
N_OUTPUTS: int = len(ANSWER_KEYS)


# Function for building questions slices, each slice contains answers for specific question
def _build_question_slices() -> list[slice]:
    slices, start = [], 0
    for _, answers in QUESTIONS:
        slices.append(slice(start, start + len(answers)))
        start += len(answers)
    return slices


QUESTION_SLICES: list[slice] = _build_question_slices()


def campaign_cols(fraction: bool = False) -> list[str]:
    suffix = "_fraction" if fraction else ""
    return [
        f"{question}-{CAMPAIGN}_{answer}{suffix}"
        for question, answers in QUESTIONS
        for answer in answers
    ]


def total_vote_cols() -> list[str]:
    return [f"{question}-{CAMPAIGN}_total-votes" for question in QUESTION_NAMES]


def validate_columns(column_names: list[str]) -> None:
    available = set(column_names)
    expected = campaign_cols(fraction=True) + total_vote_cols()
    missing = [col for col in expected if col not in available]
    if missing:
        raise ValueError(
            f"{len(missing)} expected column(s) absent from the dataset, e.g. {missing[:5]}"
        )
