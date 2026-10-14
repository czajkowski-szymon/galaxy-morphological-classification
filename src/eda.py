"""Load Galaxy Zoo 2 label columns into pandas DataFrames for exploratory analysis."""

import pandas as pd

from src import data as schema


def question_votes(split) -> pd.DataFrame:
    """Total votes per question, one column per question; missing values set to 0."""
    df = split.select_columns(schema.total_vote_cols()).to_pandas().fillna(0)
    df.columns = schema.QUESTION_NAMES
    return df


def fractions(split) -> pd.DataFrame:
    """Vote fraction per answer; NaN marks questions the galaxy never reached."""
    df = split.select_columns(schema.campaign_cols(fraction=True)).to_pandas()
    df.columns = schema.ANSWER_KEYS
    return df
