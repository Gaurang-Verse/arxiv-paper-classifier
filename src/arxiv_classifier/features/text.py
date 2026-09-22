"""Text preparation for the classifier's input."""

import pandas as pd


def combine_title_abstract(df: pd.DataFrame) -> pd.Series:
    """Join title and abstract into a single normalised text field.

    arXiv titles and abstracts contain embedded newlines and runs of
    whitespace from their original formatting, so we collapse all whitespace
    to single spaces.
    """
    combined = df["title"].fillna("") + " " + df["abstract"].fillna("")
    return combined.str.replace(r"\s+", " ", regex=True).str.strip()