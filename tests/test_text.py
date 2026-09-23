"""Title + abstract normalisation."""

import pandas as pd

from arxiv_classifier.features.text import combine_title_abstract


def test_combine_joins_title_and_abstract():
    df = pd.DataFrame({
        "title": ["A Study of Foo"],
        "abstract": ["We study foo in detail."],
    })
    result = combine_title_abstract(df)
    assert result.iloc[0] == "A Study of Foo We study foo in detail."


def test_combine_collapses_embedded_whitespace_and_newlines():
    df = pd.DataFrame({
        "title": ["Title\nwith\nnewlines"],
        "abstract": ["Abstract   with    extra   spaces"],
    })
    result = combine_title_abstract(df)
    assert "\n" not in result.iloc[0]
    assert "  " not in result.iloc[0]


def test_combine_handles_missing_values():
    df = pd.DataFrame({
        "title": ["Only Title", None],
        "abstract": [None, "Only abstract"],
    })
    result = combine_title_abstract(df)
    assert result.iloc[0] == "Only Title"
    assert result.iloc[1] == "Only abstract"