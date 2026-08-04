import pandas as pd

import config


def append_flow(features):
    """Append one classified flow to the output CSV, writing a header if new."""
    df = pd.DataFrame([features])

    path = config.CSV_FILE
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        path,
        mode="a" if path.exists() else "w",
        header=not path.exists(),
        index=False,
    )
