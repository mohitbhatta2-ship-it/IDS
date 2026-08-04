import os
import pandas as pd

CSV_FILE = "live_flows.csv"


def append_flow(features):

    df = pd.DataFrame([features])

    if not os.path.exists(CSV_FILE):

        df.to_csv(
            CSV_FILE,
            index=False
        )

    else:

        df.to_csv(
            CSV_FILE,
            mode="a",
            index=False,
            header=False
        )