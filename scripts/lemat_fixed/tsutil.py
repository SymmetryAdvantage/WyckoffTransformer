import re, pandas as pd
def lemat_time(s):
    m=re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?", str(s))
    return pd.Timestamp(m.group(0)).strftime("%Y-%m-%d %H:%M:%S") if m else None
