from pathlib import Path

import pandas as pd

TX_COLUMNS = [
    "signal_id",
    "tranzaksiya_vaqti",
    "kirim_chiqim",
    "tranzaksiya_turi",
    "miqdor_indeksi",
]


class CorruptParquetError(RuntimeError):
    """Raised when a parquet file cannot be deserialized."""


def load_signals(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["signal_sanasi"] = pd.to_datetime(frame["signal_sanasi"])
    return frame


def load_transactions(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path, columns=TX_COLUMNS)
    except Exception as exc:
        raise CorruptParquetError(
            f"{path.name} could not be read: {exc}. "
            f"The supplied copy was mangled by a text-mode transfer "
            f"(every byte above 0x7F replaced with U+FFFD) and cannot be repaired. "
            f"Download it again in binary mode, then verify with: "
            f"python -c \"import pyarrow.parquet as pq; "
            f"print(pq.ParquetFile('{path}').metadata.num_rows)\""
        ) from exc
    frame["tranzaksiya_vaqti"] = pd.to_datetime(frame["tranzaksiya_vaqti"])
    frame["miqdor_indeksi"] = pd.to_numeric(frame["miqdor_indeksi"])
    return frame
