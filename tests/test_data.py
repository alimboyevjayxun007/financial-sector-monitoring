import pandas as pd
import pytest

from src import config
from src.data import CorruptParquetError, load_signals, load_transactions


def test_frozen_category_schema():
    assert config.DIRECTIONS == ("kirim", "chiqim")
    assert config.TX_TYPES == ("karta", "bank_otkazmasi", "naqd", "xalqaro")
    assert config.SEED == 42


def test_load_signals_parses_dates(tmp_path):
    path = tmp_path / "signals.csv"
    path.write_text("signal_id,signal_sanasi,eskalatsiya\nSG_1,2025-03-04,1\n")
    frame = load_signals(path)
    assert frame["signal_sanasi"].dtype.kind == "M"
    assert frame.loc[0, "signal_id"] == "SG_1"


def test_load_transactions_reports_corrupt_file_with_actionable_message(tmp_path):
    path = tmp_path / "broken.parquet"
    path.write_bytes(b"PAR1" + b"\xef\xbf\xbd" * 64 + b"\x10\x00\x00\x00PAR1")
    with pytest.raises(CorruptParquetError) as excinfo:
        load_transactions(path)
    message = str(excinfo.value)
    assert "broken.parquet" in message
    assert "binary" in message.lower()
