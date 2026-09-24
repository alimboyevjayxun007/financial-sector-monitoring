"""Track B: training entry point."""
import pandas as pd

from src import config
from src.model import cross_validate, train


def load_train_features() -> pd.DataFrame:
    if not config.TRAIN_FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{config.TRAIN_FEATURES_PATH} not found — run `python3 -m src.features` "
            "first to generate it. (There used to be a silent fallback to random dummy "
            "features here for early-development bootstrapping; it was removed because "
            "training a real model on noise and saving it as model.pkl with no error is "
            "exactly the kind of silent-wrong-output failure a production pipeline must "
            "not allow — see README.md 'Production-readiness' notes.)"
        )
    # features.py already bakes TARGET_COL into this file (see build()),
    # so it's read as-is — no re-merge needed.
    return pd.read_parquet(config.TRAIN_FEATURES_PATH)


if __name__ == "__main__":
    df = load_train_features()
    X = df[config.FEATURE_COLUMNS]
    y = df[config.TARGET_COL]

    auc = cross_validate(X, y)
    print(f"CV ROC-AUC: {auc:.4f}")

    model = train(X, y)
    import joblib

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.OUTPUTS_DIR / "model.pkl")
    print(f"model saved to {config.OUTPUTS_DIR / 'model.pkl'}")
