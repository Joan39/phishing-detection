"""Does the hybrid rescue the rows model A scores below 0.10?

Model A's bottom decile on the held-out fold is 78.9% phishing (982 rows). This
scores the same rows through B, the deployed quantum artifact and the live
fusion so the damage that family actually causes is measured, not assumed.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from backend.quantum_live import QUANTUM_FEATURES, load_quantum_model
from backend.train_models import (
    DEFAULT_DATASET,
    load_dataset,
    make_baseline,
    make_ensemble,
    select_features,
)
from backend.url_features import extract_url_features_frame
from scripts.production_benchmark import (
    BOOSTING_ITERATIONS,
    FEATURE_PROFILE,
    SEED,
    TREES,
    fold_split,
    production_negative_rows,
)

frame, target, groups = load_dataset(DEFAULT_DATASET)
x_all = extract_url_features_frame(frame["URL"].astype(str).tolist())
cols = select_features(x_all, FEATURE_PROFILE)
train_index, test_index = fold_split(x_all, target, groups, 0, cols)

x_test = x_all.iloc[test_index][cols].reset_index(drop=True)
y_test = target.iloc[test_index].reset_index(drop=True)
urls = np.asarray(frame["URL"].astype(str).iloc[test_index].tolist())

negatives, _ = production_negative_rows(cols)
x_fit = pd.concat([x_all.iloc[train_index][cols], negatives], ignore_index=True)
y_fit = pd.concat(
    [target.iloc[train_index], pd.Series([0] * len(negatives), dtype=int)],
    ignore_index=True,
)

a = make_baseline(SEED).fit(x_fit, np.asarray(y_fit))
b = make_ensemble(SEED, trees=TREES, boosting_iterations=BOOSTING_ITERATIONS).fit(
    x_fit, np.asarray(y_fit)
)
pa = np.asarray(a.predict_proba(x_test)[:, 1], dtype=float)
pb = np.asarray(b.predict_proba(x_test)[:, 1], dtype=float)
artifact = load_quantum_model(Path("models/quantum_live.joblib"))
pq = np.asarray(
    artifact["classifier"]
    .predict_proba(artifact["scaler"].transform(x_test[QUANTUM_FEATURES]))[:, 1],
    dtype=float,
)
y = np.asarray(y_test)

classical = 0.35 * pa + 0.65 * pb
hybrid = 0.75 * classical + 0.25 * pq

family = pa < 0.10
print(
    f"A-bottom rows (p<0.10): n={int(family.sum())}, "
    f"phishing inside={int((family & (y == 1)).sum())} "
    f"({(family & (y == 1)).sum() / family.sum():.3f})"
)
for name, values in (("A", pa), ("B", pb), ("quantum", pq), ("classical", classical), ("hybrid", hybrid)):
    v = values[family]
    print(
        f"  {name:<9} mean={v.mean():.4f} median={np.median(v):.4f} "
        f"share>=0.40={np.mean(v >= 0.40):.4f}"
    )
missed = family & (y == 1) & (hybrid < 0.40)
total_missed = int(((y == 1) & (hybrid < 0.40)).sum())
print(f"hybrid misses inside the family: {int(missed.sum())}")
print(f"hybrid misses overall: {total_missed} -> family share of all misses: {missed.sum() / max(1, total_missed):.2%}")
hosts = pd.Series([urlparse(u).netloc for u in urls[missed]]).value_counts().head(12)
print("missed hosts:\n", hosts.to_string())
print("example missed URLs:")
for u in urls[missed][:6]:
    print("   ", u[:96])
