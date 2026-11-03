# QPhish backend

The Python API and model/feature-extraction modules live in this backend
folder. `main.py` provides the FastAPI integration
for authentication demo flows, health checks, and URL scanning.

Recommended first API routes:

- `POST /api/auth/login`
- `POST /api/auth/register`
- `POST /api/auth/logout`
- `GET /api/auth/me`
- `POST /api/scans`
- `GET /api/scans`
- `GET /api/models/comparison`
- `POST /api/feedback`

Keep model files, datasets, and credentials on the server. Do not expose them
to the browser.

Run from the project root:

```powershell
pip install -r backend/requirements.txt
uvicorn backend.main:app --reload
```

The API uses the saved models in `models/url/` and keeps the model's training
convention that original label `0` represents phishing.
Production retraining adds canonical pages from configured trusted domains as
legitimate hard negatives. Live scoring also applies a bounded trusted-domain
calibration only when the URL has no IP, obfuscation, or unusual special
characters; lookalike domains remain model-scored.

`/api/models/comparison` reads the saved classical and quantum experiment
reports from `results/`. Live URL scoring uses the saved quantum-enhanced
ensemble. `/api/feedback` persists
analyst labels in `results/feedback.jsonl`, updates the adaptive threshold in
`results/rl_state.json`, and retrains the live A+B models with the original
dataset plus all non-uncertain feedback. Retraining is synchronous and should
be moved to a job queue before production deployment.

Feedback can be submitted with a button label or typed text such as
`"it is actually phishing"` or `"this is legitimate"`. The API extracts an
explicit label from typed feedback, rejects conflicting labels, and leaves
unclear text as uncertain so it cannot affect model training.

The live adaptive threshold is intentionally conservative. Threshold
calibration should use held-out predictions, not training-set predictions:

```python
from backend.threshold_calibration import (
    evaluate_threshold,
    select_zero_false_positive_threshold,
)

threshold = select_zero_false_positive_threshold(probabilities, labels)
metrics = evaluate_threshold(probabilities, labels, threshold)
```

The selected threshold removes false positives observed in the validation
sample, but cannot guarantee perfect classification for unseen URLs.
