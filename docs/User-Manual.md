# QPhish User Manual

QPhish is a URL phishing-detection workspace. You submit a web address and the
system returns a model-backed verdict with a risk score, per-model breakdown,
and human-readable reasons. Submitted URLs are analyzed from their structure
only — QPhish never opens or fetches the websites you scan.

---

## 1. Getting access

### Create an account

1. Open the QPhish web app (development: `http://localhost:5173`).
2. Click **Create an account**.
3. Enter your full name, email, and a password of at least 8 characters.
4. If the deployment requires email verification, confirm through the link in
   your inbox before signing in.

### Sign in

Enter your email and password on the sign-in page. Your session stays active
until you sign out or the access token expires. Use the **Show / Hide** toggle
to check your password while typing.

### Forgot your password

Click **Forgot password?**, enter your email address, and follow the reset
link if an account exists. For privacy, the app always shows the same message
whether or not the address is registered.

---

## 2. Scanning a URL

1. From the **Dashboard** or **URL Scanner** view, type a URL into the
   *Website URL* field (for example `https://example.com/login`).
2. Click **Analyze URL**.
3. Only `http://` and `https://` addresses are accepted. If the URL is empty
   or malformed, a validation notice appears instead of a verdict.

### Reading the verdict

| Element | Meaning |
| --- | --- |
| **Potential phishing** | The hybrid score is at or above the decision threshold. Do not enter credentials. |
| **Likely legitimate** | The score is below the threshold. No strong phishing indicators matched. |
| **Risk percentage** | The model's phishing probability for this URL (0–100%). |
| **Decision threshold** | The current cut-off, shown in the report. It adapts over time from analyst feedback (see §4). |

### The analysis report

The right-hand panel explains the result in detail:

- **Verdict and recommendation** — ALLOW or BLOCK guidance.
- **Overall threat risk** — the final hybrid probability.
- **Model scores** — each detector's independent phishing probability:
  *Logistic regression (A)*, *Soft-voting ensemble (B)*, *Quantum VQC*, and the
  blended *Hybrid live score*.
- **Why this classification** — signal cards for HTTPS usage, IP-address
  hosts, obfuscation (`%40`-style encodings), URL length and subdomain depth,
  unusual special characters, and trusted-domain calibration.
- **Benchmark performance** — held-out accuracy of the classical models.

Trusted brands (Google, Wikipedia, GitHub, Apple, Amazon, Microsoft, YouTube)
get a bounded calibration: clean URLs on those domains have their probability
capped so ordinary site pages are not flagged. Lookalike domains such as
`paypal.com.login-verify-account.xyz` are **not** trusted and are scored
normally.

---

## 3. Detection History

The **Detection History** view lists every scan made from your account, newest
first, with the verdict and timestamp. The dashboard shows the four most
recent scans. History is stored in the backend database and is private to
your account.

---

## 4. Giving feedback (adaptive learning)

After any scan, the feedback panel asks **"Was this verdict correct?"**

1. Choose **Actually legitimate**, **Actually phishing**, or **Not sure** — or
   type free text such as *"it is actually phishing"*. The system extracts an
   explicit label from clear text and ignores ambiguous wording.
2. Click **Submit feedback**.

What happens next:

- Your label is stored in the feedback log.
- A reinforcement-learning bandit adjusts the **decision threshold** (between
  0.30 and 0.70) so future scans score better against analyst truth.
- The classical models are retrained with your labeled example mixed into the
  dataset, and the quantum model is retrained in the background. Retraining
  can take a minute or two; the app stays usable.

**Not sure** feedback is recorded but never changes model training.

The **Adaptive Learning** view shows the live threshold, the classical/quantum
blend weight, your feedback count, and the average reward the bandit has
earned.

---

## 5. Model Comparison

The **Model Comparison** view shows how each detector performed on held-out
domains (accuracy, phishing F1, ROC-AUC), the live hybrid formula, and the
quantum-weight × threshold-policy experiment matrix. Metrics come from saved
experiment reports, not from your scans.

---

## 6. Research Analytics

Your personal activity summary: total scans, phishing detections, average
phishing score, and feedback submitted — all read live from the backend.

---

## 7. Settings and About

**Settings** shows your account and a sign-out button. **About** summarizes
how detection works and the privacy stance (structure-only analysis).

---

## 8. Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| *"The backend is unavailable"* | The API is not running or `VITE_API_BASE_URL` points elsewhere. Start the backend (see the Development Setup guide). |
| Detection engine pill says *unavailable* | Model files are missing or unreadable on the server. Ask the operator to retrain or restore `models/`. |
| Verdict seems wrong | Submit feedback — the threshold adapts and the models retrain. |
| Scan of a normal site page looks risky | Check the report: if the domain is trusted but calibration did not apply, the URL likely contains obfuscation or unusual characters. |
| Sign-in fails after reset link | Verify the email address first if the deployment requires verification. |
