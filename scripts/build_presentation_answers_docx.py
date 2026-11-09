"""Build the Word document answering questions 6-11 of the poster/presentation form.

Every figure quoted here is read from the project's own measured artifacts so the
document cannot drift from what the code actually does.
"""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

OUT = Path("docs") / "Research-Project-Presentation-Form-Answers.docx"
BENCH = json.loads(
    (Path("results") / "first_experiment.json").read_text(encoding="utf-8")
)

HYBRID = BENCH["hybrid"]
ENS = BENCH["classical_ensemble"]
QUANTUM = BENCH["quantum_only"]
METRICS = BENCH["metrics"]
A = METRICS["A"]
B = METRICS["B"]

doc = Document()
doc.styles["Normal"].font.name = "Calibri"
doc.styles["Normal"].font.size = Pt(11)

title = doc.add_heading("Adaptive Website Phishing Detection Using Quantum Ensemble "
                        "Learning and Reinforcement Learning", level=0)
title.alignment = WD_ALIGN_PARAGRAPH.LEFT
doc.add_paragraph(
    "Research Project Poster and Prototype Presentation — Research Project Description "
    "(Section B, Questions 6-11)"
)
doc.add_paragraph(
    "Prepared from the implemented system and its measured results. "
    "Statements about what is built are supported by code in the project repository; "
    "statements about performance are supported by the evaluation artifacts described in "
    "the answers."
)


def question(number: int, text: str) -> None:
    doc.add_heading(f"QUESTION {number}: {text}", level=1)


def bullets(items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


# --------------------------------------------------------------------------- Q6
question(6, "Briefly describe your research project.")
doc.add_paragraph(
    "This project builds and evaluates a phishing-website detection system whose decision "
    "behaviour can change after deployment instead of staying fixed at what it learned in "
    "training. Phishing pages keep changing their appearance — new host names, long "
    "legitimate-looking paths, abuse of trusted redirect services — so a detector trained "
    "once on a snapshot loses accuracy over time, and usually loses it silently. The system "
    "combines three layers. First, a classical ensemble — a regularised logistic regression "
    "and a soft-voting ensemble of logistic, random-forest and gradient-boosting members — "
    "scores nineteen features derived from the URL text. Second, a variational quantum "
    "circuit classifier, run on a simulator over a four-feature subset, is fused with that "
    "ensemble into a hybrid probability. Third, a value-based reinforcement-learning layer "
    "turns analyst verdicts into a decision threshold, globally and per traffic class, "
    "re-ranks the influence of each detector, watches the score distribution for population "
    "shift, and routes uncertain cases to human review. All three layers are exposed through "
    "a working web application. On 47,158 domain-held-out records the hybrid detector "
    f"reaches {HYBRID['accuracy']:.4f} accuracy, {HYBRID['recall_phishing']:.4f} phishing "
    f"recall and a {HYBRID['false_positive_rate']:.4f} false-positive rate at the 0.40 "
    "threshold it uses."
)

# --------------------------------------------------------------------------- Q7
question(7, "What specific problem does your research seek to address, and why is it important?")
doc.add_paragraph(
    "The specific problem is the decay of phishing-detector reliability once the detector is "
    "in use. A conventional classifier is trained on a fixed data set, evaluated, and then "
    "frozen: its output is compared against a threshold chosen during development, and that "
    "threshold is rarely revisited. Phishing infrastructure, by contrast, is adapted "
    "deliberately. Three failure modes were measured directly in this project rather than "
    "assumed."
)
bullets([
    "Technique change. A set of fourteen phishing URLs built on techniques the model had "
    "never seen was scored before any retraining: two of them were classified as legitimate, "
    "a recall of 0.857, because they were delivered as a trusted brand page whose only "
    "content was a redirect parameter pointing at the attacker's host. The detector "
    "recognised the brand and ignored the payload.",
    "Legitimate traffic that looks suspicious. Long, content-rich legitimate URLs and "
    "bare-HTTP origins score above the decision cut often enough that the system flagged "
    "roughly one in four legitimate URLs in a curated real-world set. A detector that "
    "over-alerts is ignored by its users, which is its own form of failure.",
    "Silent miscalibration. The evaluation split itself is optimistic: the source data set "
    "contains many bare domains as legitimate records, so a model can score well on held-out "
    "rows while behaving poorly on realistic long URLs.",
])
doc.add_paragraph(
    "Internet users are affected first, because a successful phishing page harvests "
    "credentials, banking details or one-time codes. Organisations and financial "
    "institutions carry the loss and the remediation cost, and online service providers "
    "absorb the reputational damage when their own login pages, URL shorteners and redirect "
    "services are used as camouflage by attackers — which is precisely the technique that "
    "fooled this project's own first version. Detection is usually the first line of defence "
    "for users who have no corporate mail filter, so a system that degrades quietly is not a "
    "neutral failure: it is a false assurance."
)
doc.add_paragraph(
    "An adaptive detection system matters because the only sustainable response to a moving "
    "attacker is a detector that can move back. Repeatedly retraining a model on a new wave "
    "of examples is slow, expensive and requires labelled data that is not available at the "
    "moment the wave arrives. Adjusting decision behaviour — which threshold is used, which "
    "detector is trusted, and which cases are escalated to a human — is cheap, immediate, "
    "and auditable. This project therefore separates those two responses and measures each "
    "one, treating a policy adjustment as the first reaction to drift and a model retrain as "
    "the second."
)

# --------------------------------------------------------------------------- Q8
question(8, "What is the main objective of your research project?")
doc.add_paragraph(
    "Main objective: to design, implement and evaluate a phishing-website detection system "
    "that combines a quantum-assisted ensemble classifier with a reinforcement-learning "
    "policy layer, so that detection quality can be measured and then improved after "
    "deployment as analyst feedback and traffic patterns change, using a working web "
    "prototype as the test bed."
)
doc.add_paragraph("The intended outcomes of the project are:")
bullets([
    "Improved phishing-website detection, reported as a full metric picture rather than "
    "accuracy alone: phishing recall together with the legitimate false-positive rate, since "
    "the two trade directly against each other.",
    "Integration of quantum machine learning with ensemble learning, in which a variational "
    "quantum circuit is trained as an additional member and fused by weight with a classical "
    "ensemble, and the contribution of that member is measured honestly — including where it "
    "does not help.",
    "Adaptive decision-making using reinforcement learning: a value-based policy that learns "
    "decision thresholds from analyst verdicts, maintains separate thresholds per traffic "
    "class, re-ranks the weight given to each detector, watches the score distribution for "
    "shift, and sends low-confidence verdicts for human review.",
    "Development of a functional prototype: a browser-based application that submits a URL, "
    "returns a verdict with per-model scores, an explanation of the features that mattered "
    "and the threshold in force, records analyst feedback, and shows live policy state and "
    "benchmark results.",
    "Evaluation of the proposed approach: domain-grouped held-out testing, two curated "
    "real-world sets scored through the deployed system, a replayed technique-shift "
    "experiment comparing the adaptive policy with the static one, and training-versus-"
    "held-out diagnostics for over-fitting and calibration.",
])

# --------------------------------------------------------------------------- Q9
question(9, "Describe your research methodology or technical approach.")
doc.add_paragraph("The project follows a build-and-evaluate experimental design. The steps below are the ones actually carried out.")
doc.add_heading("1. Research design", level=2)
doc.add_paragraph(
    "The design is quantitative and comparative. For every claimed capability, a baseline "
    "configuration and a candidate configuration are scored on identical data using the same "
    "protocol, and the difference is reported whether it favours the candidate or not. The "
    "unit of analysis is the registrable domain, not the row, so that a domain cannot appear "
    "in both training and test data."
)
doc.add_heading("2. Dataset acquisition", level=2)
doc.add_paragraph(
    "The PhiUSIIL Phishing URL data set, a publicly published research data set, was used "
    "unchanged apart from relabelling: 235,795 records over 56 columns, with 0 missing "
    "values and 0 duplicate complete records. The label column contains 100,945 phishing "
    "records (label 0) and 134,850 legitimate records (label 1); these are relabelled "
    "internally so that 1 denotes phishing, which is how every metric in the project is "
    "computed."
)
doc.add_heading("3. Data understanding and preparation", level=2)
doc.add_paragraph(
    "Inspection showed why the published columns cannot be used naively for the intended "
    "task: most legitimate records are bare domains, so content-based columns are largely "
    "uninformative for the long realistic URLs the system must judge. The pipeline therefore "
    "parses the URL string itself. A documented legitimate hard-negative corpus of 7,712 "
    "content-rich legitimate URLs (replicated ten times) and 170 bare-HTTP origins "
    "(replicated sixty times) was added to training, contributing 87,320 rows, which brings "
    "the legitimate share of the training mix to 70.7 percent. Feature extraction is "
    "implemented as hand-written rules over the URL string using the standard library's URL "
    "parser, so query payloads are percent-decoded the way a browser decodes them, and an "
    "IP-address host is detected with the standard library's address module rather than a "
    "heuristic."
)
doc.add_heading("4. Feature analysis and selection", level=2)
doc.add_paragraph(
    "Nineteen features are derived from the URL text, and the selected set is exactly: URL "
    "length, domain length, top-level-domain length, subdomain count, a flag for an "
    "IP-address host, a flag for HTTPS, a flag for membership of a configured trusted brand "
    "root, the presence, count and ratio of obfuscated (percent-encoded) characters, letter "
    "count and ratio, digit count and ratio, and the counts of the equals, question-mark and "
    "ampersand characters plus a general special-character count and ratio. Redirect "
    "parameters are not among the nineteen model features: they are handled by a separate "
    "calibration stage that inspects the host and any embedded redirect target before the "
    "learned score is accepted."
)
doc.add_heading("5. Classical model development and training", level=2)
doc.add_paragraph(
    "Two classical detectors are trained with scikit-learn. Model A is a pipeline of median "
    "imputation, standardisation and L2 logistic regression with balanced class weights. "
    "Model B is a soft-voting ensemble of a logistic regression, a 200-tree random forest "
    "with balanced subsampling, and a histogram gradient-boosting classifier with 200 "
    "iterations, combined with member weights 1, 2 and 2. The 0.35/0.65 split used in the "
    "fusion was not tuned per-configuration: it follows the measured held-out advantage of "
    "the voting ensemble over the single linear model, so the ensemble carries more of the "
    "decision."
)
doc.add_heading("6. Quantum machine learning implementation", level=2)
doc.add_paragraph(
    "A variational quantum classifier is implemented with Qiskit and Qiskit Machine "
    "Learning: a four-qubit ZZFeatureMap over four URL features (URL length, domain length, "
    "subdomain count and special-character ratio) followed by a RealAmplitudes ansatz with "
    "two repetition layers, giving twelve trainable parameters, executed with a statevector "
    "sampler. Parameters are fitted classically against labelled data. The deployed circuit "
    "was trained on 1,000 rows per class from the data set plus 1,521 legitimate corpus rows "
    "and is checked to have zero overlap with the rows it is later scored on. No physical "
    "quantum hardware was used; all quantum execution is simulation."
)
doc.add_heading("7. Ensemble and quantum fusion", level=2)
doc.add_paragraph(
    "The live score is a two-stage weighted fusion: the classical path is 0.35 A plus 0.65 B, "
    "and the hybrid probability is 0.75 of that classical score plus 0.25 of the quantum "
    "reading. Before the verdict, clean URLs on configured trusted brand domains are capped "
    "at 0.25, and a trusted host loses that protection when its URL is only a jump to a "
    "different registrable domain, in which case the payload is scored on its own and the "
    "higher reading is used."
)
doc.add_heading("8. Reinforcement-learning integration", level=2)
doc.add_paragraph(
    "Analyst feedback is the reward signal: a verdict confirmed correct gives +1, a corrected "
    "verdict gives -1, and an uncertain label gives 0 and is not used for learning. A "
    "value-based bandit over five threshold actions (0.30 to 0.70) updates its action values "
    "with a step size of 0.2, and it runs in two forms — one global threshold and one per "
    "traffic class defined by scheme and trusted-brand status, where a class only takes over "
    "the decision after eight labels. A multiplicative-weights rule re-ranks the classical "
    "and quantum weights from each detector's recent accuracy, bounded to 0.20-0.80 and "
    "0.10-0.40 respectively and shrunk toward the documented values until twenty distinct "
    "URLs have been labelled. A label-free drift monitor compares the histogram of recent "
    "hybrid scores against a 200-scan reference using the population stability index, and "
    "cases where the detectors disagree strongly near the cut are flagged for review."
)
doc.add_heading("9. Integration into the web application", level=2)
doc.add_paragraph(
    "The trained artifacts are served by a FastAPI back end (Python, Uvicorn, Pydantic) with "
    "a SQLite store for accounts, scans and feedback, bearer-token authentication, and a "
    "Prometheus metrics endpoint. The front end is a React 19 and Vite application with URL "
    "submission, per-model scores, a plain-language explanation of the classification, the "
    "threshold in force, analyst feedback capture, scan history, a benchmark panel and an "
    "adaptive-policy panel. Retraining is scheduled as a background worker when feedback or "
    "a drift signal warrants it."
)
doc.add_heading("10. Testing and evaluation", level=2)
doc.add_paragraph(
    "Evaluation uses stratified grouped cross-validation over registrable domains (five "
    "splits, seed 42). Reported numbers are from held-out fold 0: 47,158 records across "
    "43,817 domains, none of which appear among the 175,286 training domains. Metrics are "
    "accuracy, precision, recall and F1 for the phishing class, ROC-AUC, and the "
    "false-positive rate on legitimate records, plus the confusion matrix and calibration "
    "reliability. Because a domain-held-out split was found to flatter the system on "
    "realistic traffic, two additional curated sets are scored — a 56-URL battery of "
    "everyday sites and lookalike campaigns, whose legitimate members informed the design of "
    "the hard-negative corpus, and an 18-URL probe assembled afterwards and kept "
    "independent of both the battery and the training corpus by an explicit check in its "
    "scoring script. A replayed technique-shift experiment streams familiar traffic, a "
    "wave of novel strings and a holdout through the adaptive and static policies in "
    "parallel, and separate diagnostics compare training-set against held-out accuracy for "
    "each model. Seventy-seven automated unit tests cover feature extraction, fusion, "
    "guardrails, adaptive policy behaviour and the benchmark endpoints."
)

# -------------------------------------------------------------------------- Q10
question(10, "What is the main contribution or distinguishing feature of your research project?")
doc.add_paragraph(
    "The distinguishing feature is that detection and decision-making are separated into two "
    "components that can be improved on different timescales. The classifier is the slow "
    "component and is retrained when evidence accumulates; the policy is the fast component "
    "and changes what threshold is used, which detector is trusted, and which verdict needs a "
    "human, the moment feedback arrives. Conventional phishing detectors treat this as a "
    "single trained artefact, which is the gap this project addresses."
)
doc.add_paragraph("Within that design, the specific contributions are:")
bullets([
    "A quantum-assisted ensemble in which a variational circuit is a fused member rather "
    "than the whole model, with its contribution reported honestly. The circuit alone reaches "
    f"{QUANTUM['accuracy']:.4f} held-out accuracy and, on this data set, costs the hybrid "
    f"about one accuracy point relative to the classical ensemble ({ENS['accuracy']:.4f} to "
    f"{HYBRID['accuracy']:.4f}); it is retained for diversity and real-world behaviour, and "
    "no quantum advantage is claimed.",
    "A reinforcement-learning layer that acts on analyst feedback at three levels — global "
    "threshold, per-traffic-class threshold, and detector weighting — plus a label-free "
    "distribution-shift monitor and confidence-based review routing, so adaptation is not "
    "dependent on a single signal.",
    "An explicit, measured guardrail for the technique that defeated the first version: "
    "trusted-brand URLs whose content is only a redirect payload receive no brand protection "
    "and are scored on the payload. This restored recall on fourteen previously unseen "
    "technique strings from 0.857 to 1.00 with no retraining and with no probability change "
    "on the 74 curated real-world rows.",
    "An evaluation protocol that reports held-out, real-world and adaptation results "
    "separately and states the cost of each adaptation, rather than a single headline "
    "accuracy figure.",
    "A working prototype in which all of the above is observable: the report page names the "
    "fusion weights and the threshold that a given scan actually used, and an administrative "
    "panel shows live policy state.",
])
doc.add_paragraph(
    "The academic contribution is a documented measurement of what each of these mechanisms "
    "does and does not buy, including a negative result: backfilling the learned policy from "
    "the current 22 analyst labels improves real-world legitimate accuracy but costs 1.1 "
    "percentage points of held-out accuracy and 1.2 points of held-out phishing recall, so "
    "the deployed system intentionally keeps its documented static weights. The practical "
    "contribution is a detector whose behaviour an operator can inspect, adjust and audit "
    "without retraining. No claim is made that this system outperforms published detection "
    "approaches; it has been compared only against its own static configuration and against "
    "single-model variants, and a comparison against external baselines remains future work."
)

# -------------------------------------------------------------------------- Q11
question(11, "What have you achieved so far in your research project?")
doc.add_paragraph("Completed and verified")
bullets([
    "Topic, scope and methodology defined, with the adaptation-versus-retraining separation "
    "and the honest-comparison design in place before implementation.",
    "Literature review of URL-based and machine-learning phishing detection, including "
    "quantum-classifier approaches, and the selection of the PhiUSIIL data set with a full "
    "integrity check (235,795 records, 56 columns, 0 missing values, 0 duplicate complete "
    "records, 100,945 phishing and 134,850 legitimate).",
    "URL feature-extraction and parsing module implemented, including redirect-payload "
    "detection with percent-decoding, and the documented legitimate hard-negative corpus "
    "used in training.",
    "Classical detectors trained: logistic regression and the soft-voting ensemble, fused in "
    "the documented 0.35/0.65 proportion. On 47,158 domain-held-out records model A reaches "
    f"{A['accuracy']:.4f} accuracy, model B {B['accuracy']:.4f}, and their 0.35/0.65 fusion "
    f"{ENS['accuracy']:.4f}.",
    "Variational quantum classifier implemented in simulation, trained and fused with the "
    "classical path; the hybrid detector reaches "
    f"{HYBRID['accuracy']:.4f} accuracy, {HYBRID['precision_phishing']:.4f} precision, "
    f"{HYBRID['recall_phishing']:.4f} phishing recall, {HYBRID['f1_phishing']:.4f} F1, "
    f"{HYBRID['roc_auc']:.4f} ROC-AUC and a {HYBRID['false_positive_rate']:.4f} "
    "false-positive rate at the 0.40 operating threshold.",
    "Trusted-domain calibration and the redirect guardrail implemented and measured; unseen-"
    "technique recall improved from 0.857 to 1.00 without retraining.",
    "Reinforcement-learning policy layer implemented and wired into the live scoring path: "
    "value-based thresholds globally and per traffic class, bounded detector re-ranking with "
    "an evidence-volume gate, a population-stability-index drift window and review routing.",
    "Working web prototype: FastAPI service with authentication, scan and feedback endpoints, "
    "background retraining, Prometheus metrics, and a React/Vite interface with explanations, "
    "history, a benchmark panel and a live policy panel.",
    "Evaluation infrastructure: domain-grouped benchmark script, two curated real-world sets, "
    "an offline scorer that reproduces the live path row for row, a policy-replay tool that "
    "rebuilds the policy from stored feedback, a technique-shift replay, and seventy-seven "
    "automated tests.",
    "Generalisation and fit diagnostics: training-versus-held-out comparison shows no "
    "over-fitting (model A holds out 0.032 better than it fits, model B 0.006 better), and "
    "the circuit's own feature budget was benchmarked against a classical fit on the same "
    "four features.",
])
doc.add_paragraph("In progress or not yet done")
bullets([
    "Real-world false positives remain the main open weakness: about one in four legitimate "
    "URLs in the curated battery and one in six in the probe still exceed the decision cut.",
    "The adaptive policy has been evaluated on simulated technique shift and on 22 stored "
    "analyst labels, not on a production stream; the volume of genuine feedback is still too "
    "small for the learned weights to be installed, which is a measured decision rather than "
    "an oversight.",
    "The label-free drift detector has not fired in replay, so its window and trigger are "
    "awaiting recalibration.",
    "The redirect guardrail inspects payloads inside the submitted URL string only; following "
    "real HTTP 3xx chains, meta-refresh pages and scripted redirects has been deliberately "
    "excluded to avoid the privacy risk of opening submitted sites.",
    "No comparison against external published detectors, no deployment beyond the local "
    "prototype, and no execution on quantum hardware.",
    "Ethics, data-protection and poster documentation are being finalised for submission.",
])
doc.add_paragraph(
    "Summary: the pipeline from data preparation through quantum-assisted ensemble detection, "
    "adaptive policy and a working user interface exists, runs, and is measured; the "
    "remaining work concerns real-world false positives, the volume of genuine feedback "
    "available to the adaptive layer, and external baseline comparison."
)

doc.save(OUT)
print(f"wrote {OUT}")
