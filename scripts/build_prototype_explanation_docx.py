"""Build a presentation-ready explanation guide for the QPhish prototype."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "QPhish_Prototype_Explanation_and_Presentation_Script.docx"


def shade(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    properties.append(element)


def set_cell_text(cell, text: str, bold: bool = False, color: str | None = None) -> None:
    cell.text = ""
    paragraph = cell.paragraphs[0]
    run = paragraph.add_run(text)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_table(doc: Document, headers: list[str], rows: list[list[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    for cell, header in zip(table.rows[0].cells, headers):
        set_cell_text(cell, header, bold=True, color="FFFFFF")
        shade(cell, "123C4A")
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            set_cell_text(cell, value)
    doc.add_paragraph()


def heading(doc: Document, text: str, level: int = 1) -> None:
    doc.add_heading(text, level=level)


def say(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    run = paragraph.add_run("SAY THIS: ")
    run.bold = True
    run.font.color.rgb = RGBColor(0x0B, 0x7A, 0x7A)
    paragraph.add_run(text)


def show(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    run = paragraph.add_run("SHOW: ")
    run.bold = True
    paragraph.add_run(text)


def note(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    run = paragraph.add_run("PRESENTER NOTE: ")
    run.bold = True
    run.font.color.rgb = RGBColor(0x9A, 0x5B, 0x00)
    paragraph.add_run(text)


def bullets(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def numbered(doc: Document, items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Number")


def add_page_number(paragraph) -> None:
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    run._r.addnext(field)


def build() -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10.5)
    for style_name, size, color in [
        ("Title", 27, "123C4A"),
        ("Heading 1", 17, "123C4A"),
        ("Heading 2", 13, "0B7A7A"),
        ("Heading 3", 11, "123C4A"),
    ]:
        style = doc.styles[style_name]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True

    footer = section.footer.paragraphs[0]
    footer.text = "QPhish prototype explanation guide  |  "
    add_page_number(footer)

    title = doc.add_heading("QPhish Prototype", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("Exact presentation script and deep technical explanation").bold = True
    subtitle.add_run("\nPrepared from the implemented React frontend, FastAPI backend, model pipeline, and measured evaluation results.")
    doc.add_paragraph()

    doc.add_paragraph(
        "Purpose of this guide: use the sections marked “SAY THIS” as the words to speak "
        "during a demonstration or viva. Use “SHOW” to control the live prototype. The "
        "technical notes explain the concepts behind each statement so the explanation is "
        "accurate, confident, and not overstated."
    )
    add_table(doc, ["At a glance", "What to remember"], [
        ["Name", "QPhish — Quantum-Enhanced Adaptive Phishing Detection"],
        ["Input", "An HTTP or HTTPS URL, treated as text"],
        ["Output", "Phishing probability, verdict, explanation, threshold, and model scores"],
        ["Main technology", "React/Vite frontend, FastAPI backend, classical ML, 4-qubit VQC, adaptive policy"],
        ["Privacy boundary", "The submitted website is never opened, fetched, or rendered"],
        ["Main research idea", "Use a hybrid detector and analyst feedback to adapt decisions after deployment"],
    ])

    heading(doc, "1. The 60-second opening", 1)
    say(doc, "QPhish is an adaptive phishing-detection security workspace. A user submits a website address, and the system evaluates the structure of that URL without opening the website. It extracts measurable URL features, sends them through two classical machine-learning models and a four-qubit variational quantum classifier, combines their probabilities, applies safety checks and an adaptive decision threshold, and returns an explainable phishing or legitimate verdict. The user can then provide feedback, allowing the system to improve its future decision policy. The prototype is designed as a decision-support tool, not as the only security control.")
    show(doc, "The sign-in page, then the Dashboard, then the URL Scanner.")
    note(doc, "Do not begin with quantum computing. Begin with the problem, the safe URL-only boundary, and the user outcome. Introduce the quantum component only after the basic pipeline is understood.")

    heading(doc, "2. The problem being solved", 1)
    say(doc, "Phishing URLs are designed to look trustworthy while directing users to attacker-controlled infrastructure. They often contain structural warning signs such as excessive length, deep subdomains, IP-address hosts, missing HTTPS, encoded characters, suspicious query parameters, and unusual punctuation. The problem is not only detecting phishing once; it is maintaining reliable decisions when attackers change their URL patterns and when legitimate websites also use long, complicated URLs.")
    bullets(doc, [
        "A detector that flags too many legitimate links creates alert fatigue.",
        "A detector that misses new attack techniques creates false confidence.",
        "A dataset can make performance look better than real-world behaviour if its legitimate URLs are too simple.",
        "A deployed system needs explanations and a feedback path, not only a binary label.",
    ])
    say(doc, "QPhish addresses these problems by using URL morphology, multiple model perspectives, calibration rules, an adaptive threshold policy, and analyst feedback.")

    heading(doc, "3. What the prototype actually is", 1)
    say(doc, "This is a full-stack research prototype. The browser provides the security workspace, but the browser does not perform the detection itself. The React frontend sends authenticated requests to the FastAPI backend. The backend owns feature extraction, model inference, threshold selection, persistence, feedback processing, and metrics.")
    add_table(doc, ["Layer", "Role", "What to say"], [
        ["Frontend", "React 19 and Vite", "Provides authentication, scanning, reports, history, comparison, feedback, and analytics."],
        ["API", "FastAPI", "Validates requests, authenticates users, runs the scan pipeline, and returns structured results."],
        ["Features", "URL-only extraction", "Converts the URL string into 19 numeric structural features."],
        ["Models", "A, B, and VQC", "Produce independent phishing probabilities."],
        ["Policy", "Threshold and adaptation", "Converts probability into a verdict and learns from analyst labels."],
        ["Storage", "SQLite or PostgreSQL", "Stores users, scans, feedback, and policy state."],
    ])
    show(doc, "The sidebar navigation and the “Detection engine available” status pill.")

    heading(doc, "4. Exact live demonstration script", 1)
    heading(doc, "4.1 Authentication", 2)
    say(doc, "The prototype begins with an authenticated workspace. A user can create an account or sign in. Authentication is important because scan history and feedback belong to a specific analyst. The frontend stores an access token, and protected API requests send that token to the backend.")
    show(doc, "Create an account or sign in. Point out the password validation and the Forgot password option.")
    say(doc, "The application also supports email verification and password reset in deployments where those features are enabled. This is not just a visual login screen; authentication is connected to backend routes and protected data.")

    heading(doc, "4.2 Submitting a URL", 2)
    say(doc, "I will now submit a URL. The application accepts only HTTP and HTTPS addresses. The frontend validates that an address is present, and the backend validates the request before scoring it.")
    show(doc, "Enter a URL such as https://example.com/login and click Analyze URL.")
    say(doc, "The website is not opened. The URL is treated as a string. This is a deliberate privacy and safety design choice: the system can inspect suspicious address patterns without making a network request to the suspicious site.")

    heading(doc, "4.3 Reading the result", 2)
    say(doc, "The result contains more than a label. It shows the final threat probability, the verdict, the recommended action, the decision threshold, the individual model scores, and the URL signals that explain the result. If the probability is at or above the active threshold, the verdict is phishing; otherwise it is legitimate.")
    show(doc, "Point to the verdict, threat-risk percentage, model-score bars, decision threshold, and explanation cards.")
    say(doc, "A phishing verdict means the URL contains patterns associated with phishing and the recommended action is to block it or avoid entering credentials. A legitimate verdict means no strong phishing pattern was detected; it does not prove that the website is absolutely safe.")
    note(doc, "Always say “likely legitimate” or “no immediate risk detected,” not “guaranteed safe.” Machine learning produces evidence and probability, not certainty.")

    heading(doc, "4.4 Explaining the signal cards", 2)
    add_table(doc, ["Signal", "Exact explanation"], [
        ["Secure connection", "HTTPS is a positive transport signal, but HTTPS alone does not prove legitimacy."],
        ["Domain format", "An IP address as the host is a warning sign because phishing infrastructure often avoids a normal registered domain."],
        ["URL obfuscation", "Percent encoding or similar patterns can hide credentials, redirects, or suspicious content."],
        ["URL structure", "Long URLs and many subdomains can indicate complexity, although legitimate sites can also have them."],
        ["Special characters", "Unusual punctuation and query structure can be associated with generated phishing links."],
        ["Trusted-domain safeguard", "Clean URLs under configured trusted roots receive bounded calibration; lookalike domains do not."],
        ["Redirect payload", "A foreign redirect destination is scored so a trusted-looking wrapper cannot hide a suspicious target."],
        ["Adaptive decision cut", "The current threshold may be global or specific to a traffic class learned from feedback."],
    ])
    say(doc, "These cards are evidence summaries, not claims that one feature independently proves phishing. The final decision comes from the combined model score, calibration rules, and threshold.")

    heading(doc, "5. The detection pipeline in depth", 1)
    say(doc, "The complete pipeline is: URL string, feature extraction, classical model scoring, quantum model scoring, hybrid fusion, trusted-domain and redirect checks, adaptive threshold selection, verdict, persistence, and feedback observation.")
    add_table(doc, ["Stage", "What happens", "Why it matters"], [
        ["1. Validate", "Accept only a well-formed HTTP or HTTPS URL.", "Prevents malformed input from entering the scoring path."],
        ["2. Extract", "Calculate 19 structural features.", "Turns URL morphology into model-readable data."],
        ["3. Score A", "Logistic regression produces a probability.", "Provides a stable linear reference."],
        ["4. Score B", "Soft-voting ensemble produces a probability.", "Captures non-linear combinations of signals."],
        ["5. Score quantum", "A 4-qubit VQC produces a probability from four features.", "Adds a different hypothesis class for ensemble diversity."],
        ["6. Fuse", "Combine classical and quantum probabilities.", "Produces the hybrid risk score."],
        ["7. Calibrate", "Apply trusted-domain and redirect guardrails.", "Reduces known false positives and hidden redirect risk."],
        ["8. Decide", "Compare score with the active threshold.", "Converts probability into phishing or legitimate."],
        ["9. Explain and save", "Return reasons and store the scan.", "Makes the result auditable and visible in history."],
    ])

    heading(doc, "6. The 19 URL features", 1)
    say(doc, "The models do not read the webpage. They read numerical properties of the URL. The 19 features describe size, host structure, encoding, character composition, query shape, and transport.")
    add_table(doc, ["Feature group", "Examples", "Interpretation"], [
        ["Size", "URL length, domain length, TLD length", "Long or stuffed addresses can hide the real destination."],
        ["Host structure", "Subdomain count, IP-host flag, trusted-root flag", "Deep chains and raw IP hosts can indicate attacker infrastructure."],
        ["Encoding", "Obfuscation flag, encoded-character count and ratio", "Encoding can conceal credentials or redirect information."],
        ["Character mix", "Letters, digits, digit ratio", "Random-looking or digit-heavy tokens can be suspicious."],
        ["Query shape", "Equals, question marks, ampersands", "Large query structures can represent tracking, login, or redirect payloads."],
        ["Special characters", "Other special-character count and ratio", "Unusual punctuation can indicate generated or obfuscated links."],
        ["Transport", "HTTPS flag", "HTTPS is a useful signal but is not proof of legitimacy."],
    ])
    say(doc, "The key concept is URL morphology: the shape and structure of the address can reveal risk even before the page is visited.")

    heading(doc, "7. The three model components", 1)
    heading(doc, "7.1 Model A — logistic regression", 2)
    say(doc, "Model A is a logistic-regression pipeline. The features are imputed and standardized, and the model learns a linear decision boundary. Conceptually, it calculates a weighted combination of the features and maps that value to a probability using the logistic function.")
    say(doc, "Its strength is stability and interpretability. Its limitation is that a linear boundary may not capture interactions such as a URL being simultaneously long, hyphen-heavy, deeply nested, and non-HTTPS.")

    heading(doc, "7.2 Model B — soft-voting ensemble", 2)
    say(doc, "Model B combines logistic regression, a random forest, and gradient boosting. The random forest captures feature interactions through many decision trees. Gradient boosting builds a sequence of trees that correct earlier errors. Their probabilities are combined by soft voting.")
    say(doc, "Model B is the strongest classical detector in the measured comparison, so the deployed classical score gives it more influence than Model A.")

    heading(doc, "7.3 Quantum VQC", 2)
    say(doc, "The quantum component is a four-qubit variational quantum classifier, or VQC. Four selected URL features are scaled into angles. A ZZFeatureMap encodes and entangles those values, and a RealAmplitudes circuit adds trainable rotations. The circuit is optimized during training and measured over repeated shots to produce a phishing probability.")
    say(doc, "The correct claim is not that the quantum model is independently better. In the evaluation it is weaker on its own. Its role is ensemble diversity: it represents URL patterns differently from the classical models, so the hybrid system can use a second type of signal.")
    note(doc, "This honest explanation demonstrates deeper understanding. Do not claim “quantum advantage,” quantum hardware, or that the circuit replaces the classical ensemble. The live prototype uses a Qiskit-based simulator/artifact.")

    heading(doc, "8. Exact fusion mathematics", 1)
    say(doc, "The classical probability is a weighted combination of Model A and Model B. With the documented defaults, it is 35 percent Model A and 65 percent Model B. The hybrid probability is then 75 percent classical and 25 percent quantum.")
    doc.add_paragraph("P(classical) = 0.35 × P(A) + 0.65 × P(B)")
    doc.add_paragraph("P(hybrid) = 0.75 × P(classical) + 0.25 × P(quantum)")
    add_table(doc, ["Component", "Share of final score"], [
        ["Model A", "26.25% = 0.35 × 0.75"],
        ["Model B", "48.75% = 0.65 × 0.75"],
        ["Classical block", "75.00%"],
        ["Quantum VQC", "25.00%"],
    ])
    say(doc, "These are shares of the probability calculation, not four separate votes. The quantum component can move the score, but it cannot dominate the decision under the default policy.")

    heading(doc, "9. Adaptive learning and feedback", 1)
    say(doc, "After a scan, the analyst can say that the result was actually legitimate, actually phishing, or uncertain. Clear labels are stored. Uncertain feedback is recorded for audit purposes but is not treated as a training label.")
    say(doc, "The feedback updates an epsilon-greedy threshold bandit. The bandit chooses among bounded thresholds, observes the reward from analyst outcomes, and gradually prefers the threshold that performs better for the current traffic. The policy can also maintain context-specific thresholds, such as for HTTPS or trusted-root traffic.")
    say(doc, "The feedback path can trigger background retraining of the classical and quantum components. The request returns without blocking the interface, while the adaptation work is guarded so concurrent feedback does not launch uncontrolled overlapping retraining.")
    say(doc, "The system also uses disagreement and proximity to the threshold to recommend analyst review. A low-confidence case should be escalated rather than treated as a perfect automated decision.")
    show(doc, "Select “Actually phishing” or “Actually legitimate,” submit feedback, then open Adaptive Learning.")
    note(doc, "Explain that feedback improves future behaviour; it does not rewrite the already-issued verdict. Also explain that a few repeated labels are not enough to re-rank model weights because that would overfit.")

    heading(doc, "10. Security and privacy controls", 1)
    say(doc, "The prototype includes application security controls around the model. It uses authenticated API access, expiring bearer tokens, password hashing with a server-side secret, restricted CORS origins, trusted-host validation, optional HTTPS enforcement, API rate limiting, security response headers, and private per-user scan history.")
    say(doc, "The URL handling is also deliberately conservative. The system analyzes the submitted string and does not fetch the target website. Stored and displayed results redact query strings and fragments where appropriate. Prometheus metrics are intended for internal deployment traffic.")
    note(doc, "Do not describe the prototype as production-complete security infrastructure. Say that these controls establish a security-minded prototype posture and still require deployment configuration, monitoring, and operational review.")

    heading(doc, "11. Model Comparison and measured results", 1)
    say(doc, "The Model Comparison page reports domain-held-out evaluation rather than only training accuracy. Domain-held-out means that domains in the evaluation fold are kept separate from the domains used for training, which tests generalization to unseen domains.")
    add_table(doc, ["Component", "Accuracy", "Phishing F1", "ROC-AUC"], [
        ["Model A — logistic regression", "87.41%", "85.29%", "90.91%"],
        ["Model B — soft-voting ensemble", "98.48%", "98.21%", "99.65%"],
        ["Weighted classical ensemble", "97.65%", "97.25%", "99.37%"],
        ["Hybrid live model", "96.74%", "96.22%", "99.23%"],
    ])
    say(doc, "The hybrid score is not the highest held-out accuracy because the quantum term introduces a weaker but different signal. That trade-off is part of the research question. The project therefore reports both the classical strength and the hybrid behaviour instead of hiding the trade-off.")
    say(doc, "On curated real-world sets, the same stack maintained 100 percent phishing detection while legitimate accuracy was approximately 73 percent on one set and 83 percent on the never-tuned probe. This gap matters: real legitimate URLs are often more complex than the legitimate examples in a benchmark dataset.")

    heading(doc, "12. How to explain the limitations", 1)
    say(doc, "QPhish is a URL-only detector. It cannot see the page content, visual design, JavaScript, certificate details, reputation history, or what happens after a user interacts with the page. A malicious URL may look normal, and a legitimate article or tracking URL may look suspicious. Therefore, the system should support browser protection, email security, domain reputation, sandboxing, and human review rather than replace them.")
    bullets(doc, [
        "The quantum model is a research component and is not independently superior to the classical ensemble.",
        "Real-world legitimate traffic is harder than the held-out dataset.",
        "Feedback quality determines the quality of adaptation.",
        "The prototype depends on trained model artifacts and a running backend.",
        "A probability is not certainty; users must interpret the verdict as risk evidence.",
    ])

    heading(doc, "13. What to show on each prototype page", 1)
    add_table(doc, ["Page", "Show", "Explain"], [
        ["Dashboard", "Engine status and scan card", "The browser is connected to the backend detection engine."],
        ["URL Scanner", "URL input and result", "A URL is analyzed as text and returns a live model-backed verdict."],
        ["Analysis Report", "Risk, model scores, explanations", "The system exposes evidence instead of hiding the reasoning."],
        ["Detection History", "Recent scans", "Authenticated scans are persisted per user."],
        ["Model Comparison", "Metrics and fusion formula", "The system is evaluated with held-out domains and reports trade-offs."],
        ["Adaptive Learning", "Threshold, weights, feedback count", "Analyst labels influence future policy and retraining."],
        ["Research Analytics", "Scan and feedback totals", "Operational activity is measured through the backend."],
        ["About", "Privacy and architecture summary", "The application clearly states the URL-only boundary."],
    ])

    heading(doc, "14. Suggested 8-minute presentation order", 1)
    numbered(doc, [
        "Introduce the phishing problem and the risk of both missed attacks and excessive false positives.",
        "State the central safety property: QPhish analyzes URL structure and never opens the submitted site.",
        "Show authentication and explain why scan history and feedback are user-specific.",
        "Submit a URL and walk through the verdict, probability, threshold, model scores, and explanations.",
        "Show the pipeline: 19 features, Model A, Model B, quantum VQC, fusion, safeguards, threshold.",
        "Explain the exact weights: 35/65 inside the classical block and 75/25 at the hybrid layer.",
        "Submit analyst feedback and show how the adaptive threshold and feedback counters respond.",
        "Show model comparison results, then state the limitations and the role of QPhish as decision support.",
    ])

    heading(doc, "15. Questions you may be asked", 1)
    qa = [
        ("Why do you need quantum computing?", "The quantum model is included to test ensemble diversity, not to claim that it is independently more accurate. It uses a different hypothesis class and can contribute complementary information to the classical models."),
        ("Does the system visit the website?", "No. It analyzes the URL string only. This reduces exposure to malicious content and keeps the detection path focused on URL morphology."),
        ("Why is Model B weighted more than Model A?", "Model B performs better on the held-out comparison because its forest and boosting components capture non-linear feature interactions. Model A remains as a stable linear reference."),
        ("Why is the quantum weight only 25 percent?", "The weight was selected as a bounded research configuration that preserves a majority classical contribution while retaining quantum diversity. The circuit is not allowed to control the result by itself."),
        ("What does the threshold mean?", "It is the probability cut-off used to convert a continuous phishing score into a binary verdict. The policy can adapt it within a safe bounded range from analyst feedback."),
        ("What happens if the analyst is unsure?", "The feedback is recorded as uncertain but excluded from supervised retraining, because an unclear label should not teach the model the wrong target."),
        ("Can the system detect every phishing site?", "No. URL-only analysis cannot inspect page content or behaviour. It is a useful risk signal and should be combined with other controls."),
        ("Why are real-world results lower than benchmark results?", "Real legitimate URLs contain complex paths, tracking parameters, and redirects that may resemble phishing URLs. The dataset has a different distribution, so deployment evaluation is more realistic."),
        ("What is the main innovation?", "The integration of URL-only hybrid detection, quantum ensemble experimentation, explainable reporting, and feedback-driven adaptive decision policy in one working prototype."),
        ("What would you improve next?", "Add page-content and certificate signals in a sandbox, expand host-disjoint real-world evaluation, improve calibration, add stronger drift monitoring, and validate the adaptive policy with more independent analyst labels."),
    ]
    for question, answer in qa:
        heading(doc, question, 2)
        say(doc, answer)

    heading(doc, "16. Final closing statement", 1)
    say(doc, "In conclusion, QPhish demonstrates an end-to-end adaptive phishing-detection workflow. It accepts a URL, extracts structural signals, evaluates the address with classical and quantum model components, combines their probabilities, applies safety and calibration rules, and produces an explainable verdict. Its most important research feature is the feedback loop: analyst labels can influence future thresholds, model adaptation, and review decisions. The results show strong phishing detection, while the limitations show why URL-only analysis should support, not replace, a broader cybersecurity defence system.")

    heading(doc, "17. Technical vocabulary to use correctly", 1)
    add_table(doc, ["Term", "Correct meaning"], [
        ["Feature", "A measurable input derived from the URL, such as length or subdomain count."],
        ["Probability", "The model's estimated phishing likelihood, not a guarantee."],
        ["Threshold", "The cut-off that converts probability into a verdict."],
        ["False positive", "A legitimate URL incorrectly classified as phishing."],
        ["Recall", "The proportion of actual phishing URLs detected."],
        ["F1 score", "A balance between precision and recall."],
        ["ROC-AUC", "A ranking-quality measure across possible thresholds."],
        ["Domain-held-out", "Evaluation where test domains are not present in training."],
        ["VQC", "Variational quantum classifier: a trainable quantum circuit used for classification."],
        ["Concept drift", "A change in traffic or attack patterns that makes old decisions less reliable."],
        ["Adaptive policy", "The threshold and weighting logic that updates from feedback and traffic evidence."],
    ])

    heading(doc, "18. Sources inside the project", 1)
    bullets(doc, [
        "README.md — project purpose, hybrid formula, adaptation, cost, and research aims.",
        "docs/Architecture.md — system components, live scan pipeline, persistence, and security posture.",
        "docs/How-Detection-Works.md — the 19 features, model details, VQC, fusion, and adaptive policy.",
        "docs/User-Manual.md — user workflow, report interpretation, feedback, and troubleshooting.",
        "docs/Experiments-and-Results.md — held-out protocol, measured metrics, real-world evaluation, and limitations.",
        "backend/main.py — authenticated API, scan response, safeguards, persistence, and feedback endpoints.",
        "frontend/src/main.jsx — visible application flow, report, navigation, feedback, and comparison screens.",
    ])

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
