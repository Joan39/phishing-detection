# Documentation

| Guide | Audience | Contents |
| --- | --- | --- |
| [User Manual](User-Manual.md) | Everyone | Accounts, scanning, reading verdicts, feedback, all workspace views, troubleshooting |
| [How Detection Works](How-Detection-Works.md) | Everyone | The 19 URL features, what each model learns, quantum circuit mechanics, fusion, calibration, adaptive threshold, worked examples |
| [Architecture](Architecture.md) | Engineers, researchers | System overview, live scan pipeline, adaptive learning loop, data conventions, security posture |
| [API Reference](API-Reference.md) | Integrators | Every endpoint with request/response examples and error semantics |
| [Development Setup](Development-Setup.md) | Contributors | Environment, commands, CLIs, model artifacts, repository layout, conventions |
| [Experiments and Results](Experiments-and-Results.md) | Researchers | Experiment catalog, classical/quantum metrics, quantum × RL configuration matrix, limitations |
| [Deployment](Deployment.md) | Operators | Docker Compose stack, TLS, preflight checks, production notes |

The generated Word report —
[QPhish System Architecture and Experimental Results](../reports/QPhish_System_Architecture_and_Experimental_Results_Report.docx)
— consolidates the same experimental findings in a single document and can be
rebuilt with `python scripts/generate_system_report.py`.
