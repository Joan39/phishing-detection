import React, { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";
import { apiRequest, clearAccessToken, getAccessToken } from "./api";

const navItems = [
  ["⌂", "Dashboard"],
  ["⌕", "URL Scanner"],
  ["◷", "Detection History"],
  ["◈", "Model Comparison"],
  ["♧", "Adaptive Learning"],
  ["⌁", "Research Analytics"],
];

function AuthPage({ mode, onAuthenticated, onSwitch }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [resetMessage, setResetMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);

  async function requestReset() {
    if (!email) {
      setError("Enter your email address first.");
      return;
    }
    setError("");
    try {
      await apiRequest("/api/auth/password-reset/request", {
        method: "POST",
        body: JSON.stringify({ email }),
      });
      setResetMessage("If an account exists, a password-reset email has been sent.");
    } catch (requestError) {
      setError(requestError.message);
    }
  }

  async function submit(event) {
    event.preventDefault();
    if (!email || !password || (mode === "signup" && !name)) {
      setError("Complete all required fields to continue.");
      return;
    }

    if (password.length < 8) {
      setError("Use a password with at least 8 characters.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const user = await apiRequest(`/api/auth/${mode === "login" ? "login" : "register"}`, {
        method: "POST",
        body: JSON.stringify({ email, password, name: name || undefined }),
      });
      if (mode === "signup") {
        setPassword("");
        setName("");
        setShowPassword(false);
        setResetMessage(
          user.verification_required
            ? "Account created. Check your email, then sign in."
            : "Account created. Please sign in to continue.",
        );
        onSwitch("login");
        return;
      }
      if (!user.access_token) {
        setResetMessage("Sign-in requires account verification.");
        return;
      }
      localStorage.setItem("qphish_access_token", user.access_token);
      onAuthenticated(user.user);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-layout">
      <section className="auth-brand">
        <div className="brand-mark">◈</div>
        <p className="eyebrow">SECURITY WORKSPACE</p>
        <h1>See risk clearly.<br /><span>Act before compromise.</span></h1>
        <p className="auth-copy">
          QPhish turns URL intelligence into a clear, actionable security
          decision for every investigation.
        </p>
        <div className="trust-row"><span>◉</span> Backend-ready detection engine</div>
        <div className="trust-row"><span>◉</span> Privacy-first URL handling</div>
      </section>
      <section className="auth-card-wrap">
        <div className="auth-card">
          <div className="brand compact"><div className="brand-mark">◈</div><strong>QPhish</strong></div>
          <div className="auth-heading">
            <p className="eyebrow">{mode === "login" ? "WELCOME BACK" : "GET STARTED"}</p>
            <h2>{mode === "login" ? "Sign in to your workspace" : "Create your workspace"}</h2>
            <p>{mode === "login" ? "Continue your threat intelligence work." : "Start monitoring website risk with QPhish."}</p>
          </div>
          <form onSubmit={submit}>
            {mode === "signup" && (
              <label>Full name<input value={name} onChange={(e) => setName(e.target.value)} autoComplete="name" /></label>
            )}
            <label>Email<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="email" /></label>
            <label>Password<div className="password-input"><input type={showPassword ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)} autoComplete={mode === "login" ? "current-password" : "new-password"} /><button type="button" className="password-toggle" onClick={() => setShowPassword((visible) => !visible)} aria-label={showPassword ? "Hide password" : "Show password"} aria-pressed={showPassword}>{showPassword ? "Hide" : "Show"}</button></div></label>
            {mode === "login" && <button type="button" className="text-button forgot" onClick={requestReset}>Forgot password?</button>}
            {error && <div className="form-error">{error}</div>}
            {resetMessage && <div className="inline-alert">{resetMessage}</div>}
            <button className="primary full" type="submit" disabled={loading}>{loading ? "Connecting…" : mode === "login" ? "Sign in" : "Create account"} <span>→</span></button>
          </form>
          <div className="divider"><span>or</span></div>
          <p className="auth-note">Accounts and scan history are stored securely by the backend.</p>
          <p className="switch-copy">
            {mode === "login" ? "New to QPhish?" : "Already have an account?"}
            <button className="text-button" onClick={() => onSwitch()}>{mode === "login" ? "Create an account" : "Sign in"}</button>
          </p>
        </div>
      </section>
    </main>
  );
}

function Dashboard({ user, onSignOut }) {
  const [active, setActive] = useState("Dashboard");
  const [url, setUrl] = useState("");
  const [scanState, setScanState] = useState("idle");
  const [scanProbability, setScanProbability] = useState(null);
  const [scanResult, setScanResult] = useState(null);
  const [scans, setScans] = useState([]);
  const [engineAvailable, setEngineAvailable] = useState(false);
  const [scanError, setScanError] = useState("");
  const [comparison, setComparison] = useState(null);
  const [feedback, setFeedback] = useState({ verdict: "", comment: "", status: "" });

  useEffect(() => {
    apiRequest("/api/health")
      .then((health) => setEngineAvailable(health.detection_engine === "available"))
      .catch(() => setEngineAvailable(false));
    apiRequest("/api/models/comparison")
      .then(setComparison)
      .catch(() => setComparison(null));
    apiRequest("/api/scans/history")
      .then((history) => setScans(history.map((item) => ({
        url: item.scan_url,
        result: item.verdict === "phishing" ? "Potential phishing" : "Likely safe",
        time: new Date(item.created_at).toLocaleString(),
      }))))
      .catch(() => setScans([]));
  }, []);

  async function scan(event) {
    event.preventDefault();
    if (!url.trim()) {
      setScanState("validation");
      return;
    }

    setScanState("loading");
    setScanError("");
    try {
      const result = await apiRequest("/api/scans", {
        method: "POST",
        body: JSON.stringify({ url }),
      });
      setEngineAvailable(true);
      setScanState(result.verdict);
      setScanProbability(result.probability);
      setScanResult(result);
      setScans((current) => [{ url: result.url, result: result.verdict === "phishing" ? "Potential phishing" : "Likely safe", time: "Just now" }, ...current].slice(0, 4));
    } catch (requestError) {
      setScanState(requestError.message.includes("valid") ? "validation" : "error");
      setScanError(requestError.message);
    }
  }

  const result = scanState === "phishing" || scanState === "legitimate";
  const riskScore = scanState === "phishing" ? "High risk" : "Low risk";
  async function submitFeedback(event) {
    event.preventDefault();
    if (!url || (!feedback.verdict && !feedback.comment.trim()) || !result) {
      setFeedback((current) => ({ ...current, status: "Choose a verdict or type feedback such as “it is actually phishing”." }));
      return;
    }
    try {
      const response = await apiRequest("/api/feedback", {
        method: "POST",
        body: JSON.stringify({
          scan_url: url,
          predicted_verdict: scanState,
          analyst_verdict: feedback.verdict || "uncertain",
          probability: scanProbability,
          decision_threshold: scanResult.decision_threshold,
          comment: feedback.comment || undefined,
          feedback_text: feedback.comment || undefined,
        }),
      });
      setFeedback((current) => ({
        ...current,
        status: `Feedback recorded. ${response.trained_feedback_labels} labeled examples are being used for adaptive learning. Active threshold: ${Number(response.adaptive_threshold).toFixed(2)}.`,
      }));
    } catch (requestError) {
      setFeedback((current) => ({ ...current, status: requestError.message }));
    }
  }
  const sectionDescriptions = {
    "URL Scanner": "Submit a URL to the connected detection engine for a model-backed verdict.",
    "Detection History": "Recent authenticated scan results are persisted in the QPhish database.",
    "Model Comparison": "The backend currently combines the saved Model A and Model B probabilities.",
    "Adaptive Learning": "Analyst feedback updates the live threshold policy and periodically adapts the quantum model.",
    "Research Analytics": "Authenticated scan and feedback metrics are available from the monitoring service.",
    Settings: "Workspace preferences and secure account settings.",
    About: "QPhish analyzes URL structure without opening submitted websites.",
  };
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand sidebar-brand"><div className="brand-mark">◈</div><div><strong>QPhish</strong><small>THREAT INTELLIGENCE</small></div></div>
        <nav>{navItems.map(([icon, label]) => <button key={label} className={active === label ? "nav-item active" : "nav-item"} onClick={() => setActive(label)}><span>{icon}</span>{label}</button>)}</nav>
        <div className="sidebar-bottom">
          <button className="nav-item" onClick={() => setActive("Settings")}><span>⚙</span>Settings</button>
          <button className="nav-item" onClick={() => setActive("About")}><span>ⓘ</span>About</button>
          <div className="user-chip"><div className="avatar">{user.name[0].toUpperCase()}</div><div><strong>{user.name}</strong><small>{user.email}</small></div><button onClick={onSignOut} title="Sign out">↪</button></div>
        </div>
      </aside>
      <main className="main-content">
        <header className="topbar"><div><p className="eyebrow">SECURITY WORKSPACE</p><p className="muted">Website risk assessment · backend-connected when available</p></div><div className={`engine-pill ${engineAvailable ? "available" : ""}`}><i /> Detection engine {engineAvailable ? "available" : "unavailable"}</div></header>
        <div className="content">
          <div className="hero-grid">
            <section className="hero-card"><p className="eyebrow">YOUR SECURITY OVERVIEW</p><h1>Welcome to a clearer view of website risk.</h1><p>Submit a website address for analysis by your connected detection service. No submitted website is opened in this page.</p><button className="primary" onClick={() => document.getElementById("url-input").focus()}>Scan a website <span>→</span></button></section>
            <section className="status-card"><p className="eyebrow">DETECTION ENGINE</p><h3><i className={engineAvailable ? "online" : ""} /> Detection engine<br /> {engineAvailable ? "available" : "unavailable"}</h3><p>Engine status is read from the backend. An unavailable connection is never reported as active.</p></section>
          </div>
          {active !== "Dashboard" && <section className="section-notice"><p className="eyebrow">{active.toUpperCase()}</p><h2>{active}</h2><p>{sectionDescriptions[active]}</p>{active === "URL Scanner" && <button className="primary" onClick={() => document.getElementById("url-input").focus()}>Start a scan <span>→</span></button>}</section>}
          <div className="analysis-layout">
            <section className="scanner-card">
              <div className="section-title"><div className="scan-icon">⌕</div><div><h2>Scan a website</h2><p>Connect to detection API for a verdict</p></div></div>
              <form onSubmit={scan} className="scan-form"><label htmlFor="url-input">Website URL</label><div className="input-row"><input id="url-input" value={url} onChange={(e) => { setUrl(e.target.value); setScanState("idle"); setScanResult(null); setFeedback({ verdict: "", comment: "", status: "" }); }} placeholder="https://example.org" /><button className="primary" type="submit">{scanState === "loading" ? "Analyzing…" : "Analyze URL"}</button></div><small>Only HTTP or HTTPS URLs are accepted. Query parameters and fragments are hidden from displayed results.</small></form>
              {scanState === "validation" && <div className="inline-alert warning">{scanError || "Enter a valid HTTP or HTTPS website URL before scanning."}</div>}
              {scanState === "error" && <div className="inline-alert warning">{scanError || "The detection service is unavailable. Start the backend and try again."}</div>}
              {result && <div className={`result ${scanState}`}><div><span className="result-label">{scanState === "phishing" ? "POTENTIAL PHISHING" : "LIKELY LEGITIMATE"}</span><h2>{scanState === "phishing" ? "Proceed with caution" : "No immediate risk detected"}</h2><p>{scanState === "phishing" ? "This URL contains patterns commonly associated with phishing. Do not enter credentials." : "The URL does not currently match the strongest phishing indicators."}</p></div><strong>{riskScore}<small> backend verdict</small></strong></div>}
              {result && <form className="feedback-panel" onSubmit={submitFeedback}>
                <div>
                  <p className="eyebrow">HELP IMPROVE DETECTION</p>
                  <h3>Was this verdict correct?</h3>
                  <p className="feedback-copy">Your label is saved securely and used to improve future URL decisions.</p>
                </div>
                <div className="feedback-actions" role="group" aria-label="Analyst verdict">
                  {["legitimate", "phishing", "uncertain"].map((verdict) => (
                    <button
                      type="button"
                      key={verdict}
                      className={feedback.verdict === verdict ? "secondary selected" : "secondary"}
                      onClick={() => setFeedback((current) => ({ ...current, verdict }))}
                    >
                      {verdict === "uncertain" ? "Not sure" : `Actually ${verdict}`}
                    </button>
                  ))}
                </div>
                <input
                  aria-label="Optional feedback comment"
                  value={feedback.comment}
                  onChange={(event) => setFeedback((current) => ({ ...current, comment: event.target.value }))}
                  placeholder="Type feedback, e.g. “it is actually phishing”"
                  maxLength={1000}
                />
                <button className="primary feedback-submit" type="submit">Submit feedback</button>
                {feedback.status && <p className="feedback-status">{feedback.status}</p>}
              </form>}
            </section>
            <AnalysisReport result={scanResult} comparison={comparison} />
          </div>
          <div className="metrics"><Metric label="URLs scanned" value="1,284" change="+12.4%" /><Metric label="High-risk detections" value="86" change="+4.8%" danger /><Metric label="Average confidence" value="94.2%" change="+2.1%" /><Metric label="Engine status" value={engineAvailable ? "Online" : "Offline"} change={engineAvailable ? "Connected" : "Backend required"} muted /></div>
          <section className="lower-grid"><div className="panel"><div className="panel-heading"><div><p className="eyebrow">RECENT ACTIVITY</p><h2>Recent scans</h2></div><button className="text-button" onClick={() => setActive("Detection History")}>View history →</button></div>{scans.length ? scans.map((item, index) => <div className="scan-row" key={`${item.url}-${index}`}><span className={item.result === "Potential phishing" ? "dot red" : "dot green"} /><div><strong>{item.url}</strong><small>{item.time}</small></div><em className={item.result === "Potential phishing" ? "risk" : "safe"}>{item.result}</em></div>) : <div className="empty-state">Your recent scan results will appear here.</div>}</div><div className="panel"><p className="eyebrow">RISK DISTRIBUTION</p><h2>Detection overview</h2><div className="donut"><div><strong>1,284</strong><small>Total scans</small></div></div><div className="legend"><span><i className="legend-safe" />Likely safe <b>78%</b></span><span><i className="legend-risk" />Potential phishing <b>22%</b></span></div></div></section>
          {active === "Model Comparison" && <section className="comparison-panel panel"><div className="panel-heading"><div><p className="eyebrow">MODEL COMPARISON</p><h2>Classical and quantum ensemble</h2></div><span className="engine-pill available">Live hybrid scoring</span></div><p className="panel-copy">Metrics come from the saved domain-held-out experiments. Live URL scores now blend the classical ensemble with the persisted quantum circuit probability.</p>{comparison ? <div className="model-table"><div className="model-row model-header"><span>Model</span><span>Type</span><span>Accuracy</span><span>F1 phishing</span><span>ROC AUC</span></div>{comparison.models.map((model) => <div className="model-row" key={model.id}><strong>{model.name}</strong><span>{model.type}</span><b>{(model.metrics.accuracy * 100).toFixed(1)}%</b><b>{(model.metrics.f1_phishing * 100).toFixed(1)}%</b><b>{(model.metrics.roc_auc * 100).toFixed(1)}%</b></div>)}</div> : <div className="empty-state">Model comparison is unavailable.</div>}<div className="final-score"><div><p className="eyebrow">FINAL LIVE RISK SCORE</p><h3>{comparison?.final_risk_score.live_model || "75% classical + 25% quantum interference"}</h3><p>{scanProbability === null ? "Run a scan to calculate the current score." : `Latest scan: ${(scanProbability * 100).toFixed(1)}% hybrid phishing probability`} · Quantum component: {comparison?.final_risk_score.quantum_included ? "included" : "unavailable"}</p></div><strong>{scanProbability === null ? "—" : `${(scanProbability * 100).toFixed(1)}%`}</strong></div><div className="feedback-box"><div><p className="eyebrow">ANALYST FEEDBACK LOOP</p><h3>Correct the verdict for future learning</h3><p>Labeled feedback is stored, updates the reinforcement-learning threshold, and retrains the live classical models. Uncertain feedback is stored but excluded from training.</p></div><form onSubmit={submitFeedback}><div className="feedback-actions"><button type="button" className={feedback.verdict === "legitimate" ? "feedback-choice selected" : "feedback-choice"} onClick={() => setFeedback((current) => ({ ...current, verdict: "legitimate" }))}>Actually legitimate</button><button type="button" className={feedback.verdict === "phishing" ? "feedback-choice selected" : "feedback-choice"} onClick={() => setFeedback((current) => ({ ...current, verdict: "phishing" }))}>Actually phishing</button><button type="button" className={feedback.verdict === "uncertain" ? "feedback-choice selected" : "feedback-choice"} onClick={() => setFeedback((current) => ({ ...current, verdict: "uncertain" }))}>Uncertain</button></div><input value={feedback.comment} onChange={(event) => setFeedback((current) => ({ ...current, comment: event.target.value }))} placeholder="Optional analyst note" /><button className="primary" type="submit">Submit feedback</button>{feedback.status && <small className="feedback-status">{feedback.status}</small>}</form></div></section>}
        </div>
      </main>
    </div>
  );
}

function AnalysisReport({ result, comparison }) {
  const score = result ? Math.round(result.probability * 100) : null;
  const reportModels = comparison?.models?.filter((model) => model.id !== "quantum-ensemble") || [];
  const modelNames = { A: "Logistic regression", B: "Soft-voting ensemble", quantum: "Quantum VQC", hybrid: "Hybrid live score" };
  return (
    <section className="report-card panel">
      <div className="panel-heading"><div><p className="eyebrow">ANALYSIS REPORT</p><h2>Threat assessment</h2></div><span className="report-badge">LIVE</span></div>
      {result ? (
        <>
          <div className={`report-verdict ${result.verdict}`}><strong>VERDICT: {result.verdict === "phishing" ? "PHISHING" : "LEGITIMATE"}</strong><span>RECOMMENDATION: {result.verdict === "phishing" ? "BLOCK" : "ALLOW"}</span></div>
          <p className="report-url">{result.url}</p>
          <div className="risk-score"><div><span>Overall threat risk</span><strong>{score}%</strong></div><div className="risk-track"><i style={{ width: `${score}%` }} /></div></div>
          <div className="model-score-grid">
            {Object.entries(result.model_scores || {}).map(([id, value]) => <div className="model-score" key={id}><span>{modelNames[id] || id}</span><strong>{Math.round(value * 100)}%</strong><i style={{ width: `${Math.round(value * 100)}%` }} /></div>)}
          </div>
          <div className="report-section-label">Benchmark performance</div>
          <div className="benchmark-grid">{reportModels.map((model) => <div className="benchmark-card" key={model.id}><span>{model.name}</span><strong>{Math.round((model.metrics.accuracy || 0) * 100)}%</strong><small>accuracy</small></div>)}</div>
          <p className="report-note">Live risk blends the classical ensemble with quantum interference. Benchmark values come from domain-held-out evaluation.</p>
        </>
      ) : <div className="report-empty"><div className="report-empty-icon">◈</div><h3>Your report will appear here</h3><p>Run a URL scan to see the overall risk score, model agreement, and benchmark performance.</p></div>}
    </section>
  );
}

function Metric({ label, value, change, danger, muted }) { return <div className="metric"><span>{label}</span><strong className={danger ? "danger" : muted ? "muted-value" : ""}>{value}</strong><small className={danger ? "danger" : ""}>{change}</small></div>; }

function App() {
  const [authMode, setAuthMode] = useState("login");
  const [user, setUser] = useState(null);
  useEffect(() => {
    if (getAccessToken()) {
      apiRequest("/api/auth/me").then(setUser).catch(() => clearAccessToken());
    }
  }, []);
  return user ? <Dashboard user={user} onSignOut={async () => { try { await apiRequest("/api/auth/logout", { method: "POST" }); } finally { clearAccessToken(); setUser(null); } }} /> : <AuthPage mode={authMode} onAuthenticated={setUser} onSwitch={(nextMode) => setAuthMode(nextMode || (authMode === "login" ? "signup" : "login"))} />;
}

createRoot(document.getElementById("root")).render(<StrictMode><App /></StrictMode>);
