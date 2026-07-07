// src/components/Dashboard.jsx
import React, { useEffect, useState } from "react";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";
import "./dashboard.css";
import retinaImg from "../assets/image.png";
import { Link } from "react-router-dom";

// --- API Endpoint Configuration ---
const API_URL = "http://localhost:8000/api/v1/predict";

// --- MAPPING Frontend Form Keys to Backend Keys ---
const FORM_KEY_MAP = {
  hba1c: "HbA1c",
  glucose: "fasting_glucose",
  duration: "duration_years",
  cholesterol: "cholesterol",
  age: "age",
};

export default function Dashboard() {
  const [init, setInit] = useState(false);

  // --- New states for upload/form (added) ---
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [errors, setErrors] = useState({});
  const [formData, setFormData] = useState({
    hba1c: "",
    glucose: "",
    duration: "",
    cholesterol: "",
    age: "",
  });
  const [submitting, setSubmitting] = useState(false);

  // ----- New states for results & modal -----
  const [resultData, setResultData] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [minimized, setMinimized] = useState(false);

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    }).then(() => {
      setInit(true);
    });
  }, []);

  const particlesOptions = {
    fullScreen: { enable: true, zIndex: -1 },
    background: { color: { value: "#070707" } },
    particles: {
      number: { value: 170 },
      color: { value: "#cc6600" },
      links: {
        enable: true,
        color: "#cc6600",
        distance: 80,
        opacity: 0.5,
      },
      move: {
        enable: true,
        speed: 0.6,
        direction: "none",
        straight: false,
        outModes: { default: "out" },
      },
      size: { value: 5 },
      opacity: { value: 0.2 },
    },

    interactivity: {
      detectsOn: "window",
      events: { onHover: { enable: true, mode: "repulse" }, resize: true },
      modes: { repulse: { distance: 110, duration: 0.6 } },
    },
    detectRetina: true,
  };

  // --------- File handlers ----------
  const handleFileChange = (e) => {
    const chosen = e?.target?.files?.[0];
    if (!chosen) return;

    // validation
    if (chosen.size > 5 * 1024 * 1024) {
      setErrors((s) => ({ ...s, file: "File must be under 5 MB" }));
      setFile(null);
      setPreview(null);
      return;
    }
    if (!["image/jpeg", "image/png"].includes(chosen.type)) {
      setErrors((s) => ({ ...s, file: "Only JPG or PNG allowed" }));
      setFile(null);
      setPreview(null);
      return;
    }

    setErrors((s) => ({ ...s, file: null }));
    setFile(chosen);
    setPreview(URL.createObjectURL(chosen));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) {
      const fakeEvent = { target: { files: [dropped] } };
      handleFileChange(fakeEvent);
    }
  };

  // --------- Form handlers ----------
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData((s) => ({ ...s, [name]: value }));
    setErrors((s) => ({ ...s, [name]: null }));
  };

  // --- Risk/Suggestion Logic (fallback only) ---
  const getPredictionSuggestion = (predictionLabel, months) => {
    if (predictionLabel === "Need Specialist Review") {
      return "Uncertain result. Specialist review highly recommended for confirmation.";
    }
    if (predictionLabel === "Stage 1") {
      if (months <= 3)
        return "HIGH RISK. Urgent specialist review and intervention required.";
      if (months <= 6)
        return "High risk. Schedule specialist review soon and tighten control of clinical factors.";
      return "Moderate risk. Follow specialist recommendations and monitor closely.";
    }
    return "Low risk. Maintain strict control of HbA1c, Glucose, and Cholesterol. Recheck annually.";
  };

  // --- CORE SUBMIT FUNCTION ---
  const handleSubmit = (e) => {
    e.preventDefault();
    const errs = {};

    if (!file) errs.file = "Retinal image is required";
    if (!formData.hba1c) errs.hba1c = "Required";
    if (!formData.glucose) errs.glucose = "Required";
    if (!formData.duration) errs.duration = "Required";
    if (!formData.cholesterol) errs.cholesterol = "Required";
    if (!formData.age) errs.age = "Required";

    setErrors(errs);
    if (Object.keys(errs).length === 0) {
      setSubmitting(true);

      const apiFormData = new FormData();
      apiFormData.append("file", file);

      Object.entries(formData).forEach(([feKey, value]) => {
        const beKey = FORM_KEY_MAP[feKey];
        if (beKey) {
          apiFormData.append(beKey, value);
        }
      });

      fetch(API_URL, {
        method: "POST",
        body: apiFormData,
      })
        .then((response) => {
          if (!response.ok) {
            return response.json().then((data) => {
              throw new Error(
                data.error || `HTTP error! status: ${response.status}`
              );
            });
          }
          return response.json();
        })
        .then((data) => {
          setSubmitting(false);

          console.log("🔍 API Response data:", data);
          console.log("🔍 SHAP scores from API:", data.shap_scores);

          // ----- UPDATED: include backend recommendation -----
          const record = {
            id: Date.now(),
            date: new Date().toISOString(),
            risk: data.risk,
            months: data.months_to_prog,
            shap: data.shap_scores,
            preview: data.gradcam_base64_url,
            prediction: data.prediction,
            recommendation: data.recommendation || "",
            formData,
            // NEW: model-wise probabilities from backend
            image_model_says: data.image_model_says,
            clinical_model_says: data.clinical_model_says,
            final_fusion_risk: data.final_fusion_risk,
            confidence: data.confidence,
          };

          console.log("🔍 Created record with shap:", record.shap);

          const hist = JSON.parse(localStorage.getItem("edi_history") || "[]");
          hist.unshift(record);
          localStorage.setItem(
            "edi_history",
            JSON.stringify(hist.slice(0, 30))
          );

          setResultData(record);
          setShowModal(true);
          setMinimized(false);
          if (preview) URL.revokeObjectURL(preview);
          setPreview(null);
          setFile(null);
        })
        .catch((error) => {
          console.error("Prediction failed:", error);
          setErrors((s) => ({
            ...s,
            form: `Prediction failed: ${error.message}`,
          }));
          setSubmitting(false);
        });
    }
  };

  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const scrollToUpload = (e) => {
    if (e.metaKey || e.ctrlKey || e.which === 2) return;
    e.preventDefault();
    const el = document.getElementById("upload");
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
      el.setAttribute("tabindex", "-1");
      el.focus({ preventScroll: true });
    }
  };

  const closeModal = () => {
    setShowModal(false);
    setResultData(null);
    setMinimized(false);
  };
  const minimizeModal = () => {
    setMinimized(true);
  };
  const restoreModal = () => {
    setMinimized(false);
    setShowModal(true);
  };

  const btnPrimaryStyle = {
    background: "linear-gradient(180deg,#ff851a,#ff6a00)",
    border: "none",
    color: "#0b0b0b",
    padding: "10px 14px",
    borderRadius: 10,
    cursor: "pointer",
    fontWeight: 700,
    boxShadow: "0 6px 20px rgba(255,110,10,0.12)",
    transition: "box-shadow 180ms ease, transform 120ms ease",
  };
  const btnSecondaryStyle = {
    background: "transparent",
    border: "1px solid rgba(255,110,10,0.14)",
    color: "var(--text)",
    padding: "8px 12px",
    borderRadius: 8,
    cursor: "pointer",
    transition: "box-shadow 180ms ease, transform 120ms ease",
  };
  const spinnerSvg = (
    <svg
      width="18"
      height="18"
      viewBox="0 0 50 50"
      style={{ verticalAlign: "middle", marginRight: 8 }}
    >
      <path
        fill="#fff"
        d="M43.935,25.145c0-10.318-8.364-18.682-18.682-18.682c-10.318,0-18.682,8.364-18.682,18.682h4.068
      c0-8.06,6.553-14.613,14.613-14.613c8.06,0,14.613,6.553,14.613,14.613H43.935z"
      >
        <animateTransform
          attributeType="xml"
          attributeName="transform"
          type="rotate"
          from="0 25 25"
          to="360 25 25"
          dur="0.9s"
          repeatCount="indefinite"
        />
      </path>
    </svg>
  );

  // SHAP bars component - robust bar chart visualization
  const ShapBars = ({ bars = [] }) => {
    // Normalize incoming bars to an array of numeric scaled scores
    let validBars = [];

    if (Array.isArray(bars) && bars.length > 0) {
      // If array of objects with scaled_score, extract and coerce
      if (
        typeof bars[0] === "object" &&
        bars[0] !== null &&
        "scaled_score" in bars[0]
      ) {
        validBars = bars
          .map((it) => {
            const v = Number(it.scaled_score);
            return Number.isFinite(v) ? v : null;
          })
          .filter((v) => v !== null);
      } else {
        // Assume array of numbers or numeric strings
        validBars = bars
          .map((v) => Number(v))
          .filter((v) => Number.isFinite(v));
      }
    }

    // Check if there are valid bars to render
    if (!validBars || validBars.length === 0) {
      return (
        <div
          style={{
            height: 110,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--muted)",
            fontSize: "0.8em",
          }}
        >
          No SHAP data available.
        </div>
      );
    }

    const max = Math.max(...validBars, 1);
    const featureLabels = ["HbA1c", "Gluc", "Chol", "Dura", "Age"];

    return (
      <div
        style={{
          display: "flex",
          alignItems: "flex-end",
          gap: 6,
          height: 90,
          paddingTop: 20,
        }}
      >
        {validBars.map((b, i) => {
          const heightPerc = Math.round((b / max) * 100);
          const displayHeight = heightPerc; // show true scaled height

          return (
            <div
              key={i}
              style={{
                textAlign: "center",
                flex: 1,
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "flex-end",
                height: "100%",
              }}
            >
              <div
                style={{
                  fontSize: 10,
                  color: "#ffb573",
                  fontWeight: "bold",
                  marginBottom: 2,
                  minHeight: 14,
                }}
              >
                {b}
              </div>
              <div
                style={{
                  width: "100%",
                  height: `${displayHeight}%`,
                  background: "linear-gradient(180deg,#ffb57a,#ff7a00)",
                  borderRadius: "6px 6px 4px 4px",
                  boxShadow: "0 4px 8px rgba(0,0,0,0.3)",
                  transition: "height 300ms ease",
                }}
                title={`${featureLabels[i] || i}: ${b}`}
              />
              <div
                style={{
                  fontSize: 10,
                  color: "var(--muted)",
                  marginTop: 4,
                  whiteSpace: "nowrap",
                }}
              >
                {featureLabels[i] || `F${i}`}
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="dashboard-root">
      {init && <Particles id="tsparticles" options={particlesOptions} />}

      <header className="topnav" role="banner">
        <div className="brand">
          <h1 className="title-split">
            {"Early DR Detector".split("").map((char, index) => (
              <span key={index} className="letter">
                {char}
              </span>
            ))}
          </h1>
        </div>

        <nav
          className="toplinks"
          role="navigation"
          aria-label="Main navigation"
        >
          <a className="nav-link" href="#dashboard">
            Dashboard
          </a>

          <a
            className="nav-link"
            href="#upload"
            onClick={scrollToUpload}
            title="Jump to the Upload section"
          >
            Upload
          </a>

          <Link className="nav-link" to="/history">
            History
          </Link>
        </nav>
      </header>

      <main className="dashboard-container" id="dashboard">
        {/* HERO SECTION */}
        <section
          className="hero"
          aria-labelledby="hero-title"
          style={{
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            justifyContent: "center",
            paddingTop: "40px",
            paddingBottom: "40px",
            gap: "18px",
          }}
        >
          <h1 id="hero-title" className="hero-title">
            Detect <span className="highlight">early</span> — protect vision
          </h1>

          <p className="hero-sub" style={{ maxWidth: 920 }}>
            Diabetic Retinopathy (DR) is a complication of diabetes that affects
            the tiny blood vessels in the retina — the light-sensitive layer at
            the back of the eye. Early DR is often symptom-free but detectable
            with retinal imaging; timely detection and control of risk factors
            (blood sugar, blood pressure, lipids) can prevent vision loss.
          </p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "1fr 360px",
              gap: 18,
              alignItems: "start",
              width: "100%",
              maxWidth: 980,
            }}
          >
            <div style={{ color: "var(--muted)" }}>
              <h3 style={{ margin: "6px 0 8px 0", color: "var(--text)" }}>
                What to know
              </h3>

              <p style={{ marginTop: 0 }}>
                DR progresses through stages — from no visible signs (Stage 0)
                to mild signs (Stage 1) and, if uncontrolled, to
                sight-threatening stages. Our focus: reliably detect **Stage 0
                vs Stage 1** so patients get earlier follow-up and preventive
                care.
              </p>

              <ul
                className="dr-bullets"
                style={{ marginTop: 10, color: "var(--muted)" }}
              >
                <li>
                  <strong>Key causes:</strong> prolonged high blood sugar,
                  uncontrolled blood pressure, and unhealthy lipid levels.
                </li>
                <li>
                  <strong>Common signs (found on imaging):</strong>{" "}
                  microaneurysms, tiny haemorrhages, and retinal swelling.
                </li>
                <li>
                  <strong>Why early check matters:</strong> treatment or
                  lifestyle changes at Stage 0–1 can greatly reduce risk of
                  progression.
                </li>
              </ul>

              <h4 style={{ marginTop: 14, color: "var(--text)" }}>
                Screening & follow-up
              </h4>
              <ol
                style={{ marginTop: 6, color: "var(--muted)", paddingLeft: 18 }}
              >
                <li>
                  People with diabetes: retinal screening at least once a year
                  (more often if high risk).
                </li>
                <li>
                  If Stage 1 or suspicious findings: specialist referral for
                  closer monitoring.
                </li>
                <li>
                  Keep HbA1c, BP, and cholesterol under target to reduce
                  progression risk.
                </li>
              </ol>

              <ul
                className="dr-bullets"
                style={{ marginTop: 12, margin: 0, paddingLeft: 18 }}
              >
                <li>
                  <strong>Stage 0:</strong> No visible diabetic retinopathy.
                </li>
                <li>
                  <strong>Stage 1:</strong> Mild — small microaneurysms and
                  early signs.
                </li>
                <li>
                  <strong>Stage 2–4:</strong> Moderate to proliferative changes
                  requiring specialist care.
                </li>
              </ul>

              <p style={{ marginTop: 12 }}>
                For more info:{" "}
                <a
                  href="https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy"
                  target="_blank"
                  rel="noopener noreferrer"
                  style={{
                    color: "#ffb573",
                    textDecoration: "none",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 8,
                  }}
                  title="Open NIH/NEI page about diabetic retinopathy (new tab)"
                >
                  <svg
                    width="16"
                    height="16"
                    viewBox="0 0 24 24"
                    fill="none"
                    xmlns="http://www.w3.org/2000/svg"
                    aria-hidden="true"
                    focusable="false"
                    style={{ display: "inline-block", verticalAlign: "middle" }}
                  >
                    <circle
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="#ffb573"
                      strokeWidth="1.2"
                      fill="rgba(255,181,115,0.06)"
                    />
                    <path d="M11.5 11h1v5h-1zM11.5 7h1v1h-1z" fill="#ffb573" />
                  </svg>
                  Read more about diabetic retinopathy (NIH/NEI)
                </a>
              </p>
            </div>

            <aside
              aria-label="quick-facts"
              style={{
                background: "rgba(255,255,255,0.02)",
                borderRadius: 12,
                padding: 14,
                boxShadow: "0 8px 22px rgba(0,0,0,0.45)",
                border: "1px solid rgba(255,255,255,0.02)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 12,
                }}
              >
                <strong style={{ color: "var(--text)" }}>Quick facts</strong>
                <span style={{ color: "var(--muted)", fontSize: 13 }}>
                  Stage 0–1 focused
                </span>
              </div>

              <ul
                style={{
                  marginTop: 12,
                  listStyle: "none",
                  padding: 0,
                  color: "var(--muted)",
                }}
              >
                <li style={{ marginBottom: 8 }}>
                  <strong style={{ color: "var(--accent)" }}>Screening:</strong>{" "}
                  annual (or earlier if high risk)
                </li>
                <li style={{ marginBottom: 8 }}>
                  <strong style={{ color: "var(--accent)" }}>
                    Asymptomatic:
                  </strong>{" "}
                  early stages usually show no symptoms
                </li>
                <li style={{ marginBottom: 8 }}>
                  <strong style={{ color: "var(--accent)" }}>
                    Prevention:
                  </strong>{" "}
                  control HbA1c, BP, cholesterol
                </li>
              </ul>

              <div style={{ marginTop: 10 }}>
                <button
                  onClick={() => {
                    const el = document.getElementById("upload");
                    if (el)
                      el.scrollIntoView({ behavior: "smooth", block: "start" });
                  }}
                  className="btn-primary"
                  style={{ width: "100%", padding: "8px 10px" }}
                >
                  Go to Upload
                </button>
              </div>

              <div style={{ marginTop: 16, textAlign: "center" }}>
                <h5 style={{ marginBottom: 8, color: "var(--text)" }}>
                  Retinal diagram (healthy eye)
                </h5>
                <img
                  src={retinaImg}
                  alt="Vector diagram representing a healthy retina"
                  style={{
                    width: "100%",
                    borderRadius: 10,
                    boxShadow: "0 6px 16px rgba(0,0,0,0.5)",
                  }}
                />
              </div>
            </aside>
          </div>
        </section>

        {/* UPLOAD SECTION */}
        <section
          id="upload"
          className="upload-card"
          aria-label="upload-section"
        >
          <div className="upload-inner">
            {errors?.form && <div className="error">{errors.form}</div>}
            <h3>Upload retinal image & enter blood values</h3>
            <p className="muted">
              Drag & drop retinal image here or choose a file. Max 5 MB. JPG/PNG
              only.
            </p>
            <div
              className="drop-zone"
              onDrop={handleDrop}
              onDragOver={(e) => e.preventDefault()}
            >
              <div>
                <p>Drag & drop retinal image here</p>
                <label className="choose-file-btn">
                  Choose File
                  <input
                    type="file"
                    accept="image/png, image/jpeg"
                    onChange={handleFileChange}
                  />
                </label>
              </div>
            </div>
            {errors?.file && <div className="error">{errors.file}</div>}
            {preview && (
              <div className="preview-wrap">
                <img src={preview} alt="preview" className="preview-img" />
              </div>
            )}
            <form className="blood-form" onSubmit={handleSubmit}>
              <div className="form-row">
                <input
                  name="hba1c"
                  placeholder="HbA1c (e.g., 6.5%)"
                  value={formData.hba1c}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
                {errors?.hba1c && <div className="error">{errors.hba1c}</div>}
              </div>

              <div className="form-row">
                <input
                  name="glucose"
                  placeholder="Fasting Glucose (e.g., 110 mg/dL)"
                  value={formData.glucose}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
                {errors?.glucose && (
                  <div className="error">{errors.glucose}</div>
                )}
              </div>

              <div className="form-row">
                <input
                  name="duration"
                  placeholder="Duration of Diabetes (e.g., 4 years)"
                  value={formData.duration}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
                {errors?.duration && (
                  <div className="error">{errors.duration}</div>
                )}
              </div>

              <div className="form-row">
                <input
                  name="cholesterol"
                  placeholder="Cholesterol (e.g., 180 mg/dL)"
                  value={formData.cholesterol}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
                {errors?.cholesterol && (
                  <div className="error">{errors.cholesterol}</div>
                )}
              </div>

              <div className="form-row">
                <input
                  name="age"
                  placeholder="Patient Age (e.g., 45)"
                  value={formData.age}
                  onChange={handleInputChange}
                  disabled={submitting}
                />
                {errors?.age && <div className="error">{errors.age}</div>}
              </div>

              <div
                className="form-row submit-row"
                style={{ display: "flex", gap: 12 }}
              >
                <button
                  type="submit"
                  className="btn-primary"
                  disabled={submitting || !file}
                  style={{
                    ...btnPrimaryStyle,
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 6,
                    ...(submitting ? { opacity: 0.9 } : {}),
                  }}
                >
                  {submitting && spinnerSvg}
                  <span style={{ color: "#0b0b0b" }}>
                    {submitting ? "Processing..." : "Submit"}
                  </span>
                </button>

                {resultData && !minimized && (
                  <button
                    type="button"
                    onClick={() => {
                      setShowModal(true);
                      setMinimized(false);
                    }}
                    className="btn-secondary"
                    style={{
                      ...btnSecondaryStyle,
                      opacity: submitting ? 0.5 : 1,
                    }}
                    disabled={submitting}
                  >
                    View Result
                  </button>
                )}
              </div>
            </form>
          </div>
        </section>

        {/* RESULTS MODAL */}
        {showModal && resultData && !minimized && (
          <div
            role="dialog"
            aria-modal="true"
            aria-label="Result popup"
            style={{
              position: "fixed",
              left: 0,
              top: 0,
              right: 0,
              bottom: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              zIndex: 9999,
              background: "rgba(3,3,3,0.55)",
              padding: 20,
            }}
          >
            <div
              style={{
                width: 880,
                maxWidth: "96%",
                borderRadius: 14,
                background: "linear-gradient(180deg,#0b0b0d,#0f0f10)",
                boxShadow: "0 30px 80px rgba(0,0,0,0.6)",
                padding: 18,
                color: "white",
                position: "relative",
              }}
            >
              <div>
                <div style={{ fontSize: 12, color: "var(--muted)" }}>
                  Prediction
                </div>
                <div
                  style={{
                    fontSize: 20,
                    fontWeight: 800,
                    color: resultData.prediction.includes("Stage 1")
                      ? "#ff7070"
                      : "#8cff8c",
                  }}
                >
                  {resultData.prediction}
                </div>
                <div
                  style={{
                    fontSize: 18,
                    fontWeight: 800,
                    color: "#ffb573",
                    marginTop: 4,
                  }}
                >
                  Risk: {resultData.risk}%
                </div>

                {/* NEW: show per-model probabilities */}
                <div
                  style={{
                    marginTop: 6,
                    fontSize: 13,
                    color: "var(--muted)",
                    lineHeight: 1.5,
                  }}
                >
                  <div>
                    Image model (Stage 1 probability):{" "}
                    <span style={{ color: "#ffb573" }}>
                      {resultData.image_model_says || "N/A"}
                    </span>
                  </div>
                  <div>
                    Clinical model (Stage 1 probability):{" "}
                    <span style={{ color: "#ffb573" }}>
                      {resultData.clinical_model_says || "N/A"}
                    </span>
                  </div>
                  <div>
                    Fusion model (Stage 1 probability):{" "}
                    <span style={{ color: "#ffb573" }}>
                      {resultData.final_fusion_risk || "N/A"}
                    </span>
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 12 }}>
                <div style={{ marginBottom: 10 }}>
                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                    }}
                  >
                    <div style={{ color: "var(--muted)", fontSize: 13 }}>
                      Estimated time to progression (based on regression model)
                    </div>
                    <div style={{ fontWeight: 700 }}>
                      {resultData.months} months
                    </div>
                  </div>
                  <div
                    style={{
                      height: 12,
                      background: "rgba(255,255,255,0.06)",
                      borderRadius: 8,
                      marginTop: 8,
                    }}
                  >
                    {/* FIXED: use backend months and scale against 48 months max */}
                    <div
                      style={{
                        width: `${Math.round((resultData.months / 48) * 100)}%`,
                        height: "100%",
                        borderRadius: 8,
                        background: "linear-gradient(90deg,#ff7a00,#ffb573)",
                        boxShadow: "inset 0 -6px 14px rgba(0,0,0,0.45)",
                        transition: "width 400ms ease",
                      }}
                    />
                  </div>
                </div>

                <div
                  style={{
                    marginTop: 12,
                    marginBottom: 16,
                    color: "var(--muted)",
                    fontWeight: 600,
                  }}
                >
                  <strong>Suggestion: </strong>
                  <span style={{ color: "#fff", fontWeight: 500 }}>
                    {/* UPDATED: display backend recommendation if present, fallback to old function */}
                    {resultData.recommendation ||
                      getPredictionSuggestion(
                        resultData.prediction,
                        resultData.months
                      )}
                  </span>
                </div>

                <div style={{ display: "flex", gap: 12 }}>
                  <div style={{ flex: "0 0 55%" }}>
                    <div
                      style={{
                        fontSize: 13,
                        color: "var(--muted)",
                        marginBottom: 6,
                      }}
                    >
                      Grad-CAM (Localization of risk)
                    </div>
                    <div
                      style={{
                        borderRadius: 10,
                        overflow: "hidden",
                        background: "#0b0b0b",
                        padding: 8,
                      }}
                    >
                      <img
                        src={resultData.preview || retinaImg}
                        alt="Grad-CAM output heatmap"
                        style={{
                          width: "100%",
                          height: "auto",
                          maxHeight: "220px",
                          objectFit: "contain",
                          display: "block",
                          margin: "0 auto",
                          borderRadius: 8,
                          filter: "contrast(1.05)",
                        }}
                      />
                      <div
                        style={{
                          position: "relative",
                          marginTop: 8,
                          color: "var(--muted)",
                          fontSize: 12,
                        }}
                      >
                        The heatmap highlights (yellow/red) regions contributing
                        most to the Stage 0/1 prediction.
                      </div>
                    </div>
                  </div>

                  <div style={{ flex: "0 0 42%" }}>
                    <div
                      style={{
                        fontSize: 13,
                        color: "var(--muted)",
                        marginBottom: 6,
                      }}
                    >
                      SHAP (Clinical feature impact)
                    </div>
                    <div
                      style={{
                        background: "rgba(255,255,255,0.02)",
                        padding: 12,
                        borderRadius: 8,
                        height: "80%",
                      }}
                    >
                      <ShapBars bars={resultData.shap || []} />
                      <div
                        style={{
                          fontSize: 11,
                          color: "var(--muted)",
                          marginTop: 8,
                          textAlign: "center",
                        }}
                      >
                        Bar height = feature influence on risk
                      </div>
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: 20, textAlign: "center" }}>
                  <button
                    onClick={minimizeModal}
                    style={{
                      ...btnPrimaryStyle,
                      width: "60%",
                      padding: "10px",
                      fontSize: 15,
                    }}
                  >
                    Save & Minimize
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {minimized && (
          <button
            onClick={restoreModal}
            style={{
              position: "fixed",
              bottom: 20,
              right: 20,
              ...btnPrimaryStyle,
              padding: "10px 16px",
              borderRadius: 50,
            }}
          >
            View Last Result
          </button>
        )}
      </main>
    </div>
  );
}
