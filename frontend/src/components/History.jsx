// src/components/History.jsx
import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import { loadSlim } from "@tsparticles/slim";
import "./history.css";

// --- API Endpoint Configuration ---
const API_HISTORY_URL = "http://localhost:8000/api/v1/history";

// --- New Component: Image Zoom Modal ---
const ImageZoomModal = ({ src, onClose }) => {
  if (!src) return null;
  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "rgba(0, 0, 0, 0.9)",
        zIndex: 10000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        cursor: "pointer",
      }}
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label="Zoomed Grad-CAM Image"
    >
      <div
        style={{
          maxWidth: "90%",
          maxHeight: "90%",
          boxShadow: "0 0 40px rgba(255, 122, 0, 0.5)",
          borderRadius: 10,
          overflow: "hidden",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <img
          src={src}
          alt="Zoomed Grad-CAM"
          style={{
            display: "block",
            maxWidth: "100%",
            maxHeight: "100%",
            objectFit: "contain",
          }}
        />
      </div>
      <button
        onClick={onClose}
        style={{
          position: "absolute",
          top: 20,
          right: 20,
          background: "rgba(255, 255, 255, 0.1)",
          color: "white",
          border: "none",
          borderRadius: "50%",
          width: 40,
          height: 40,
          fontSize: 18,
          cursor: "pointer",
          transition: "background 200ms",
        }}
        aria-label="Close image zoom"
      >
        &times;
      </button>
    </div>
  );
};

// ------------------------------------

export default function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [sortOrder, setSortOrder] = useState("newest");
  const [init, setInit] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [zoomedImageSrc, setZoomedImageSrc] = useState(null);

  // --- Particles useEffect ---
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

  // --- History Data Fetching ---
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await fetch(API_HISTORY_URL);
        if (!response.ok) {
          throw new Error("Failed to fetch history from server.");
        }
        const data = await response.json();

        // Sort the data immediately after fetching
        const sortedData = data.sort(
          (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
        );
        setHistory(sortedData);
        setSortOrder("newest");
      } catch (err) {
        console.error("History fetch error:", err);
        setError("Could not load history. Ensure backend server is running.");
        // Fallback to local storage if API fails
        const stored = JSON.parse(localStorage.getItem("edi_history") || "[]");
        setHistory(stored);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const toggleSort = () => {
    const newOrder = sortOrder === "newest" ? "oldest" : "newest";
    const sorted = [...history].sort((a, b) => {
      const dateA = new Date(a.date).getTime();
      const dateB = new Date(b.date).getTime();
      return newOrder === "newest" ? dateB - dateA : dateA - dateB;
    });
    setHistory(sorted);
    setSortOrder(newOrder);
  };

  const getStageClass = (prediction) => {
    if (prediction.includes("Stage 1")) return "stage-alert";
    if (prediction.includes("Stage 0")) return "stage-ok";
    return "stage-review";
  };

  return (
    <div className="history-page dashboard-root">
      {init && (
        <Particles id="tsparticles-history" options={particlesOptions} />
      )}

      {/* Header/Nav bar */}
      <header className="topnav" role="banner">
        <div className="brand">
          <h1 className="title-split">
            {"Early DR Detector".split("").map((char, i) => (
              <span key={i} className="letter">
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
          <Link className="nav-link" to="/">
            Dashboard
          </Link>
          <Link className="nav-link" to="/history">
            History
          </Link>
        </nav>
      </header>

      {/* Main Content */}
      <main className="history-container main-content-area">
        <h2 className="history-title">Patient History</h2>

        {loading && (
          <p className="no-history">Loading history from server...</p>
        )}
        {error && (
          <p
            className="error"
            style={{ textAlign: "center", color: "#ff7070" }}
          >
            {error}
          </p>
        )}

        {!loading && history.length === 0 ? (
          <p className="no-history">No records found yet.</p>
        ) : (
          !loading && (
            <>
              <button className="sort-btn" onClick={toggleSort}>
                Sort:{" "}
                {sortOrder === "newest" ? "Newest → Oldest" : "Oldest → Newest"}
              </button>

              <table className="history-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Date & Time</th>
                    <th>Months to Prog.</th>
                    <th>Predicted Stage</th>
                    <th>HbA1c</th>
                    <th>Grad-CAM</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((record, idx) => (
                    <tr key={idx}>
                      <td>{idx + 1}</td>
                      <td>
                        {record.date_local ||
                          new Date(record.date).toLocaleString()}
                      </td>
                      <td>{record.months_to_prog} months</td>
                      <td>
                        <span className={getStageClass(record.prediction)}>
                          {record.prediction}
                        </span>
                      </td>
                      <td>{record.HbA1c}</td>
                      <td>
                        {record.gradcam_preview ? (
                          <div
                            onClick={() =>
                              setZoomedImageSrc(record.gradcam_preview)
                            }
                            style={{
                              cursor: "zoom-in",
                              display: "inline-block",
                            }}
                            title="Click to zoom image"
                          >
                            <img
                              src={record.gradcam_preview}
                              alt="Grad-CAM"
                              className="gradcam-thumb"
                            />
                          </div>
                        ) : (
                          <span
                            style={{
                              color: "var(--muted)",
                              fontSize: "0.8em",
                            }}
                          >
                            N/A
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )
        )}
      </main>

      {/* Image Zoom Modal */}
      <ImageZoomModal
        src={zoomedImageSrc}
        onClose={() => setZoomedImageSrc(null)}
      />
    </div>
  );
}
