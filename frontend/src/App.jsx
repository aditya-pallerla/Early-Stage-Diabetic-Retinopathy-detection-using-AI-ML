// import Particles, { initParticlesEngine } from "@tsparticles/react";
// import { useEffect, useState } from "react";
// import { loadSlim } from "@tsparticles/slim";

// function App() {
//   const [init, setInit] = useState(false);

//   // initialize tsparticles
//   useEffect(() => {
//     initParticlesEngine(async (engine) => {
//       await loadSlim(engine);
//     }).then(() => {
//       setInit(true);
//     });
//   }, []);

//   return (
//     <div
//       style={{
//         height: "100vh",
//         width: "100vw",
//         backgroundColor: "#0d0d0d",
//         color: "white",
//       }}
//     >
//       {init && (
//         <Particles
//           id="tsparticles"
//           options={{
//             background: { color: "#0d0d0d" },
//             particles: {
//               number: { value: 80 },
//               color: { value: "#ff7a00" },
//               links: {
//                 enable: true,
//                 color: "#ff7a00",
//                 distance: 120,
//                 opacity: 0.3,
//               },
//               move: { enable: true, speed: 0.5 },
//               size: { value: 2 },
//               opacity: { value: 0.4 },
//             },
//             interactivity: {
//               events: { onHover: { enable: true, mode: "repulse" } },
//               modes: { repulse: { distance: 100, duration: 0.4 } },
//             },
//           }}
//         />
//       )}

//       {/* Hero Section */}
//       <div
//         style={{
//           position: "relative",
//           zIndex: 1,
//           padding: "3rem",
//           textAlign: "center",
//         }}
//       >
//         <h1 style={{ fontSize: "2.5rem", color: "#ff7a00" }}>
//           Early DR Detector
//         </h1>
//         <p>
//           Detect early signs of diabetic retinopathy using retinal scans + blood
//           reports.
//         </p>
//       </div>
//     </div>
//   );
// }

// export default App;
// src/App.jsx
// import React from "react";
// import Dashboard from "./components/Dashboard";
// import "./index.css";

// export default function App() {
//   return <Dashboard />;
// }

// import { BrowserRouter, Routes, Route } from "react-router-dom";
// import Dashboard from "./components/Dashboard.jsx";
// import History from "./components/History.jsx";
// import "./index.css";

// export default function App() {
//   return (
//     <BrowserRouter>
//       <Routes>
//         <Route path="/" element={<Dashboard />} />
//         <Route path="/history" element={<History />} />
//       </Routes>
//     </BrowserRouter>
//   );
// }

// src/App.jsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import HistoryPage from "./components/History";
import "./index.css";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/history" element={<HistoryPage />} />
      </Routes>
    </Router>
  );
}
