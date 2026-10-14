import { useEffect, useRef, useState } from "react";
import { Activity, ArrowRight, Check, CircleAlert, Download, Layers3, LoaderCircle, Moon, ScanLine, ShieldCheck, Sun, Upload, X } from "lucide-react";

const API = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
export default function App() {
  const input = useRef(null);
  const [theme, setTheme] = useState(() => localStorage.getItem("mammo-theme") || "light");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [api, setApi] = useState("checking");

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem("mammo-theme", theme); }, [theme]);
  useEffect(() => { fetch(API + "/api/health").then((r) => { if (!r.ok) throw new Error(); return r.json(); }).then(() => setApi("online")).catch(() => setApi("offline")); }, []);
  useEffect(() => {
    if (!file) { setPreview(""); return undefined; }
    const url = URL.createObjectURL(file); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function choose(next) {
    setError(""); setData(null);
    if (!next) return;
    if (!/\.(png|jpe?g|bmp|pgm|tiff?)$/i.test(next.name)) { setError("Choose a PNG, JPG, BMP, PGM or TIFF image."); return; }
    if (next.size > 15 * 1024 * 1024) { setError("Maximum upload size is 15 MB."); return; }
    setFile(next);
  }

  async function run() {
    if (!file) { setError("Choose a mammography image first."); return; }
    setBusy(true); setError(""); setData(null);
    const body = new FormData(); body.append("file", file);
    try {
      const response = await fetch(API + "/api/segment", { method: "POST", body });
      const json = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(json.detail || "K-Means segmentation failed.");
      setData(json);
    } catch (e) { setError(e.message || "Could not connect to the local API."); }
    finally { setBusy(false); }
  }

  function downloadReport() {
    if (!data) return;
    const lines = ["Mammography K-Means segmentation report", "Image: " + data.filename, "Displayed size: " + data.width + " x " + data.height, "", ...data.results.map((r) => "k=" + r.k + ", inertia=" + r.inertia.toFixed(4) + ", silhouette=" + r.silhouette.toFixed(4) + ", cluster sizes=" + JSON.stringify(r.cluster_sizes)), "", data.note];
    const url = URL.createObjectURL(new Blob([lines.join("\n")], {type:"text/plain"}));
    const a = document.createElement("a"); a.href = url; a.download = "kmeans-segmentation-report.txt"; a.click(); URL.revokeObjectURL(url);
  }

  return <div className="lab-shell">
    <aside className="rail"><a className="brand" href="#top"><span className="brand-icon"><ScanLine size={22}/></span><span><b>IMAGE LAB</b><small>MAMMOGRAPHY / K-MEANS</small></span></a><div className="rail-caption">WORKBENCH</div><div className="rail-active"><Activity size={17}/> Intensity clustering <span>01</span></div><div className="rail-bottom"><span className={"status-dot " + api}/><div><b>{api === "online" ? "Local API connected" : api === "checking" ? "Connecting to API" : "API offline"}</b><small>Local image processing</small></div></div><div className="scope"><ShieldCheck size={14}/> Academic segmentation exercise</div></aside>
    <main className="main" id="top">
      <header className="topbar"><div><span>MEDICAL IMAGE PROCESSING</span><i>/</i><strong>Mammography K-Means segmentation</strong></div><button className="theme-btn" onClick={() => setTheme(theme === "light" ? "dark" : "light")} aria-label="Toggle theme">{theme === "light" ? <Moon size={17}/> : <Sun size={17}/>}</button></header>
      <section className="hero"><div><span className="eyebrow">PIXEL INTENSITY · UNSUPERVISED LEARNING</span><h1>Segment by<br/><em>intensity.</em></h1><p>Compare K-Means partitions at k = 2, 3 and 4, then review internal cluster metrics side by side.</p></div><div className="hero-index"><span>METHOD</span><b>K</b><small>MEANS</small></div></section>
      {error && <div className="alert"><CircleAlert size={17}/><span>{error}</span><button onClick={() => setError("")}><X size={15}/></button></div>}
      {api === "offline" && <div className="alert"><CircleAlert size={17}/><span>The Python API is offline. Start the local service on port 8000 and refresh.</span></div>}
      <div className="section-head"><span>01</span><h2>Input image</h2><small>UPLOAD A LOCAL IMAGE</small></div>
      <div className="upload-grid">
        <section className="card upload-card"><div className="card-title"><div><small>INPUT</small><h3>Original image</h3></div><span>A</span></div><input ref={input} className="hidden" type="file" accept=".png,.jpg,.jpeg,.bmp,.pgm,.tif,.tiff" onChange={(e) => choose(e.target.files && e.target.files[0])}/>
          {preview ? <><div className="preview"><img src={preview} alt="Selected mammography image"/></div><div className="file-meta"><div><b>{file.name}</b><small>{(file.size/1024).toFixed(0)} KB · preview loaded</small></div><button onClick={() => {setFile(null);setData(null);if(input.current)input.current.value="";}} aria-label="Remove image"><X size={15}/></button></div></> : <button className="dropzone" onClick={() => input.current && input.current.click()} onDragOver={(e) => e.preventDefault()} onDrop={(e) => {e.preventDefault();choose(e.dataTransfer.files && e.dataTransfer.files[0]);}}><span className="upload-glyph"><Upload size={22}/></span><b>Drop image to begin</b><small>or browse local files</small><span>PNG · JPG · PGM · TIFF</span></button>}
          {preview && <button className="change-image" onClick={() => input.current && input.current.click()}><Upload size={14}/> Replace image</button>}
        </section>
        <section className="card config-card"><div className="card-title"><div><small>PROCESSING CONFIGURATION</small><h3>Cluster comparison</h3></div><span>B</span></div><div className="config-number"><b>3</b><span>cluster configurations</span></div><div className="k-list"><div><span className="k-chip k2">K = 2</span><div><b>Coarse partition</b><small>Two intensity groups</small></div></div><div><span className="k-chip k3">K = 3</span><div><b>Intermediate partition</b><small>Three intensity groups</small></div></div><div><span className="k-chip k4">K = 4</span><div><b>Fine partition</b><small>Four intensity groups</small></div></div></div><button className="run-btn" disabled={!file || busy || api !== "online"} onClick={run}>{busy ? <><LoaderCircle className="spin" size={16}/> Running K-Means…</> : <>Run cluster comparison <ArrowRight size={16}/></>}</button><p className="fine-print">Large images are resized to a maximum dimension of 256 pixels for interactive responsiveness.</p></section>
      </div>
      <div className="section-head results-head"><span>02</span><h2>Segmentation results</h2><small>{data ? data.width + " × " + data.height + " PX" : "OUTPUTS AFTER PROCESSING"}</small></div>
      {data ? <><div className="original-strip"><img src={data.original} alt="Processed grayscale input"/><div><small>PROCESSED INPUT</small><b>{data.filename}</b><span>{data.original_width} × {data.original_height} original pixels{data.downsampled ? " · resized for interactive analysis" : ""}</span></div><button onClick={downloadReport}><Download size={15}/> Export metrics</button></div><div className="results-grid">{data.results.map((r) => <article className="result-card" key={r.k}><div className="result-top"><div><span>K = {r.k}</span><h3>{r.k} intensity clusters</h3></div><small>RESULT</small></div><div className="result-img"><img src={r.preview} alt={"K-Means segmentation with " + r.k + " clusters"}/></div><div className="metric-pair"><div><small>SILHOUETTE</small><b>{r.silhouette.toFixed(4)}</b></div><div><small>INERTIA</small><b>{r.inertia.toFixed(1)}</b></div></div><div className="cluster-sizes"><span>Pixels by cluster</span><div>{Object.entries(r.cluster_sizes).map(([label, size]) => <span key={label}>C{Number(label)+1} <b>{Number(size).toLocaleString()}</b></span>)}</div></div></article>)}</div><div className="note"><ShieldCheck size={17}/><p><b>Interpretation note</b><br/>{data.note} Silhouette score and inertia describe clustering structure, not clinical segmentation quality.</p></div></> : <div className="empty"><Layers3 size={28}/><b>No results to compare yet</b><span>Run the same image through K-Means with k = 2, 3 and 4 to inspect how the intensity partitions change.</span></div>}
      <footer><span>IMAGE LAB <i>/</i> MAMMOGRAPHY SEGMENTATION</span><span>UNSUPERVISED PIXEL INTENSITY CLUSTERING · NOT DIAGNOSTIC</span></footer>
    </main>
  </div>;
}
