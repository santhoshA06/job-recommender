import React, { useState } from "react";

const API_URL = "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [model, setModel] = useState("tfidf");
  const [topK, setTopK] = useState(10);
  const [locationFilter, setLocationFilter] = useState("");
  const [titleFilter, setTitleFilter] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [data, setData] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setData(null);

    if (!file) {
      setError("Please upload a PDF resume.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("model", model);
    formData.append("top_k", topK);
    formData.append("location_filter", locationFilter);
    formData.append("title_filter", titleFilter);

    try {
      setLoading(true);

      const response = await fetch(`${API_URL}/recommend`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Something went wrong.");
      }

      setData(result);
    } catch (err) {
      setError(err.message || "Failed to fetch recommendations.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-shell">
      <div className="bg-blob blob-1"></div>
      <div className="bg-blob blob-2"></div>
      <div className="bg-blob blob-3"></div>

      <header className="hero">
        <div className="hero-pill">Smart Career Matching</div>
        <h1>
          Find Better Jobs
          <br />
          From Your Resume
        </h1>
        <p>
          Upload your resume PDF and get relevant job recommendations using
          retrieval models like TF-IDF and BM25.
        </p>
      </header>

      <section className="top-panel">
        <div className="upload-card">
          <div className="upload-left">
            <span className="section-tag">Resume Input</span>
            <h2>Upload your resume and personalize the search</h2>
            <p>
              Choose a retrieval model, set result count, and optionally narrow
              recommendations by title or location.
            </p>
          </div>

          <form onSubmit={handleSubmit} className="upload-form">
            <div className="field">
              <label>Resume PDF</label>
              <input
                type="file"
                accept=".pdf"
                onChange={(e) => setFile(e.target.files[0])}
              />
            </div>

            <div className="field-grid">
              <div className="field">
                <label>Model</label>
                <select value={model} onChange={(e) => setModel(e.target.value)}>
                  <option value="tfidf">TF-IDF</option>
                  <option value="bm25">BM25</option>
                </select>
              </div>

              <div className="field">
                <label>Top K</label>
                <input
                  type="number"
                  min="5"
                  max="20"
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </div>
            </div>

            <div className="field-grid">
              <div className="field">
                <label>Location Filter</label>
                <input
                  type="text"
                  placeholder="e.g. Remote, California"
                  value={locationFilter}
                  onChange={(e) => setLocationFilter(e.target.value)}
                />
              </div>

              <div className="field">
                <label>Job Title Filter</label>
                <input
                  type="text"
                  placeholder="e.g. Data Scientist"
                  value={titleFilter}
                  onChange={(e) => setTitleFilter(e.target.value)}
                />
              </div>
            </div>

            <button type="submit" className="primary-btn" disabled={loading}>
              {loading ? "Searching..." : "Get Recommendations"}
            </button>
          </form>
        </div>
      </section>

      {error && <div className="error-box">{error}</div>}

      {!data && !loading && (
        <section className="placeholder-card">
          <div className="placeholder-icon">✦</div>
          <h3>Ready to analyze your resume</h3>
          <p>
            Upload a resume PDF and the system will extract important signals,
            build a focused query, and rank matching jobs.
          </p>
        </section>
      )}

      {data && (
        <section className="content-grid">
          <div className="left-column">
            <div className="panel-card">
              <div className="panel-head">
                <h3>Generated Query</h3>
                <span className="panel-badge">{data.model?.toUpperCase()}</span>
              </div>
              <div className="query-box">{data.query}</div>
            </div>

            <div className="panel-card">
              <div className="panel-head">
                <h3>Extracted Signals</h3>
              </div>

              <div className="signal-block">
                <h4>Matched Titles</h4>
                <div className="chips">
                  {data.matched_titles?.length ? (
                    data.matched_titles.map((item, idx) => (
                      <span key={idx} className="chip">
                        {item}
                      </span>
                    ))
                  ) : (
                    <span className="muted-text">None found</span>
                  )}
                </div>
              </div>

              <div className="signal-block">
                <h4>Matched Skills</h4>
                <div className="chips">
                  {data.matched_skills?.length ? (
                    data.matched_skills.map((item, idx) => (
                      <span key={idx} className="chip chip-blue">
                        {item}
                      </span>
                    ))
                  ) : (
                    <span className="muted-text">None found</span>
                  )}
                </div>
              </div>

              <div className="signal-block">
                <h4>Top Terms</h4>
                <div className="chips">
                  {data.top_terms?.length ? (
                    data.top_terms.map((item, idx) => (
                      <span key={idx} className="chip chip-purple">
                        {item}
                      </span>
                    ))
                  ) : (
                    <span className="muted-text">None found</span>
                  )}
                </div>
              </div>
            </div>

            <div className="panel-card">
              <div className="panel-head">
                <h3>Resume Preview</h3>
              </div>
              <div className="resume-box">{data.resume_preview}</div>
            </div>
          </div>

          <div className="right-column">
            <div className="panel-card jobs-panel">
              <div className="panel-head">
                <h3>Top Recommended Jobs</h3>
                <span className="results-count">
                  {data.results?.length || 0} results
                </span>
              </div>

              <div className="jobs-grid">
                {data.results.map((job) => (
                  <div key={job.rank} className="job-card">
                    <div className="job-meta-top">
                      <span className="rank-pill">#{job.rank}</span>
                      <span className="score-pill">{job.score.toFixed(4)}</span>
                    </div>

                    <h4>{job.title}</h4>

                    <div className="job-meta">
                      <span>
                        <strong>Company:</strong> {job.company || "N/A"}
                      </span>
                      <span>
                        <strong>Location:</strong> {job.location || "N/A"}
                      </span>
                    </div>

                    <p>{job.snippet}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}