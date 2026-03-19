"""
MXDiscovery Interactive Dashboard
=================================
Streamlit-based UI for the MXDiscovery computational pipeline.
Discover novel non-toxic MXene composites for wearable thermoelectric energy harvesting.

Run:  streamlit run app.py
"""

import streamlit as st
import json
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DFT_DIR = DATA_DIR / "results" / "dft"

st.set_page_config(
    page_title="MXDiscovery",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .stage-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-right: 0.5rem;
    }
    .badge-complete { background: #10b981; }
    .badge-active  { background: #3b82f6; }
    div[data-testid="stMetric"] {
        background-color: #1e1e2e;
        border: 1px solid #3b3b5c;
        border-radius: 12px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label {
        color: #a0a0c0 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #e0e0ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Data loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_json(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return json.load(f)

@st.cache_data
def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
    return records

@st.cache_data
def load_dos_data(material="Mo2C_O"):
    dos_file = DFT_DIR / material / f"{material}.dos"
    if not dos_file.exists():
        return None, None, None, None
    energies, dos_vals, int_dos = [], [], []
    ef = 0.0
    with open(dos_file) as f:
        for line in f:
            if line.startswith("#"):
                # parse Fermi energy from header
                parts = line.split("EFermi")
                if len(parts) > 1:
                    ef = float(parts[1].split("eV")[0].replace("=", "").strip())
                continue
            cols = line.split()
            if len(cols) >= 3:
                energies.append(float(cols[0]))
                dos_vals.append(float(cols[1]))
                int_dos.append(float(cols[2]))
    return np.array(energies), np.array(dos_vals), np.array(int_dos), ef

@st.cache_data
def load_bands_data(material="Mo2C_O"):
    bands_file = DFT_DIR / material / f"{material}.bands.dat.gnu"
    if not bands_file.exists():
        return None
    bands = []
    current_band = {"k": [], "e": []}
    with open(bands_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_band["k"]:
                    bands.append(current_band)
                    current_band = {"k": [], "e": []}
                continue
            cols = line.split()
            if len(cols) >= 2:
                current_band["k"].append(float(cols[0]))
                current_band["e"].append(float(cols[1]))
    if current_band["k"]:
        bands.append(current_band)
    return bands

@st.cache_data
def load_papers_stats():
    papers_path = DATA_DIR / "papers" / "papers.jsonl"
    records_path = DATA_DIR / "papers" / "extracted_records.jsonl"
    papers = []
    if papers_path.exists():
        with open(papers_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        papers.append(json.loads(line))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
    records = []
    if records_path.exists():
        with open(records_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        pass
    return papers, records

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "Select View",
    [
        "Overview",
        "Stage 1: Literature Mining",
        "Stage 2: Gap Analysis",
        "Stage 3-4: Structures & Stability",
        "Stage 5: TE Properties & Ranking",
        "Stage 6: DFT Validation",
        "Final Results",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline Status")
stages = [
    ("1a", "Paper Fetching", True),
    ("1b", "LLM Extraction", True),
    ("2", "Gap Analysis", True),
    ("2b", "Toxicity Screen", True),
    ("3", "Structure Gen", True),
    ("4", "CHGNet Screen", True),
    ("5", "TE Prediction", True),
    ("6", "DFT Validation", True),
]
for sid, name, done in stages:
    icon = "✅" if done else "⏳"
    st.sidebar.markdown(f"{icon} **Stage {sid}:** {name}")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>MXDiscovery v1.0 | Gudibandi Sri Nikhil Reddy<br>"
    "Shizuoka University, Japan</small>",
    unsafe_allow_html=True,
)

# ===========================================================================
# PAGE: Overview
# ===========================================================================
if page == "Overview":
    st.markdown('<h1 class="main-header">MXDiscovery</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Computational Discovery of Novel Non-Toxic MXene Composites '
        'for Wearable Thermoelectric Energy Harvesting</p>',
        unsafe_allow_html=True,
    )

    # Load key numbers
    papers, records = load_papers_stats()
    gap_path = DATA_DIR / "gap_analysis_candidates.json"
    safe_path = DATA_DIR / "safe_candidates.json"
    rank_path = DATA_DIR / "final_rankings.json"
    gap_candidates = load_json(str(gap_path)) if gap_path.exists() else []
    safe_candidates = load_json(str(safe_path)) if safe_path.exists() else []
    rankings = load_json(str(rank_path)) if rank_path.exists() else []

    # Key metrics row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Papers Mined", f"{len(papers):,}")
    c2.metric("TE Records", f"{len(records)}")
    c3.metric("Gap Candidates", f"{len(gap_candidates)}")
    c4.metric("Safe Candidates", f"{len(safe_candidates)}")
    c5.metric("Final Ranked", f"{len(rankings)}")

    st.markdown("---")

    # Pipeline flow diagram — HTML/CSS version
    st.markdown("### Pipeline Architecture")
    pipeline_stages = [
        ("1a", "Paper Fetching", "2,000 papers", "#6366f1", "OpenAlex API"),
        ("1b", "LLM Extraction", "123 TE records", "#7c3aed", "Qwen2.5:7b"),
        ("2", "Gap Analysis", "50 candidates", "#9333ea", "Composition space"),
        ("2b", "Toxicity Filter", "20 safe", "#c026d3", "Biocompatibility"),
        ("3-4", "Structure + CHGNet", "20 stable", "#db2777", "ASE + ML potential"),
        ("5", "TE + TOPSIS", "20 ranked", "#e11d48", "Multi-criteria"),
        ("6", "DFT Validation", "Mo2CO2 confirmed", "#dc2626", "Quantum ESPRESSO"),
    ]
    pipeline_html = '<div style="display:flex;align-items:flex-start;justify-content:space-between;padding:15px 0;gap:4px;">'
    for i, (sid, title, result, color, tool) in enumerate(pipeline_stages):
        pipeline_html += f'''
        <div style="display:flex;flex-direction:column;align-items:center;flex:1;min-width:0;">
            <div style="
                width:56px;height:56px;border-radius:14px;
                background:linear-gradient(135deg, {color}, {color}cc);
                display:flex;align-items:center;justify-content:center;
                box-shadow:0 4px 12px {color}44;
                border:2px solid {color}66;
            ">
                <span style="color:white;font-weight:800;font-size:0.95rem;">{sid}</span>
            </div>
            <div style="margin-top:8px;text-align:center;padding:0 2px;">
                <div style="color:#e2e8f0;font-weight:700;font-size:0.75rem;line-height:1.2;">{title}</div>
                <div style="color:#10b981;font-weight:600;font-size:0.72rem;margin-top:2px;">{result}</div>
                <div style="color:#64748b;font-size:0.65rem;margin-top:1px;">{tool}</div>
            </div>
        </div>'''
        if i < len(pipeline_stages) - 1:
            pipeline_html += '''
            <div style="display:flex;align-items:center;padding-top:18px;flex-shrink:0;">
                <svg width="22" height="14" viewBox="0 0 22 14">
                    <line x1="0" y1="7" x2="14" y2="7" stroke="#64748b" stroke-width="2"/>
                    <polygon points="14,3 22,7 14,11" fill="#94a3b8"/>
                </svg>
            </div>'''
    pipeline_html += '</div>'
    st.markdown(pipeline_html, unsafe_allow_html=True)

    # Top candidate highlight
    if rankings:
        st.markdown("### Top Candidate")
        top = rankings[0]
        tc1, tc2, tc3, tc4, tc5 = st.columns(5)
        tc1.metric("Composition", f"{top['mxene_formula']}T{top['termination']}")
        tc2.metric("Partner", top["composite_partner"])
        tc3.metric("TOPSIS Score", f"{top['topsis_score']:.3f}")
        tc4.metric("Seebeck", f"{top['seebeck']:.1f} uV/K")
        tc5.metric("Power Factor", f"{top['power_factor']:.1f} uW/mK2")

# ===========================================================================
# PAGE: Stage 1 — Literature Mining
# ===========================================================================
elif page == "Stage 1: Literature Mining":
    st.markdown("## Stage 1: Literature Mining")
    papers, records = load_papers_stats()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Papers", f"{len(papers):,}")
    col2.metric("Valid TE Records", f"{len(records)}")
    col3.metric("Extraction Rate", f"{len(records)/max(len(papers),1)*100:.1f}%")

    st.markdown("---")

    # Year distribution
    if papers:
        years = [p.get("year", 0) for p in papers if p.get("year")]
        if years:
            year_df = pd.DataFrame({"year": years})
            fig_year = px.histogram(
                year_df, x="year", nbins=30,
                title="Publication Year Distribution",
                color_discrete_sequence=["#6366f1"],
            )
            fig_year.update_layout(xaxis_title="Year", yaxis_title="Number of Papers")
            st.plotly_chart(fig_year, width="stretch")

    # Citation distribution
    if papers:
        citations = [p.get("citation_count", 0) for p in papers if p.get("citation_count") is not None]
        if citations:
            c1, c2 = st.columns(2)
            with c1:
                fig_cite = px.histogram(
                    pd.DataFrame({"citations": citations}),
                    x="citations", nbins=50,
                    title="Citation Count Distribution",
                    color_discrete_sequence=["#8b5cf6"],
                )
                fig_cite.update_layout(xaxis_title="Citations", yaxis_title="Count")
                st.plotly_chart(fig_cite, width="stretch")

            with c2:
                # Top cited papers
                top_papers = sorted(papers, key=lambda x: x.get("citation_count", 0), reverse=True)[:10]
                st.markdown("#### Top 10 Most Cited Papers")
                for i, p in enumerate(top_papers):
                    st.markdown(
                        f"**{i+1}.** {p.get('title', 'N/A')[:80]}... "
                        f"({p.get('citation_count', 0)} citations, {p.get('year', '?')})"
                    )

    # Extracted records summary
    if records:
        st.markdown("---")
        st.markdown("### Extracted TE Records")
        rec_df = pd.DataFrame(records)
        display_cols = [c for c in [
            "mxene_composition", "termination", "composite_partner",
            "seebeck_coefficient", "electrical_conductivity", "power_factor",
            "zt_value", "synthesis_method", "confidence"
        ] if c in rec_df.columns]
        st.dataframe(rec_df[display_cols], width="stretch", height=400)

        # Composition frequency
        if "mxene_composition" in rec_df.columns:
            comp_counts = rec_df["mxene_composition"].value_counts().head(15)
            fig_comp = px.bar(
                x=comp_counts.index, y=comp_counts.values,
                title="Most Studied MXene Compositions in Literature",
                labels={"x": "Composition", "y": "Count"},
                color_discrete_sequence=["#a855f7"],
            )
            st.plotly_chart(fig_comp, width="stretch")

# ===========================================================================
# PAGE: Stage 2 — Gap Analysis
# ===========================================================================
elif page == "Stage 2: Gap Analysis":
    st.markdown("## Stage 2: Gap Analysis & Toxicity Screening")

    gap_path = DATA_DIR / "gap_analysis_candidates.json"
    safe_path = DATA_DIR / "safe_candidates.json"
    gap_candidates = load_json(str(gap_path)) if gap_path.exists() else []
    safe_candidates = load_json(str(safe_path)) if safe_path.exists() else []

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Gap Candidates", len(gap_candidates))
    c2.metric("After Toxicity Filter", len(safe_candidates))
    c3.metric("Filtered Out", len(gap_candidates) - len(safe_candidates))

    st.markdown("---")

    if gap_candidates:
        gap_df = pd.DataFrame(gap_candidates)

        # Composition space heatmap
        st.markdown("### Composition-Space Exploration")
        if "mxene_formula" in gap_df.columns and "termination" in gap_df.columns:
            pivot = gap_df.groupby(["mxene_formula", "termination"]).agg(
                score=("overall_score", "mean")
            ).reset_index()
            pivot_table = pivot.pivot(index="mxene_formula", columns="termination", values="score")

            fig_hm = px.imshow(
                pivot_table,
                title="Average Score by MXene Formula and Termination",
                color_continuous_scale="Viridis",
                labels=dict(color="Score"),
                aspect="auto",
            )
            fig_hm.update_layout(height=400)
            st.plotly_chart(fig_hm, width="stretch")

    if safe_candidates:
        safe_df = pd.DataFrame(safe_candidates)
        st.markdown("### Safe Candidates (Toxicity-Screened)")

        # Radar chart for top safe candidates
        unique_formulas = safe_df["mxene_formula"].unique()[:6]
        agg_safe = safe_df.groupby("mxene_formula").agg({
            "novelty_score": "mean",
            "analogy_score": "mean",
            "synthesizability": "mean",
            "overall_score": "mean",
        }).reset_index()

        fig_radar = go.Figure()
        categories = ["Novelty", "Analogy", "Synthesizability", "Overall"]
        for _, row in agg_safe.iterrows():
            values = [row["novelty_score"], row["analogy_score"], row["synthesizability"], row["overall_score"]]
            fig_radar.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                name=row["mxene_formula"],
                fill="toself",
                opacity=0.6,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Candidate Score Profiles",
            height=450,
        )
        st.plotly_chart(fig_radar, width="stretch")

        # Toxicity breakdown
        if "toxicity_class" in safe_df.columns:
            tox_counts = safe_df["toxicity_class"].value_counts()
            fig_tox = px.pie(
                values=tox_counts.values, names=tox_counts.index,
                title="Toxicity Classification Distribution",
                color_discrete_map={"SAFE": "#10b981", "CAUTION": "#f59e0b", "TOXIC": "#ef4444"},
            )
            st.plotly_chart(fig_tox, width="stretch")

# ===========================================================================
# PAGE: Stage 3-4 — Structures & Stability
# ===========================================================================
elif page == "Stage 3-4: Structures & Stability":
    st.markdown("## Stage 3-4: Structure Generation & CHGNet Stability Screening")

    screen_path = DATA_DIR / "screening_results.json"
    screening = load_json(str(screen_path)) if screen_path.exists() else []

    if screening:
        screen_df = pd.DataFrame(screening)
        # Convert string types
        for col in ["formation_energy", "energy_per_atom", "max_force", "final_energy"]:
            if col in screen_df.columns:
                screen_df[col] = pd.to_numeric(screen_df[col], errors="coerce")

        c1, c2, c3 = st.columns(3)
        c1.metric("Structures Generated", len(screening))
        c2.metric("Stable (E_f < 0)", int((screen_df["formation_energy"] < 0).sum()))
        c3.metric("Avg Formation Energy", f"{screen_df['formation_energy'].mean():.3f} eV/atom")

        st.markdown("---")

        # Formation energy bar chart
        st.markdown("### Formation Energy by Candidate")
        sorted_df = screen_df.sort_values("formation_energy")
        fig_fe = px.bar(
            sorted_df, x="name", y="formation_energy",
            title="CHGNet Formation Energy (eV/atom) - More negative = more stable",
            color="formation_energy",
            color_continuous_scale="RdYlGn_r",
        )
        fig_fe.update_layout(xaxis_tickangle=-45, height=500)
        fig_fe.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Stability threshold")
        st.plotly_chart(fig_fe, width="stretch")

        # Relaxation convergence
        st.markdown("### Relaxation Details")
        col1, col2 = st.columns(2)
        with col1:
            fig_steps = px.bar(
                screen_df, x="name", y="n_steps",
                title="CHGNet Relaxation Steps",
                color_discrete_sequence=["#6366f1"],
            )
            fig_steps.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_steps, width="stretch")

        with col2:
            fig_force = px.bar(
                screen_df, x="name", y="max_force",
                title="Max Residual Force After Relaxation (eV/A)",
                color_discrete_sequence=["#ec4899"],
            )
            fig_force.update_layout(xaxis_tickangle=-45)
            fig_force.add_hline(y=0.05, line_dash="dash", line_color="green",
                                annotation_text="Convergence threshold")
            st.plotly_chart(fig_force, width="stretch")

        # Data table
        st.markdown("### Full Screening Results")
        display_cols = ["name", "mxene_formula", "termination", "composite_partner",
                        "formation_energy", "energy_per_atom", "max_force", "n_steps", "n_atoms"]
        display_cols = [c for c in display_cols if c in screen_df.columns]
        st.dataframe(
            screen_df[display_cols].style.format({
                "formation_energy": "{:.4f}",
                "energy_per_atom": "{:.4f}",
                "max_force": "{:.6f}",
            }),
            width="stretch",
        )

# ===========================================================================
# PAGE: Stage 5 — TE Properties & Ranking
# ===========================================================================
elif page == "Stage 5: TE Properties & Ranking":
    st.markdown("## Stage 5: Thermoelectric Properties & TOPSIS Ranking")

    rank_path = DATA_DIR / "final_rankings.json"
    rankings = load_json(str(rank_path)) if rank_path.exists() else []

    if rankings:
        rank_df = pd.DataFrame(rankings)

        # Top metrics
        top = rankings[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top Candidate", f"{top['mxene_formula']}-{top['termination']}/{top['composite_partner']}")
        c2.metric("Best TOPSIS", f"{top['topsis_score']:.4f}")
        c3.metric("Best Seebeck", f"{top['seebeck']:.1f} uV/K")
        c4.metric("Best PF", f"{top['power_factor']:.1f} uW/mK2")

        st.markdown("---")

        # TOPSIS ranking bar
        st.markdown("### TOPSIS Rankings")
        fig_topsis = px.bar(
            rank_df, x="name", y="topsis_score",
            color="topsis_score",
            color_continuous_scale="Viridis",
            title="TOPSIS Multi-Criteria Decision Score",
        )
        fig_topsis.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig_topsis, width="stretch")

        # TE properties comparison
        st.markdown("### Thermoelectric Properties Comparison")
        col1, col2 = st.columns(2)

        with col1:
            fig_seebeck = px.bar(
                rank_df.head(10), x="name", y="seebeck",
                title="Seebeck Coefficient (uV/K) - Top 10",
                color="partner_type",
            )
            fig_seebeck.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_seebeck, width="stretch")

        with col2:
            fig_pf = px.bar(
                rank_df.head(10), x="name", y="power_factor",
                title="Power Factor (uW/mK2) - Top 10",
                color="partner_type",
            )
            fig_pf.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_pf, width="stretch")

        # Scatter: Seebeck vs Conductivity (TE trade-off)
        st.markdown("### Seebeck-Conductivity Trade-off")
        fig_trade = px.scatter(
            rank_df, x="conductivity", y="seebeck",
            size="power_factor", color="partner_type",
            hover_name="name",
            title="Seebeck vs Electrical Conductivity (bubble size = Power Factor)",
            labels={"conductivity": "Electrical Conductivity (S/cm)", "seebeck": "Seebeck (uV/K)"},
        )
        st.plotly_chart(fig_trade, width="stretch")

        # Partner type breakdown
        st.markdown("### Performance by Composite Partner Type")
        type_agg = rank_df.groupby("partner_type").agg({
            "seebeck": "mean",
            "conductivity": "mean",
            "power_factor": "mean",
            "thermal_conductivity": "mean",
            "topsis_score": "mean",
        }).reset_index()

        fig_type = make_subplots(rows=1, cols=3,
                                 subplot_titles=["Avg Seebeck", "Avg Power Factor", "Avg Thermal Conductivity"])
        fig_type.add_trace(
            go.Bar(x=type_agg["partner_type"], y=type_agg["seebeck"],
                   marker_color="#6366f1", name="Seebeck"), row=1, col=1)
        fig_type.add_trace(
            go.Bar(x=type_agg["partner_type"], y=type_agg["power_factor"],
                   marker_color="#10b981", name="PF"), row=1, col=2)
        fig_type.add_trace(
            go.Bar(x=type_agg["partner_type"], y=type_agg["thermal_conductivity"],
                   marker_color="#f43f5e", name="kappa"), row=1, col=3)
        fig_type.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_type, width="stretch")

        # Recommendation breakdown
        st.markdown("### Candidate Recommendations")
        rec_counts = rank_df["recommendation"].value_counts()
        fig_rec = px.pie(
            values=rec_counts.values, names=rec_counts.index,
            title="Recommendation Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        st.plotly_chart(fig_rec, width="stretch")

        # Full table
        st.markdown("### Full Rankings Table")
        st.dataframe(
            rank_df.style.format({
                "topsis_score": "{:.4f}",
                "seebeck": "{:.1f}",
                "conductivity": "{:.1f}",
                "power_factor": "{:.1f}",
                "thermal_conductivity": "{:.4f}",
                "zt": "{:.6f}",
                "formation_energy": "{:.4f}",
                "novelty_score": "{:.4f}",
            }).background_gradient(subset=["topsis_score"], cmap="Greens"),
            width="stretch",
        )

# ===========================================================================
# PAGE: Stage 6 — DFT Validation
# ===========================================================================
elif page == "Stage 6: DFT Validation":
    st.markdown("## Stage 6: DFT Validation (Quantum ESPRESSO)")

    dft_path = DATA_DIR / "dft_validation_results.json"
    dft_results = load_json(str(dft_path)) if dft_path.exists() else {}

    if dft_results:
        for material, data in dft_results.items():
            st.markdown(f"### Material: {material}")

            scf = data.get("scf", {})
            dos_info = data.get("dos", {})

            # Key DFT metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Energy", f"{scf.get('total_energy_ev', 0):.2f} eV")
            c2.metric("Fermi Energy", f"{scf.get('fermi_energy_ev', 0):.4f} eV")
            c3.metric("SCF Steps", scf.get("n_scf_steps", "N/A"))
            c4.metric("DFT Bandgap", f"{dos_info.get('bandgap_ev', 0):.2f} eV")
            c5.metric("Metallic?", "Yes" if dos_info.get("is_metallic") else "No")

            c6, c7, c8 = st.columns(3)
            c6.metric("DOS at Fermi", f"{dos_info.get('dos_at_fermi', 0):.3f} states/eV")
            c7.metric("CHGNet E_f", f"{data.get('chgnet_fe', 0):.4f} eV/atom")
            c8.metric("Wall Time", scf.get("wall_time", "N/A").split("WALL")[0].split("CPU")[-1].strip() if scf.get("wall_time") else "N/A")

            st.markdown("---")

            # DOS plot
            st.markdown("### Density of States (DOS)")
            energies, dos_vals, int_dos, ef = load_dos_data(material)
            if energies is not None:
                # Shift to Fermi level
                e_shifted = energies - ef

                # Limit range for visibility
                mask = (e_shifted >= -8) & (e_shifted <= 8)
                e_plot = e_shifted[mask]
                d_plot = dos_vals[mask]

                fig_dos = go.Figure()
                fig_dos.add_trace(go.Scatter(
                    x=e_plot, y=d_plot,
                    fill="tozeroy",
                    fillcolor="rgba(99, 102, 241, 0.3)",
                    line=dict(color="#6366f1", width=1.5),
                    name="DOS",
                ))
                fig_dos.add_vline(x=0, line_dash="dash", line_color="red",
                                  annotation_text="E_F", annotation_position="top")
                fig_dos.update_layout(
                    title=f"Electronic DOS - {material} (shifted to E_F = 0)",
                    xaxis_title="E - E_F (eV)",
                    yaxis_title="DOS (states/eV)",
                    height=500,
                    template="plotly_white",
                )
                st.plotly_chart(fig_dos, width="stretch")

            # Band structure
            st.markdown("### Electronic Band Structure")
            bands = load_bands_data(material)
            if bands:
                fig_bands = go.Figure()
                for i, band in enumerate(bands):
                    e_shifted_band = [e - (ef * 13.6057) for e in band["e"]]  # Convert Fermi from eV; bands in Ry->need check
                    fig_bands.add_trace(go.Scatter(
                        x=band["k"], y=band["e"],
                        mode="lines",
                        line=dict(color="#6366f1", width=1),
                        showlegend=False,
                        hoverinfo="y",
                    ))

                # Add Fermi level line (bands are in eV already from QE bands.dat.gnu)
                fig_bands.add_hline(
                    y=scf.get("fermi_energy_ev", 0),
                    line_dash="dash", line_color="red",
                    annotation_text="E_F",
                )
                fig_bands.update_layout(
                    title=f"Band Structure - {material}",
                    xaxis_title="k-path",
                    yaxis_title="Energy (eV)",
                    height=600,
                    template="plotly_white",
                    yaxis=dict(range=[scf.get("fermi_energy_ev", 0) - 5,
                                      scf.get("fermi_energy_ev", 0) + 5]),
                )
                st.plotly_chart(fig_bands, width="stretch")

            # SCF convergence info
            st.markdown("### DFT Calculation Summary")
            summary_data = {
                "Property": [
                    "Exchange-Correlation", "Pseudopotentials", "Wavefunction Cutoff",
                    "Charge Density Cutoff", "k-point Grid", "Smearing",
                    "SCF Convergence", "Total Energy (Ry)", "Fermi Energy (eV)",
                    "Bandgap (eV)", "Electronic Character",
                ],
                "Value": [
                    "PBE (GGA)", "PAW (pslibrary)", "60 Ry",
                    "480 Ry", "6x6x1 (SCF) / 12x12x1 (NSCF)", "Methfessel-Paxton, 0.02 Ry",
                    f"Converged in {scf.get('n_scf_steps', '?')} iterations",
                    f"{scf.get('total_energy_ry', 0):.6f}",
                    f"{scf.get('fermi_energy_ev', 0):.4f}",
                    f"{dos_info.get('bandgap_ev', 0):.2f}",
                    "Metallic" if dos_info.get("is_metallic") else "Semiconducting",
                ],
            }
            st.table(pd.DataFrame(summary_data))

    else:
        st.warning("No DFT results found. Run Stage 6 first.")

# ===========================================================================
# PAGE: Final Results
# ===========================================================================
elif page == "Final Results":
    st.markdown("## Final Discovery Results")

    rank_path = DATA_DIR / "final_rankings.json"
    dft_path = DATA_DIR / "dft_validation_results.json"
    rankings = load_json(str(rank_path)) if rank_path.exists() else []
    dft_results = load_json(str(dft_path)) if dft_path.exists() else {}

    if rankings:
        st.markdown("### Discovery Summary")
        st.markdown(
            "> Starting from **2,000 papers**, the MXDiscovery pipeline identified "
            f"**{len(rankings)} novel MXene composite candidates** for wearable thermoelectric "
            "energy harvesting, ranked by multi-criteria TOPSIS analysis."
        )

        st.markdown("---")

        # Top 5 candidates cards
        st.markdown("### Top 5 Candidates")
        for i, cand in enumerate(rankings[:5]):
            with st.expander(
                f"#{cand['rank']} | {cand['mxene_formula']}-{cand['termination']} / "
                f"{cand['composite_partner']}  |  TOPSIS: {cand['topsis_score']:.4f}",
                expanded=(i == 0),
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Seebeck", f"{cand['seebeck']:.1f} uV/K")
                c2.metric("Conductivity", f"{cand['conductivity']:.1f} S/cm")
                c3.metric("Power Factor", f"{cand['power_factor']:.1f}")
                c4.metric("Formation Energy", f"{cand['formation_energy']:.4f} eV/atom")

                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Thermal Cond.", f"{cand['thermal_conductivity']:.4f} W/mK")
                c6.metric("ZT", f"{cand['zt']:.6f}")
                c7.metric("Novelty", f"{cand['novelty_score']:.4f}")
                c8.metric("Recommendation", cand["recommendation"].split(" - ")[0])

        st.markdown("---")

        # Comparative radar chart - top 5
        st.markdown("### Multi-Dimensional Comparison (Top 5)")
        fig_compare = go.Figure()
        categories = ["TOPSIS", "Seebeck", "Power Factor", "Stability", "Novelty"]

        # Normalize values for radar
        max_seebeck = max(c["seebeck"] for c in rankings)
        max_pf = max(c["power_factor"] for c in rankings)
        min_fe = min(c["formation_energy"] for c in rankings)
        max_novelty = max(c["novelty_score"] for c in rankings)

        for cand in rankings[:5]:
            values = [
                cand["topsis_score"],
                cand["seebeck"] / max_seebeck if max_seebeck else 0,
                cand["power_factor"] / max_pf if max_pf else 0,
                abs(cand["formation_energy"] / min_fe) if min_fe else 0,
                cand["novelty_score"] / max_novelty if max_novelty else 0,
            ]
            fig_compare.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                name=f"#{cand['rank']} {cand['mxene_formula']}/{cand['composite_partner']}",
                fill="toself",
                opacity=0.5,
            ))
        fig_compare.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalized Multi-Criteria Comparison",
            height=550,
        )
        st.plotly_chart(fig_compare, width="stretch")

        # DFT validation summary
        if dft_results:
            st.markdown("### DFT-Validated Materials")
            for mat, data in dft_results.items():
                dos_info = data.get("dos", {})
                scf = data.get("scf", {})
                st.success(
                    f"**{mat}** - DFT Confirmed: "
                    f"{'Metallic' if dos_info.get('is_metallic') else 'Semiconductor'} | "
                    f"Bandgap: {dos_info.get('bandgap_ev', 'N/A')} eV | "
                    f"Fermi: {scf.get('fermi_energy_ev', 'N/A')} eV | "
                    f"Total Energy: {scf.get('total_energy_ev', 0):.2f} eV"
                )

        # Key findings
        st.markdown("### Key Findings")
        st.markdown("""
        1. **Mo2N-O/PEDOT:PSS** emerges as the top candidate (TOPSIS = 0.869) with the highest
           Seebeck coefficient (338.7 uV/K) among all screened composites.

        2. **Conducting polymer composites** (PEDOT:PSS) consistently outperform carbon material
           and metal nanowire composites for thermoelectric applications due to their favorable
           Seebeck-conductivity balance.

        3. **Mo-based MXenes** dominate the top rankings, with both carbide (Mo2C) and nitride
           (Mo2N) variants showing promise. Nitrides show higher Seebeck coefficients.

        4. **DFT validation of Mo2CO2** confirms metallic character (zero bandgap, DOS at Fermi
           level = 1.269 states/eV), consistent with literature reports on O-terminated Mo2C MXenes.

        5. All 20 candidates have **negative formation energies** (thermodynamically stable) per
           CHGNet prediction, with Ti2C-O showing the most negative value (-1.75 eV/atom).
        """)

        # Export section
        st.markdown("---")
        st.markdown("### Export Data")
        col1, col2 = st.columns(2)
        with col1:
            rank_df = pd.DataFrame(rankings)
            csv = rank_df.to_csv(index=False)
            st.download_button(
                "Download Rankings (CSV)",
                csv, "mxdiscovery_rankings.csv",
                mime="text/csv",
            )
        with col2:
            st.download_button(
                "Download Rankings (JSON)",
                json.dumps(rankings, indent=2),
                "mxdiscovery_rankings.json",
                mime="application/json",
            )
