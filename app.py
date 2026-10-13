from __future__ import annotations

import json
import os
import random
import yaml
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analysis import (
    compute_evolution_metrics,
    compute_position_bias,
    compute_reciprocity_and_alliances,
    compute_vote_matrix,
)
from src.contestants import load_seed_personalities
from src.game import QuestionBank, TournamentEngine, run_single_round
from src.llm import LLMClient
from src.models import AppConfig, Question
from src.storage import Database

# Page config
st.set_page_config(
    page_title="AI Hunger Games",
    page_icon="🏹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F6B4F;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #59636B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #DDE5E0;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    .contestant-card {
        border: 1px solid #DDE5E0;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: #FFFFFF;
    }
    .answer-box {
        background-color: #F7F9F8;
        border-left: 4px solid #0F6B4F;
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_db():
    return Database("data/hunger_games.db")


@st.cache_resource
def get_config():
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return AppConfig.model_validate(cfg)
    except Exception:
        return AppConfig()


db = get_db()
config = get_config()

# Sidebar Setup
st.sidebar.markdown("## 🏹 AI Hunger Games")
st.sidebar.caption("Evolutionary Tournament of LLM Personalities")

# Inference Mode Selector
st.sidebar.markdown("### ⚙️ Inference Engine")
engine_mode = st.sidebar.selectbox(
    "Select Engine",
    ["Auto-Detect (Ollama / Fallback)", "Persona Simulation Engine (Cloud)", "Custom Ollama Host"],
    index=0,
)

ollama_host = config.models.ollama_host
force_mock = False

if engine_mode == "Custom Ollama Host":
    ollama_host = st.sidebar.text_input("Ollama Host URL", value=config.models.ollama_host)
elif engine_mode == "Persona Simulation Engine (Cloud)":
    force_mock = True

llm = LLMClient(
    host=ollama_host,
    timeout_seconds=config.models.timeout_seconds,
    max_retries=config.models.max_retries,
    force_mock=force_mock,
)

# Initialize session state
if "run_id" not in st.session_state:
    st.session_state.run_id = config.general.run_id

if "engine" not in st.session_state:
    engine = TournamentEngine(config=config, llm=llm, db=db, run_id=st.session_state.run_id)
    seeds = load_seed_personalities("personalities/seed.yaml", default_model=config.models.default_model)
    engine.initialize_contestants(seeds)

    # If database is fresh and has no rounds, seed initial demo rounds for immediate display
    existing_rounds = db.get_rounds(st.session_state.run_id)
    if not existing_rounds:
        for _ in range(2):
            s = engine.play_round()
            st.session_state.last_summary = s

    st.session_state.engine = engine

engine: TournamentEngine = st.session_state.engine
engine.llm = llm  # Update LLM in case settings changed

nav = st.sidebar.radio(
    "Navigation",
    ["1. Arena", "2. Leaderboard", "3. Contestants", "4. Analysis", "5. Experiments"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Run ID:** `{st.session_state.run_id}`")
st.sidebar.markdown(f"**Season:** {engine.current_season} · **Round:** {engine.current_round} / {config.game.rounds_per_season}")
st.sidebar.markdown(f"**Base Model:** `{config.models.default_model}`")

ollama_online = llm.is_ollama_available()
if ollama_online and not force_mock:
    st.sidebar.success(f"Ollama: Connected ({ollama_host})")
else:
    st.sidebar.info("Cloud Engine: Persona Simulation Active")


# ==========================================
# PAGE 1: ARENA
# ==========================================
if nav == "1. Arena":
    st.markdown('<div class="main-header">🏹 Tournament Arena</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sub-header">Season {engine.current_season} · Round {engine.current_round + 1} of {config.game.rounds_per_season}</div>',
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1.5, 1.5, 3])
    with col_btn1:
        step_round = st.button("▶ Step Single Round", use_container_width=True, type="primary")
    with col_btn2:
        step_season = st.button("⏩ Complete Season (8 Rounds)", use_container_width=True)

    if step_round:
        with st.spinner("Contestants are answering and evaluating..."):
            summary = engine.play_round()
            st.session_state.last_summary = summary
            if engine.current_round >= config.game.rounds_per_season:
                elim, repl = engine.resolve_season_elimination()
                st.session_state.last_elimination = (elim, repl)

    if step_season:
        with st.spinner("Running full season simulation..."):
            remaining = config.game.rounds_per_season - engine.current_round
            for _ in range(max(1, remaining)):
                summary = engine.play_round()
                st.session_state.last_summary = summary
            elim, repl = engine.resolve_season_elimination()
            st.session_state.last_elimination = (elim, repl)
            st.rerun()

    if "last_elimination" in st.session_state:
        elim, repl = st.session_state.last_elimination
        st.error(f"☠️ **Season Elimination:** **{elim.name}** was eliminated! Replaced by evolved offspring: **{repl.name}** ({repl.replacement_strategy.upper()}).")

    # Display most recent round
    if "last_summary" in st.session_state:
        summary = st.session_state.last_summary
        st.markdown("---")
        st.subheader(f"Round Question: [{summary.question.category.upper()}]")
        st.info(f"**\"{summary.question.prompt}\"**")

        tab_ans, tab_votes, tab_score = st.tabs(["Contestant Answers", "Voting & Rationale", "Round Scoreboard"])

        with tab_ans:
            for ans in summary.answers:
                st.markdown(f"**{ans.contestant_name}**:")
                st.markdown(f'<div class="answer-box">"{ans.text}"</div>', unsafe_allow_html=True)

        with tab_votes:
            for v in summary.votes:
                votee_text = f"**{v.votee_name}** (Label {v.vote_label})" if v.votee_name else "ABSTAIN"
                with st.expander(f"Voter: {v.voter_name} → Voted for {votee_text}"):
                    st.write(f"**Full Ranking:** {' > '.join(v.ranking_labels)}")
                    st.write(f"**Reasoning:** {v.reason}")

        with tab_score:
            df_round = pd.DataFrame([s.model_dump() for s in summary.scores])
            st.dataframe(df_round[["contestant_name", "borda_points", "plurality_points", "total_points"]], use_container_width=True)
    else:
        st.info("Click **'▶ Step Single Round'** above to run the next round and watch contestants debate and vote!")


# ==========================================
# PAGE 2: LEADERBOARD
# ==========================================
elif nav == "2. Leaderboard":
    st.markdown('<div class="main-header">🏆 Tournament Leaderboard</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Cumulative scores for Season {engine.current_season}</div>', unsafe_allow_html=True)

    scores_data = db.get_cumulative_season_scores(st.session_state.run_id, engine.current_season)
    if scores_data:
        df_scores = pd.DataFrame(scores_data)
        st.dataframe(df_scores, use_container_width=True)

        fig = px.bar(
            df_scores,
            x="contestant_name",
            y="cumulative_points",
            color="contestant_name",
            title=f"Cumulative Season {engine.current_season} Points",
            labels={"contestant_name": "Contestant", "cumulative_points": "Total Points"},
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No rounds have been completed yet in Season {engine.current_season}.")

    st.markdown("### 📜 Elimination History")
    seasons = db.get_seasons(st.session_state.run_id)
    if seasons:
        st.dataframe(pd.DataFrame(seasons), use_container_width=True)
    else:
        st.write("No contestants have been eliminated yet.")


# ==========================================
# PAGE 3: CONTESTANTS
# ==========================================
elif nav == "3. Contestants":
    st.markdown('<div class="main-header">🧬 Contestant Roster & Profiles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Personalities, traits, generation lineage, and judging philosophy</div>', unsafe_allow_html=True)

    contestants = db.get_contestants(st.session_state.run_id)
    col1, col2 = st.columns(2)
    for idx, c in enumerate(contestants):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            status_badge = "🟢 ACTIVE" if c["status"] == "active" else "🔴 ELIMINATED"
            parents = json.loads(c.get("parent_ids_json", "[]"))
            parent_str = ", ".join(parents) if parents else "Original Seed"
            with st.container():
                st.markdown(
                    f"""
                    <div class="contestant-card">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <h3 style="margin:0;">{c['name']}</h3>
                            <span style="font-weight:700;font-size:0.85rem;">{status_badge}</span>
                        </div>
                        <div style="font-size:0.85rem;color:#59636B;margin:0.5rem 0;">
                            <strong>Generation:</strong> {c['generation']} · <strong>Strategy:</strong> {c['replacement_strategy']} · <strong>Lineage:</strong> {parent_str}
                        </div>
                        <div style="font-size:0.9rem;line-height:1.5;background:#F7F9F8;padding:0.75rem;border-radius:4px;border:1px solid #DDE5E0;">
                            {c['system_prompt']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ==========================================
# PAGE 4: ANALYSIS
# ==========================================
elif nav == "4. Analysis":
    st.markdown('<div class="main-header">📊 Behavioral & Evolutionary Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Voting reciprocity, alliance detection, position bias, and score dynamics</div>', unsafe_allow_html=True)

    # 1. Vote Matrix Heatmap
    st.subheader("1. Vote Matrix W (Voter × Votee)")
    w_df, c_names = compute_vote_matrix(db, st.session_state.run_id)
    if not w_df.empty and w_df.values.sum() > 0:
        fig_heat = px.imshow(
            w_df,
            labels=dict(x="Votee (Recipient)", y="Voter (Caster)", color="Votes"),
            x=w_df.columns,
            y=w_df.index,
            color_continuous_scale="Viridis",
            text_auto=True,
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # 2. Reciprocity & Alliances
        st.subheader("2. Reciprocity & Collusion Detection")
        analysis_res = compute_reciprocity_and_alliances(w_df)
        st.metric("Reciprocity Correlation (W vs Wᵀ)", f"{analysis_res['reciprocity_corr']:.3f}")

        if analysis_res["collusive_pairs"]:
            st.warning("🚨 Potential Collusive Pairs Detected (Binomial Test p < 0.05):")
            st.dataframe(pd.DataFrame(analysis_res["collusive_pairs"]))
        else:
            st.success("No statistically significant collusive voting pairs detected yet.")

        if analysis_res["communities"]:
            st.write("**Detected Voting Communities (Modularity):**")
            for i, comm in enumerate(analysis_res["communities"], 1):
                st.write(f"- **Community {i}:** {', '.join(comm)}")
    else:
        st.info("Play at least one round in the Arena to view voting matrices.")

    # 3. Position Bias
    st.subheader("3. Position Bias Audit")
    pos_df = compute_position_bias(db, st.session_state.run_id)
    if not pos_df.empty:
        fig_pos = px.bar(
            pos_df,
            x="label",
            y="percentage",
            title="Vote Share by Displayed Label Position",
            labels={"label": "Candidate Label (A, B, C...)", "percentage": "Vote Share (%)"},
            text="percentage",
        )
        st.plotly_chart(fig_pos, use_container_width=True)


# ==========================================
# PAGE 5: EXPERIMENTS
# ==========================================
elif nav == "5. Experiments":
    st.markdown('<div class="main-header">🔬 Experiment Run Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Side-by-side run comparisons across configurations</div>', unsafe_allow_html=True)

    runs = db.get_runs()
    if runs:
        df_runs = pd.DataFrame(runs)
        st.dataframe(df_runs[["id", "seed", "status", "created_at"]], use_container_width=True)
    else:
        st.info("No runs recorded in database.")
