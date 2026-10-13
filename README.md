# 🏹 AI Hunger Games: Evolutionary Tournament of Local LLM Personalities

An evolutionary tournament framework where 8 local LLM contestants—each running with distinct cognitive personalities—compete, debate, vote on each other's answers, form latent alliances, and face elimination and genetic evolution over multiple seasons.

All inference is executed locally through **Ollama** without cloud APIs. Every prompt, raw generation, vote matrix, and reasoning trace is persistently logged into **SQLite** for quantitative and behavioral analysis.

---

## ⚡ Features

- **8 Diverse Seed Personalities**:
  - *The Skeptic*, *The Poet*, *The Pragmatic Engineer*, *The Contrarian*, *The Diplomat*, *The Know-it-all*, *The Minimalist*, *The Showman*.
- **Position-Bias Canceling Voting Engine**:
  - In each round, contestants evaluate all peer answers with positions shuffled per voter ($A, B, C\dots$). Self-voting is strictly excluded.
  - Votes enforce structured JSON parsing (`{"vote": "...", "ranking": [...], "reason": "..."}`) with retry fallbacks.
- **Borda & Plurality Scoring**:
  - Dual scoring tracks both 1st-place plurality popularity and robust Borda positional points.
- **Evolutionary Replacement**:
  - At the conclusion of an 8-round season, the lowest-scoring contestant is eliminated. A replacement contestant is generated via **Random**, **Mutation**, or **Crossover** of top scorers, tracking multi-generational lineage.
- **Empirical Collusion & Alliance Detection**:
  - Square Vote Matrix $W$ heatmap (Voter $\times$ Votee).
  - Reciprocity correlation ($W$ vs $W^T$).
  - Binomial test ($p < 0.05$ vs uniform $1/(n-1)$ null hypothesis) to flag statistically significant voting alliances.
  - Directed graph community detection (greedy modularity) for coalition discovery.
  - Position bias audits measuring vote share across displayed letter positions.
- **Interactive Streamlit Dashboard**:
  - **Arena**: Step individual rounds or run full 8-round seasons.
  - **Leaderboard**: Season standings, Borda points, plurality votes, elimination history.
  - **Contestants**: Profile cards, prompts, temperature, and generation lineage.
  - **Analysis**: Interactive Plotly heatmaps, reciprocity metrics, and bias distributions.
  - **Experiments**: Multi-run config comparisons.

---

## 🛠 Tech Stack

- **Python 3.11+**
- **Ollama** (local inference via the `ollama` Python package)
- **SQLite** (persistent storage per experiment run)
- **Streamlit** (interactive dashboard)
- **Pydantic** (data schemas and structured LLM outputs)
- **Pandas & NumPy** (data processing and matrix operations)
- **NetworkX** (alliance graph clustering and modularity)
- **Plotly** (interactive charts and heatmaps)
- **Pytest** (automated test suite)

---

## 🚀 Quickstart Guide

### 1. Install & Start Ollama
Ensure [Ollama](https://ollama.com) is installed and running:
```bash
# Start the Ollama background daemon
ollama serve
```

Pull your preferred small models (e.g. 1B - 3B parameters):
```bash
ollama pull llama3.2:1b
ollama pull qwen2.5:0.5b
```

### 2. Set Up Python Environment
```bash
# Clone the repository
git clone https://github.com/mdsamhussain1996/ai-hunger-games.git
cd ai-hunger-games

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run a Round via CLI
Run a live tournament round printed directly to your console:
```bash
# Run with 3 contestants (or all 8)
python -m src.cli round --contestants 3

# Filter by question category (factual, ethics, creative, advice, opinion, reasoning/math)
python -m src.cli round --contestants 8 --category ethics
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser to interact with the visual Arena, Leaderboard, and Analysis suite.

---

## 📊 Explanation of Metrics

### 1. Borda Score vs. Plurality Score
- **Plurality**: Number of times a contestant's answer was selected as the single best choice (Rank 1).
- **Borda**: With $K$ candidate answers, a ballot gives $K - 1$ points to Rank 1, $K - 2$ to Rank 2, down to $0$ points to the bottom rank. Borda scoring stabilizes rankings against vote splitting.

### 2. Vote Matrix $W$
- A square matrix where cell $W_{i, j}$ represents the count of votes cast by voter $i$ in favor of contestant $j$.
- Diagonal entries $W_{i, i} = 0$ since self-voting is impossible.

### 3. Reciprocity Correlation
- The Pearson correlation coefficient between the off-diagonal elements of $W$ and $W^T$:
  $$\rho(W, W^T)$$
- Positive correlation indicates reciprocal voting ("I vote for you if you vote for me").

### 4. Collusion Detection (Binomial Test)
- For any pair of contestants $(A, B)$, under the null hypothesis of unbiased voting, the probability of voting for any specific peer among $n-1$ alternatives is $p_0 = \frac{1}{n - 1}$.
- We perform a one-tailed binomial survival test on mutual vote counts:
  $$P(X \ge k) = \sum_{x=k}^N \binom{N}{x} p_0^x (1 - p_0)^{N - x}$$
- Pairs with joint $p < 0.05$ and elevated mutual vote counts are flagged as collusive voting blocs.

### 5. Position Bias
- Calculates the percentage of total votes received by candidate label $A, B, C, D\dots$.
- Because candidate positions are shuffled per voter, deviation from a uniform distribution indicates underlying label position preference.

---

## 🧪 Running Tests

Execute the automated test suite with:
```bash
pytest
```

---

## 📜 License

MIT License. Designed and developed for open AI behavioral and evolutionary research.
