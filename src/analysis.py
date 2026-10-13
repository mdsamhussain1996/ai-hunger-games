from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

from src.storage import Database


def compute_vote_matrix(
    db: Database,
    run_id: str,
    season_number: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compute the voter x votee square matrix W of vote counts."""
    votes = db.get_votes(run_id, season_number=season_number)
    contestants = db.get_contestants(run_id)

    # Get relevant contestants (active in that season or all in run)
    c_names = sorted(list(set(c["name"] for c in contestants)))
    df_w = pd.DataFrame(0, index=c_names, columns=c_names)

    for v in votes:
        voter = v["voter_name"]
        votee = v["votee_name"]
        if voter in df_w.index and votee in df_w.columns:
            df_w.loc[voter, votee] += 1

    return df_w, c_names


def compute_reciprocity_and_alliances(
    w_matrix: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compute reciprocity correlation and flag collusive pairs via binomial test."""
    matrix = w_matrix.values
    n = len(w_matrix)
    if n < 2:
        return {"reciprocity_corr": 0.0, "collusive_pairs": [], "graph": None}

    # Pearson correlation between W and W.T (flattened off-diagonal elements)
    off_diag_mask = ~np.eye(n, dtype=bool)
    w_flat = matrix[off_diag_mask]
    wt_flat = matrix.T[off_diag_mask]

    corr = 0.0
    if np.std(w_flat) > 0 and np.std(wt_flat) > 0:
        corr = float(np.corrcoef(w_flat, wt_flat)[0, 1])

    # Binomial test for collusion pairs (A -> B and B -> A)
    # Baseline null probability under uniform random voting: p0 = 1 / (n - 1)
    p0 = 1.0 / max(1, n - 1)
    collusive_pairs = []

    names = list(w_matrix.index)
    for i in range(n):
        for j in range(i + 1, n):
            name_a, name_b = names[i], names[j]
            votes_a_to_b = matrix[i, j]
            votes_b_to_a = matrix[j, i]
            total_votes_a = matrix[i].sum()
            total_votes_b = matrix[j].sum()

            if total_votes_a >= 3 and total_votes_b >= 3:
                # Binomial survival test
                p_val_a = stats.binom.sf(votes_a_to_b - 1, total_votes_a, p0)
                p_val_b = stats.binom.sf(votes_b_to_a - 1, total_votes_b, p0)
                joint_p = max(p_val_a, p_val_b)
                if joint_p < alpha and (votes_a_to_b + votes_b_to_a) >= 4:
                    collusive_pairs.append({
                        "pair": f"{name_a} ↔ {name_b}",
                        "a_to_b": int(votes_a_to_b),
                        "b_to_a": int(votes_b_to_a),
                        "p_value": float(joint_p),
                    })

    # Build NetworkX directed graph
    g = nx.DiGraph()
    for name in names:
        g.add_node(name)
    for i in range(n):
        for j in range(n):
            weight = matrix[i, j]
            if weight > 0 and i != j:
                g.add_edge(names[i], names[j], weight=int(weight))

    # Community detection (Greedy modularity on undirected projection)
    communities = []
    if g.number_of_edges() > 0:
        g_undir = g.to_undirected()
        try:
            communities = list(nx.community.greedy_modularity_communities(g_undir))
            communities = [sorted(list(c)) for c in communities]
        except Exception:
            communities = [[name] for name in names]

    return {
        "reciprocity_corr": corr,
        "collusive_pairs": collusive_pairs,
        "communities": communities,
        "graph": g,
    }


def compute_position_bias(db: Database, run_id: str) -> pd.DataFrame:
    """Compute vote share by displayed candidate label (A, B, C...)."""
    votes = db.get_votes(run_id)
    label_counts: Dict[str, int] = {}
    total = 0

    for v in votes:
        if not v["is_valid"]:
            continue
        lbl = v["vote_label"]
        if lbl and lbl != "ABSTAIN":
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            total += 1

    rows = []
    for lbl in sorted(label_counts.keys()):
        cnt = label_counts[lbl]
        pct = (cnt / total * 100) if total > 0 else 0.0
        rows.append({"label": lbl, "count": cnt, "percentage": round(pct, 2)})

    return pd.DataFrame(rows)


def compute_evolution_metrics(db: Database, run_id: str) -> Dict[str, Any]:
    """Compute score trends, lineage tree, and generation survival."""
    contestants = db.get_contestants(run_id)
    scores = db.get_scores(run_id)

    # Lineage links
    lineage_nodes = []
    lineage_edges = []
    for c in contestants:
        lineage_nodes.append({
            "id": c["id"],
            "name": c["name"],
            "gen": c["generation"],
            "strategy": c["replacement_strategy"],
            "status": c["status"],
        })
        parents = json.loads(c.get("parent_ids_json", "[]"))
        for p in parents:
            lineage_edges.append({"source": p, "target": c["id"]})

    # Scores DataFrame
    df_scores = pd.DataFrame(scores)

    return {
        "lineage_nodes": lineage_nodes,
        "lineage_edges": lineage_edges,
        "scores_df": df_scores,
        "contestants_df": pd.DataFrame(contestants),
    }


def compute_message_influence(db: Database, run_id: str) -> pd.DataFrame:
    """Analyze whether the recipient of a diplomacy message voted for the sender in the next round."""
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                m.round_id,
                m.season_number,
                m.sender_name,
                m.recipient_name,
                m.content,
                v.votee_name,
                (v.votee_name = m.sender_name) as recipient_voted_for_sender
            FROM diplomacy_messages m
            LEFT JOIN votes v ON v.round_id = m.round_id AND v.voter_name = m.recipient_name
            WHERE m.run_id = ?
            ORDER BY m.created_at
            """,
            (run_id,),
        )
        rows = [dict(r) for r in cursor.fetchall()]

    return pd.DataFrame(rows)
