#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute the number of parallel rounds needed so that:
  1) Every cyclic SCC is iterated (1 + settling_count) times (counting *nodes*, not edges),
  2) The influence can reach at least one output node.

Designed for small graphs (e.g., 10–100 nodes). O(V+E) per rewire; instant in practice.
"""

import numpy as np
from collections import deque


# ========= One-time build when wiring changes =========

def build_graph_cache(adj: np.ndarray, n_inputs: int, n_outputs: int):
    """
    Build and cache SCC condensation + DAG + static metadata.
    Inputs are nodes [0 .. n_inputs-1], outputs are the last n_outputs nodes.

    Returns a dict cache usable by rounds_from_cache().
    """
    assert adj.ndim == 2 and adj.shape[0] == adj.shape[1], "adj must be square"
    n = adj.shape[0]

    # Adjacency lists (neighbors) and self-loop flags
    nbrs = [np.flatnonzero(adj[i]).tolist() for i in range(n)]
    self_loop = [bool(adj[i, i]) for i in range(n)]

    # --- Tarjan SCC on adjacency lists (O(V+E)) ---
    index = 0
    stack, on_stack = [], [False] * n
    indices, low = [-1] * n, [0] * n
    sccs = []

    def strongconnect(v):
        nonlocal index
        indices[v] = low[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True
        for w in nbrs[v]:
            if indices[w] == -1:
                strongconnect(w)
                if low[w] < low[v]:
                    low[v] = low[w]
            elif on_stack[w]:
                if indices[w] < low[v]:
                    low[v] = indices[w]
        if low[v] == indices[v]:
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == v:
                    break
            sccs.append(comp)

    for v in range(n):
        if indices[v] == -1:
            strongconnect(v)

    comp_id = {u: i for i, comp in enumerate(sccs) for u in comp}
    m = len(sccs)

    # Condensed DAG
    dag = [set() for _ in range(m)]
    for u, outs in enumerate(nbrs):
        cu = comp_id[u]
        for v in outs:
            cv = comp_id[v]
            if cu != cv:
                dag[cu].add(cv)
    dag = [list(s) for s in dag]

    # Static metadata
    is_input = np.zeros(n, dtype=bool)
    is_input[:n_inputs] = True
    outputs = list(range(n - n_outputs, n))
    in_comps = {comp_id[i] for i in range(n_inputs)}
    out_comps = {comp_id[o] for o in outputs}

    # For each SCC: count non-input nodes and detect cyclicity
    non_input_count = np.zeros(m, dtype=int)
    is_cyclic = np.zeros(m, dtype=bool)
    for i, comp in enumerate(sccs):
        non_input_count[i] = sum(not is_input[u] for u in comp)
        is_cyclic[i] = (len(comp) > 1) or any(self_loop[u] for u in comp)

    return {
        "sccs": sccs,
        "dag": dag,
        "comp_id": comp_id,
        "in_comps": in_comps,
        "out_comps": out_comps,
        "non_input_count": non_input_count,
        "is_cyclic": is_cyclic,
    }


# ========= Fast query when settling_count or policy changes =========

def rounds_from_cache(cache, settling_count: int, count_output_tick: bool = True):
    """
    Given a cache (from build_graph_cache), compute the # of parallel rounds.

    Semantics:
      - Inputs are 'ready at round 0' (no cost).
      - Acyclic SCC weight = number of non-input nodes in the SCC.
      - Cyclic SCC weight = (#non-input nodes) * (1 + settling_count).
      - If count_output_tick=False, we *subtract 1* iff the terminal SCC is an acyclic
        singleton non-input (typical case for a lone output node).
    """
    sccs = cache["sccs"]
    dag = cache["dag"]
    in_c = cache["in_comps"]
    out_c = cache["out_comps"]
    k = cache["non_input_count"]
    cyc = cache["is_cyclic"]

    m = len(sccs)

    # Weights per SCC (count nodes, not edges)
    w = k * (1 + settling_count)
    w = np.where(cyc, w, k).astype(int)

    # Topological order (Kahn)
    indeg = [0] * m
    for u in range(m):
        for v in dag[u]:
            indeg[v] += 1
    q = deque([i for i in range(m) if indeg[i] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in dag[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    NEG = -10**9
    dp = [NEG] * m
    prev = [-1] * m

    # Start only at SCCs containing an input
    for ci in in_c:
        dp[ci] = w[ci]

    for u in topo:
        if dp[u] == NEG:
            continue
        for v in dag[u]:
            cand = dp[u] + w[v]
            if cand > dp[v]:
                dp[v] = cand
                prev[v] = u

    # Best terminal among output SCCs
    best_end, best_val = -1, NEG
    for oc in out_c:
        if dp[oc] > best_val:
            best_val = dp[oc]
            best_end = oc

    if best_end == -1:
        return None, {"reason": "No input→output path."}

    # Optionally do not count the final output's tick
    if not count_output_tick:
        if (not cyc[best_end]) and (w[best_end] == 1):
            best_val -= 1

    # Reconstruct component path (for debug)
    path = []
    cur = best_end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path = path[::-1]

    dbg = {
        "comp_path_indices": path,
        "comp_path_nodes_1_based": [[u + 1 for u in sccs[i]] for i in path],
        "weights": w.tolist(),
        "is_cyclic": cyc.tolist(),
    }
    return int(best_val), dbg


# ========= Convenience wrapper =========

def compute_rounds(adj: np.ndarray,
                   n_inputs: int,
                   n_outputs: int,
                   settling_count: int,
                   count_output_tick: bool = True):
    cache = build_graph_cache(adj, n_inputs, n_outputs)
    return rounds_from_cache(cache, settling_count, count_output_tick)


# ========= Demo / Test =========

if __name__ == "__main__":
    # Your sample adjacency (17 nodes). True = edge i -> j.
    adj_matrix = np.array([
        [False, False, False, False, False, True,  False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, True,  False, False, False],
        [False, False, False, False, False, False, True,  False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, True,  False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, True,  False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, True,  False, False, True ],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, True,  False, False, False, True,  False, True,  False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True,  False],
        [False, False, False, False, False, True,  False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True,  False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False],
        [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    ], dtype=bool)

    n_inputs = 5
    n_outputs = 3

    # Build cache once per rewire
    cache = build_graph_cache(adj_matrix, n_inputs, n_outputs)

    # Query quickly for various settling counts and policies
    for s in [1, 3, 4]:
        r_true, dbg_true = rounds_from_cache(cache, s, count_output_tick=True)
        r_fast, dbg_fast = rounds_from_cache(cache, s, count_output_tick=False)
        print(f"\nsettling_count={s}:")
        print(f"  count_output_tick=True  -> rounds = {r_true}")
        print(f"  count_output_tick=False -> rounds = {r_fast}")
        print(f"  critical SCC path (1-based nodes): {dbg_true['comp_path_nodes_1_based']}")
        cyc_sccs = [comp for comp, cyc in zip(
            [[u + 1 for u in comp] for comp in cache['sccs']],
            cache['is_cyclic'].tolist()
        ) if cyc]
        print(f"  cycles found (1-based): {cyc_sccs}")
