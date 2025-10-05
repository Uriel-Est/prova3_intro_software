# -*- coding: utf-8 -*-
"""
Monte Carlo de π — Python puro vs Numba (serial e paralelo) + animação + benchmark Markdown
------------------------------------------------------------------------------------------
O que este script faz:
1) Estima π por Monte Carlo:
   - Python puro (baseline)
   - Numba serial (njit)
   - Numba paralelo (prange)
   Com barra de progresso em blocos e IC 95%.
2) Gera um GIF animado com:
   - Painel esquerdo: pontos (x,y) no quadrante [0,1]^2 com 1/4 de círculo
   - Painel direito: curva de convergência de π̂ ao longo das amostras
3) Benchmark (gera tabela em Markdown para README)

Saídas:
- GIF e PNG da animação salvos na mesma pasta do .py
- Arquivo Markdown com os resultados de benchmark (bench_pi_results.md por padrão)
"""

from __future__ import annotations

# =============================================================================
# 1) IMPORTS & CONFIG
# =============================================================================
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

from statistics import median

import numpy as np

# Matplotlib sem GUI para salvar GIF/PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge

# Numba (a atividade pede Numba)
from numba import njit, prange

# Progresso (opcional)
try:
    from tqdm import tqdm
    HAVE_TQDM = True
except Exception:
    HAVE_TQDM = False

# Base dir: mesma pasta do script (.py)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()


# =============================================================================
# 2) PYTHON PURO — baseline (sem Numba)
# =============================================================================
def pi_monte_carlo_py(n: int, seed: Optional[int] = None) -> float:
    """
    Estima π por Monte Carlo em Python puro: amostra n pontos (x,y) ~ U(0,1)^2
    e estima π = 4 * P(x^2 + y^2 ≤ 1). Custo O(n).
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n deve ser int positivo")
    import random
    if seed is not None:
        random.seed(seed)
    hits = 0
    for _ in range(n):
        x = random.random()
        y = random.random()
        if x*x + y*y <= 1.0:
            hits += 1
    return 4.0 * (hits / n)


# =============================================================================
# 3) NÚCLEOS NUMBA — serial e paralelo
# =============================================================================
@njit(fastmath=True, cache=True)
def mc_hits_block(n: int, seed: int) -> int:
    """
    Núcleo NUMBA (serial): retorna o número de acertos no quarto de círculo.
    """
    np.random.seed(seed)
    hits = 0
    for _ in range(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            hits += 1
    return hits


@njit(fastmath=True, parallel=True, cache=True)
def mc_hits_block_par(n: int, seed: int) -> int:
    """
    Núcleo NUMBA (paralelo): usa prange para distribuir o laço.
    """
    np.random.seed(seed)
    hits = 0
    for i in prange(n):
        x = np.random.random()
        y = np.random.random()
        if x*x + y*y <= 1.0:
            hits += 1
    return hits


# Contadores jitados para arrays (usados na animação)
@njit(fastmath=True, cache=True)
def count_hits_xy_nb(x: np.ndarray, y: np.ndarray) -> int:
    hits = 0
    for i in range(x.size):
        if x[i]*x[i] + y[i]*y[i] <= 1.0:
            hits += 1
    return hits

@njit(fastmath=True, parallel=True, cache=True)
def count_hits_xy_nb_par(x: np.ndarray, y: np.ndarray) -> int:
    hits = 0
    for i in prange(x.size):
        if x[i]*x[i] + y[i]*y[i] <= 1.0:
            hits += 1
    return hits


# =============================================================================
# 4) ESTIMATIVA EM BLOCOS + IC 95%
# =============================================================================
@dataclass
class PiEstimate:
    n_total: int
    pi_hat: float
    se: float
    ci95: Tuple[float, float]
    elapsed_s: float
    method: str  # "py" | "numba" | "numba_par"

def _stderr_and_ci(pi_hat: float, n: int) -> Tuple[float, Tuple[float, float]]:
    """
    EP(π̂) ≈ 4 * sqrt(p̂ (1 - p̂) / n), p̂=π̂/4; IC95% ≈ π̂ ± 1.96*EP
    """
    p_hat = pi_hat / 4.0
    se = 4.0 * math.sqrt(max(p_hat * (1.0 - p_hat), 0.0) / n)
    return se, (pi_hat - 1.96*se, pi_hat + 1.96*se)

def estimate_pi(n: int,
                seed: int = 123456,
                backend: str = "numba",      # "py" | "numba" | "numba_par"
                block_size: int = 1_000_000,
                show_progress: bool = True) -> PiEstimate:
    """
    Estima π consumindo N amostras em blocos (para acompanhar andamento).
    backends:
      - "py":        Python puro (mais lento)
      - "numba":     kernel serial jitado
      - "numba_par": kernel paralelo jitado
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n deve ser int positivo")
    if block_size <= 0:
        raise ValueError("block_size deve ser positivo")

    if backend == "py":
        def run_block(bs: int, s: int) -> int:
            import random
            random.seed(s)
            h = 0
            for _ in range(bs):
                x = random.random(); y = random.random()
                if x*x + y*y <= 1.0:
                    h += 1
            return h
        method = "py"
    elif backend == "numba":
        run_block = mc_hits_block
        method = "numba"
    elif backend == "numba_par":
        run_block = mc_hits_block_par
        method = "numba_par"
    else:
        raise ValueError("backend deve ser 'py', 'numba' ou 'numba_par'")

    n_blocks = (n + block_size - 1) // block_size
    total_hits = 0
    t0 = time.perf_counter()

    iterator = range(n_blocks)
    if show_progress and HAVE_TQDM:
        iterator = tqdm(iterator, total=n_blocks, desc=f"Monte Carlo π [{method}]")

    for b in iterator:
        start = b * block_size
        end = min((b + 1) * block_size, n)
        bs = end - start
        block_seed = seed + 9973*b          # semente distinta por bloco (determinística)
        total_hits += run_block(bs, block_seed)

    elapsed = time.perf_counter() - t0
    pi_hat = 4.0 * (total_hits / n)
    se, ci = _stderr_and_ci(pi_hat, n)
    return PiEstimate(n_total=n, pi_hat=pi_hat, se=se, ci95=ci, elapsed_s=elapsed, method=method)


# =============================================================================
# 5) ANIMAÇÃO — pontos + curva de convergência
# =============================================================================
def animate_pi_gif(n_frames: int = 200,
                   batch_size: int = 20_000,
                   seed: int = 123,
                   use_parallel: bool = True,
                   scatter_max: int = 4000,
                   outfile: str = "pi_montecarlo.gif",
                   visual_mode: str = "grow_then_shuffle",
                   grow_ratio: float = 0.5) -> Path:
    """
    Gera um GIF animado com dois painéis:
      • esquerda: pontos (x,y) ~ U(0,1)^2 e arco do 1/4 de círculo;
      • direita: π̂(N) ao longo das amostras, com linha de referência em π.

    N total usado na estimativa = n_frames * batch_size.
    visual_mode:
      - "accumulate": mantém só os últimos 'scatter_max' pontos (comportamento clássico).
      - "grow_then_shuffle": primeiro 'cresce' a nuvem até 'scatter_max', depois
        realoca tudo aleatoriamente a cada frame (efeito de “multiplicar e movimentar”).
    grow_ratio: fração inicial dos frames dedicada ao crescimento (0–1).
    """
    rng = np.random.default_rng(seed)
    total_hits = 0
    total_n = 0

    # Figura com 2 painéis
    fig = plt.figure(figsize=(10.5, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.25)
    ax_sc = fig.add_subplot(gs[0, 0])
    ax_ln = fig.add_subplot(gs[0, 1])

    # Painel esquerdo: quadrado [0,1]x[0,1] + 1/4 de círculo
    ax_sc.set_aspect("equal")
    ax_sc.set_xlim(0, 1); ax_sc.set_ylim(0, 1)
    wedge = Wedge((0, 0), 1.0, 0, 90, width=None, fill=False, lw=1.5)
    ax_sc.add_patch(wedge)
    sc_in = ax_sc.scatter([], [], s=8, alpha=0.85, label="dentro")
    sc_out = ax_sc.scatter([], [], s=8, alpha=0.5, label="fora")
    ax_sc.legend(loc="lower left", frameon=False)
    title_sc = ax_sc.set_title("Monte Carlo de π — Pontos")

    # Painel direito: curva π̂(N)
    ax_ln.set_xlim(0, n_frames * batch_size)
    ax_ln.set_ylim(3.05, 3.25)  # ajustado dinamicamente
    (line_hat,) = ax_ln.plot([], [], lw=2, label="π̂")
    ax_ln.axhline(math.pi, lw=1.2, linestyle="--", label="π (referência)")
    ax_ln.set_xlabel("Amostras acumuladas (N)")
    ax_ln.set_ylabel("Estimativa de π")
    ax_ln.legend(loc="upper right")
    title_ln = ax_ln.set_title("Convergência de π̂")

    xs: List[int] = []
    ys: List[float] = []

    # Buffers visuais
    inside_xy = np.empty((0, 2), dtype=np.float64)
    outside_xy = np.empty((0, 2), dtype=np.float64)

    # Contador rápido (Numba) consistente com os batches estatísticos
    counter = count_hits_xy_nb_par if use_parallel else count_hits_xy_nb

    # Planejamento da fase de crescimento vs shuffle
    grow_frames = max(1, int(n_frames * max(0.0, min(1.0, grow_ratio))))
    mode = visual_mode.lower()

    def _append_limited(arr: np.ndarray, pts: np.ndarray, limit: int) -> np.ndarray:
        if pts.size == 0:
            return arr
        arr = np.vstack([arr, pts])
        if arr.shape[0] > limit:
            arr = arr[-limit:]  # mantém só os últimos
        return arr

    def update(frame: int):
        nonlocal total_hits, total_n, inside_xy, outside_xy

        # === (A) Estatística: gera novo lote para estimar π̂ (independente do visual) ===
        xb = rng.random(batch_size)
        yb = rng.random(batch_size)
        hits = counter(xb, yb)
        total_hits += hits
        total_n += batch_size
        pi_hat = 4.0 * (total_hits / total_n)
        xs.append(total_n); ys.append(pi_hat)
        line_hat.set_data(xs, ys)

        # ajusta eixo Y da curva conforme os valores
        ymin = min(min(ys), math.pi) - 0.02
        ymax = max(max(ys), math.pi) + 0.02
        ax_ln.set_ylim(ymin, ymax)

        # === (B) Visual: pontos surgindo / realocando ===
        if mode == "accumulate":
            # Mostra só os últimos 'scatter_max' pontos (comportamento clássico)
            mask_in = (xb*xb + yb*yb) <= 1.0
            new_in = np.column_stack([xb[mask_in], yb[mask_in]])
            new_out = np.column_stack([xb[~mask_in], yb[~mask_in]])
            inside_xy  = _append_limited(inside_xy, new_in, scatter_max)
            outside_xy = _append_limited(outside_xy, new_out, scatter_max)

        elif mode == "grow_then_shuffle":
            # Fase 1 (crescimento): a nuvem "multiplica"
            if frame < grow_frames:
                # alvo de pontos visuais (cresce de 0 até scatter_max)
                display_target = int(np.ceil(scatter_max * (frame + 1) / grow_frames))
                display_target = max(50, min(scatter_max, display_target))

                need = display_target - (inside_xy.shape[0] + outside_xy.shape[0])
                if need > 0:
                    x = rng.random(need); y = rng.random(need)
                    mask_in = (x*x + y*y) <= 1.0
                    inside_xy  = _append_limited(inside_xy,  np.column_stack([x[mask_in],  y[mask_in]]),  display_target)
                    outside_xy = _append_limited(outside_xy, np.column_stack([x[~mask_in], y[~mask_in]]), display_target)
                # senão, já atingiu o alvo; apenas mantém
            else:
                # Fase 2 (shuffle): realoca todos os pontos a cada frame
                x = rng.random(scatter_max); y = rng.random(scatter_max)
                mask_in = (x*x + y*y) <= 1.0
                inside_xy  = np.column_stack([x[mask_in],  y[mask_in]])
                outside_xy = np.column_stack([x[~mask_in], y[~mask_in]])

        else:
            # Fallback: mesmo que "accumulate"
            mask_in = (xb*xb + yb*yb) <= 1.0
            inside_xy  = _append_limited(inside_xy,  np.column_stack([xb[mask_in],  yb[mask_in]]),  scatter_max)
            outside_xy = _append_limited(outside_xy, np.column_stack([xb[~mask_in], yb[~mask_in]]), scatter_max)

        # aplica nos artists
        sc_in.set_offsets(inside_xy)
        sc_out.set_offsets(outside_xy)

        # títulos dinâmicos
        if mode == "grow_then_shuffle" and frame < grow_frames:
            fase_txt = "Crescimento"
        elif mode == "grow_then_shuffle":
            fase_txt = "Realocação"
        else:
            fase_txt = "Acúmulo"

        title_sc.set_text(f"Monte Carlo de π — Pontos ({fase_txt}, N={total_n:,})")
        title_ln.set_text(f"Convergência — π̂={pi_hat:.6f} (N={total_n:,})")

        return (sc_in, sc_out, line_hat, title_sc, title_ln)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=60, blit=False)
    out_path = BASE_DIR / outfile
    ani.save(str(out_path), writer="pillow", fps=20, dpi=110)
    fig.savefig(str(out_path.with_suffix(".png")), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"GIF salvo: {out_path.name} (+ PNG)")
    return out_path


# =============================================================================
# 6) BENCHMARK → Gera Markdown p/ README
# =============================================================================
def bench_pi(N: int = 5_000_000,
             block_size: int = 500_000,
             repeats: int = 3,
             seed: int = 123456,
             save_md: bool = True,
             md_filename: str = "bench_pi_results.md"):
    """
    Compara 'py' x 'numba' x 'numba_par' para o mesmo N.
    - Faz warm-up p/ não contar compilação JIT.
    - Usa mediana de 'repeats' execuções.
    - Imprime e salva um snippet Markdown.
    """
    if N <= 0 or block_size <= 0 or repeats <= 0:
        raise ValueError("params inválidos")

    # --- WARM-UP: compila kernels Numba antes do cronômetro ---
    try:
        _ = estimate_pi(100_000, backend="numba",     block_size=50_000, show_progress=False, seed=seed)
        _ = estimate_pi(100_000, backend="numba_par", block_size=50_000, show_progress=False, seed=seed)
    except Exception:
        pass

    backends = ["py", "numba", "numba_par"]
    results = []

    for be in backends:
        times = []
        last_est = None
        for r in range(repeats):
            est = estimate_pi(N, backend=be, block_size=block_size,
                              show_progress=False, seed=seed + r)
            times.append(est.elapsed_s)
            last_est = est
        t_med = median(times)
        results.append({
            "backend": be,
            "N": N,
            "block_size": block_size,
            "time_s": t_med,
            "pi_hat": last_est.pi_hat,
            "ci_low": last_est.ci95[0],
            "ci_high": last_est.ci95[1],
        })

    # Speedup vs Python
    t_py = next(x["time_s"] for x in results if x["backend"] == "py")
    for x in results:
        x["speedup_vs_py"] = t_py / x["time_s"]

    # (opcional) quantos threads o Numba usou
    try:
        import numba as nb
        num_threads = nb.get_num_threads()
    except Exception:
        num_threads = None

    # --- Markdown prontinho p/ README ---
    md_lines = []
    md_lines.append(f"### Benchmark Monte Carlo de π  \nN = {N:,} • block_size = {block_size:,} • repeats = {repeats}")
    if num_threads:
        md_lines.append(f"*Numba threads*: **{num_threads}**")
    md_lines.append("")
    md_lines.append("| Backend | Tempo (s) | Aceleração vs Python | π̂ | IC95% |")
    md_lines.append("|---|---:|---:|---:|---:|")
    for x in results:
        md_lines.append(
            f"| {x['backend']} | {x['time_s']:.3f} | {x['speedup_vs_py']:.2f}× | "
            f"{x['pi_hat']:.6f} | [{x['ci_low']:.6f}, {x['ci_high']:.6f}] |"
        )
    md_text = "\n".join(md_lines)

    # imprime e salva
    print("\n" + md_text + "\n")
    if save_md:
        out = (BASE_DIR if 'BASE_DIR' in globals() else Path.cwd()) / md_filename
        out.write_text(md_text, encoding="utf-8")
        print(f"→ Benchmark salvo em: {out}")

    return results, md_text


# =============================================================================
# 7) Execução de exemplo
# =============================================================================
if __name__ == "__main__":
    # 1) Estimativa com Numba paralela (com progresso)
    N = 1_000_000
    est = estimate_pi(N, backend="numba_par", block_size=200_000, show_progress=True)
    print(f"\nπ̂ = {est.pi_hat:.6f}  ± {1.96*est.se:.6f} (95%)  "
          f"[{est.ci95[0]:.6f}, {est.ci95[1]:.6f}]  "
          f"| n={est.n_total:,}  | método={est.method}  | tempo={est.elapsed_s:.2f}s")

    # 2) Animação (use valores moderados para rodar rápido)
    #    N_total = n_frames * batch_size
    animate_pi_gif(
        n_frames=180,
        batch_size=20_000,
        seed=42,
        use_parallel=True,
        scatter_max=5000,
        outfile="pi_montecarlo.gif",
        visual_mode="grow_then_shuffle",
        grow_ratio=0.5,
    )

    # 3) Benchmark com Markdown
    bench_pi(N=5_000_000, block_size=500_000, repeats=3, md_filename="bench_pi_results.md")