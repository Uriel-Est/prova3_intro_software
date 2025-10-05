# -*- coding: utf-8 -*-
"""
SIDRA helper — PNADC & Censo Agro (PB/NE/BR)
--------------------------------------------
• Gráfico 1: PNADC (tabela 6468, var 4099) — animação trimestral (BR × NE × PB).
• Gráfico 2: Censo Agro (tabela 6960) — barras agrupadas (BR × NE × PB) para o ano mais recente (em geral, 2017).

Boas práticas adotadas:
- Seções claras: imports, config, utilitários, dados (por tabela), gráficos, main.
- Comentários explicativos e funções coesas.
- Checagens essenciais (sem exagero).
- Salva tudo na **mesma pasta** do script (.gif/.png).
"""

# =============================================================================
# 1) IMPORTS
# =============================================================================
# Em seu terminal de escolha rode: pip install - r requirements.txt

from __future__ import annotations

import re
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Backend "Agg" evita problemas com GUI ao salvar GIF/PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =============================================================================
# 2) CONFIG & DEFINIÇÕES GERAIS
# =============================================================================

# Diretório base: mesma pasta do script (fallback para cwd em sessões interativas)
BASE_DIR: Path = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# --- PNADC (tabela 6468, var 4099: taxa de desocupação) ---
URL_PNADC: str = (
    "https://apisidra.ibge.gov.br/values/"
    "t/6468/n1/all/n2/2/n3/25/v/4099/p/all/d/v4099%201"
)

# --- Censo Agro (tabela 6960, João Pessoa/NE/BR) ---
# Versão sem sufixo 'd/...'; usaremos 'v=10094,10095,10096,10097' diretamente.
URL_6960_BASE: str = (
    "https://apisidra.ibge.gov.br/values/"
    "t/6960/n1/all/n2/2/n3/25/v/{vars}/p/all/"
    "c829/allxt/c12528/allxt/c218/4783,46503,46504,111518/c12517/allxt"
)

# Códigos → nomes amigáveis das variáveis
VARS_6960: Dict[str, str] = {
    "10094": "Produção",
    "10095": "Venda",
    "10096": "Valor da produção",
    "10097": "Valor da venda",
}

# Rótulos esperados das séries geográficas
SERIES_TARGETS: Tuple[str, ...] = ("Brasil", "Nordeste", "Paraíba")


# =============================================================================
# 3) FUNÇÕES UTILITÁRIAS (HTTP, parsing, eixos)
# =============================================================================

def make_session(retries: int = 4, backoff: float = 0.7) -> requests.Session:
    """
    Cria sessão Requests com retry leve (o suficiente para oscilações breves).
    """
    r = Retry(
        total=retries, connect=retries, read=retries, status=retries,
        backoff_factor=backoff,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    ad = HTTPAdapter(max_retries=r, pool_connections=10, pool_maxsize=10)
    s = requests.Session()
    s.mount("https://", ad)
    s.mount("http://", ad)
    s.headers.update({"User-Agent": "sidra-helper/1.2"})
    return s


SESSION = make_session()


def fetch_json(url: str, tries: int = 3, base_sleep: float = 0.5, timeout: Tuple[int, int] = (10, 60)) -> list:
    """
    Baixa JSON com algumas tentativas extras (timeout/conexão/HTTP).
    """
    last_exc = None
    for k in range(tries):
        try:
            r = SESSION.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError,
                ValueError) as e:
            last_exc = e
            time.sleep(base_sleep * (2 ** k) + random.uniform(0, 0.25))
    raise last_exc


def detect_period_col(df: pd.DataFrame) -> str:
    """
    Detecta qual coluna D?N contém períodos (tem ano 'YYYY' em boa parte das linhas).
    """
    candidates = [c for c in df.columns if re.fullmatch(r"D\d+N", c or "")]
    best_col, best_rate = "", 0.0
    for c in candidates:
        s = df[c].astype(str)
        rate = s.str.contains(r"\b\d{4}\b", regex=True, na=False).mean()
        if rate > best_rate:
            best_col, best_rate = c, rate
    return best_col


def detect_series_col(df: pd.DataFrame) -> str:
    """
    Detecta a coluna D?N com rótulos geográficos (Brasil/Nordeste/Paraíba).
    """
    candidates = [c for c in df.columns if re.fullmatch(r"D\d+N", c or "")]
    targets = tuple(t.lower() for t in SERIES_TARGETS)
    for c in candidates:
        vals = df[c].astype(str).str.lower().unique()
        if any(any(t in v for v in vals) for t in targets):
            return c
    return ""


def detect_variable_col(df: pd.DataFrame, nomes_alvo: Tuple[str, ...]) -> str:
    """
    Detecta a coluna D?N que contém os nomes das variáveis (ex.: 'Produção', 'Venda', ...).
    """
    candidates = [c for c in df.columns if re.fullmatch(r"D\d+N", c or "")]
    alvo_lower = tuple(x.lower() for x in nomes_alvo)
    for c in candidates:
        vals = df[c].astype(str).str.lower().unique()
        if any(any(a == v or a in v for a in alvo_lower) for v in vals):
            return c
    # Fallback: D?N mais diversa (evita depender de ordem fixa)
    if candidates:
        return max(candidates, key=lambda x: df[x].nunique())
    return ""


def compute_y_limits(tab: pd.DataFrame) -> Tuple[float, float]:
    """
    Limites do eixo Y que ignoram NaN/Inf e evitam faixa degenerada.
    """
    vals = tab.to_numpy().astype(float)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        base = hi if np.isfinite(hi) else 1.0
        return max(0.0, base - 1.0), base + 1.0
    return lo * 0.95, hi * 1.05


def xticks_bienais_compacto(labels: list[str]) -> list[str]:
    """
    Rótulos do eixo X em formato compacto 'AAAA.q' (ex.: 2012.1),
    exibindo 1 trimestre a cada 2 anos (preferindo T1) + o último período.
    Mantém horizontal (sem rotação).
    """
    n = len(labels)
    if n == 0:
        return []

    # Parse ano e trimestre do texto do período (ex.: "2019 Trimestre 4")
    anos, tris = [], []
    for s in labels:
        s = str(s)
        m_ano = re.search(r"(\d{4})", s)
        m_tri = re.search(r"[Tt]r(?:i|ime)?\w*\s*([1-4])", s)
        anos.append(int(m_ano.group(1)) if m_ano else None)
        tris.append(int(m_tri.group(1)) if m_tri else None)

    def compact(a: int | None, q: int | None) -> str:
        # 'AAAA.q' quando houver ambos; senão, tenta 'AAAA'; senão string vazia.
        if a is not None and q is not None:
            return f"{a}.{q}"
        if a is not None:
            return f"{a}"
        return ""

    shown = [""] * n
    anos_validos = [a for a in anos if a is not None]
    if not anos_validos:
        # Sem anos detectados: mostra só o último rótulo (compacto, se possível)
        shown[-1] = compact(anos[-1], tris[-1])
        return shown

    ano_base = min(anos_validos)
    anos_alvo = {a for a in set(anos_validos) if ((a - ano_base) % 2 == 0)}

    # Para cada ano-alvo, preferir T1; senão, o primeiro tri disponível do ano
    for alvo in sorted(anos_alvo):
        idxs = [i for i, a in enumerate(anos) if a == alvo]
        if not idxs:
            continue
        idx_t1 = next((i for i in idxs if tris[i] == 1), None)
        k = idx_t1 if idx_t1 is not None else idxs[0]
        shown[k] = compact(anos[k], tris[k])

    # Sempre garante o último rótulo (compacto)
    shown[-1] = compact(anos[-1], tris[-1]) or labels[-1]
    return shown


# =============================================================================
# 4) DADOS & GRÁFICO 1 — PNADC 6468/4099 (animação trimestral)
# =============================================================================

def load_pnadc_table(url: str = URL_PNADC) -> pd.DataFrame:
    """
    Carrega a PNADC (6468, 4099) em formato wide (index = período textual).
    Colunas = Brasil, Nordeste, Paraíba (quando disponíveis).
    """
    data = fetch_json(url)
    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError("Resposta inesperada da PNADC.")

    df = pd.DataFrame(data[1:])
    if "V" not in df.columns:
        raise RuntimeError("Coluna 'V' ausente na PNADC.")
    df["valor"] = pd.to_numeric(df["V"], errors="coerce")

    col_periodo = detect_period_col(df)
    col_serie   = detect_series_col(df)
    if not col_periodo or not col_serie:
        raise RuntimeError("Não foi possível detectar período/série (PNADC).")

    tidy = df[[col_periodo, col_serie, "valor"]].rename(columns={col_periodo: "periodo", col_serie: "serie"})
    tidy["serie"] = tidy["serie"].astype(str).str.replace(r"\s*\(.*?\)\s*$", "", regex=True)
    tidy = tidy[tidy["serie"].isin(SERIES_TARGETS)].copy()

    # Ordenação temporal (YYYY + Trimestre q)
    per = tidy["periodo"].astype(str)
    yy  = pd.to_numeric(per.str.extract(r"(\d{4})", expand=False), errors="coerce")
    qq  = pd.to_numeric(per.str.extract(r"[Tt]r(?:i|ime)?\w*\s*([1-4])", expand=False), errors="coerce").fillna(1).astype(int)
    tidy["ord"] = yy * 10 + qq

    tab = (tidy.dropna(subset=["ord"])
                .sort_values(["ord", "serie"])
                .pivot_table(index=["ord", "periodo"], columns="serie", values="valor", aggfunc="first"))
    tab = (tab.reset_index()
             .sort_values("ord")
             .set_index("periodo")
             .drop(columns="ord")
             .apply(pd.to_numeric, errors="coerce")
             .dropna(how="all"))

    # Ordena colunas BR/NE/PB quando existirem
    cols = [s for s in SERIES_TARGETS if s in tab.columns]
    tab = tab[cols]
    if len(tab) < 2:
        raise RuntimeError("Poucos períodos retornados pela PNADC.")
    return tab


def animate_pnadc(tab: pd.DataFrame, outfile: Path) -> None:
    """
    Anima BR × NE × PB ao longo dos trimestres, com rótulos de X "bienais" (sem rotação).
    Salva GIF e um PNG final.
    """
    labels = tab.index.to_list()
    n = len(labels)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    lines = {c: ax.plot([], [], label=c, marker="o")[0] for c in tab.columns}

    ax.set_xlim(-0.5, max(0.5, n - 0.5))
    y0, y1 = compute_y_limits(tab)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Período (PNADC)")
    ax.set_ylabel("Taxa de desocupação (%)")
    title = ax.set_title("Taxa de desocupação — PB × NE × Brasil (PNADC)")
    ax.legend(loc="upper right")

    ax.set_xticks(list(range(n)))
    ax.set_xticklabels(xticks_bienais_compacto(labels), rotation=0, ha="center")

    def update(frame: int):
        for c in tab.columns:
            y = tab[c].iloc[: frame + 1].values
            x = list(range(len(y)))
            lines[c].set_data(x, y)
        title.set_text(f"Taxa de desocupação — {labels[frame]}")
        return list(lines.values()) + [title]

    ani = FuncAnimation(fig, update, frames=n, interval=150, blit=False)
    ani.save(str(outfile), writer="pillow", fps=6, dpi=110)
    fig.savefig(str(outfile.with_suffix(".png")), dpi=120, bbox_inches="tight")


# =============================================================================
# 5) DADOS & GRÁFICO 2 — Censo Agro 6960 (barras agrupadas — ano mais recente)
# =============================================================================

def load_6960(url_base: str = URL_6960_BASE, var_map: Dict[str, str] = VARS_6960) -> pd.DataFrame:
    """
    Baixa a 6960 tentando:
      1) uma chamada única com todas as variáveis (v=10094,10095,10096,10097);
      2) fallback: uma chamada por variável.
    Retorna DataFrame com colunas: ['periodo','ano','serie','variavel','valor'] (somado sobre recortes; último ano).
    """
    def _parse_block(json_list: list, var_name: str | None = None) -> pd.DataFrame:
        df = pd.DataFrame(json_list[1:])  # [0] = metadados
        if "V" not in df.columns:
            return pd.DataFrame()
        df["valor"] = pd.to_numeric(df["V"], errors="coerce")

        # detecta período (D?N com ano) e série geográfica (BR/NE/PB)
        col_periodo = detect_period_col(df)
        col_serie   = detect_series_col(df)
        if not col_periodo or not col_serie:
            return pd.DataFrame()

        out = df[[col_periodo, col_serie, "valor"]].rename(
            columns={col_periodo: "periodo", col_serie: "serie"}
        )
        out["serie"] = out["serie"].astype(str).str.replace(r"\s*\(.*?\)\s*$", "", regex=True)
        out = out[out["serie"].isin(SERIES_TARGETS)].copy()

        # ano (normalmente 2017)
        out["ano"] = pd.to_numeric(
            out["periodo"].astype(str).str.extract(r"(\d{4})", expand=False),
            errors="coerce"
        )

        # rótulo da variável: se não foi passado, tenta detectar; se não der, usa texto genérico
        if var_name is None:
            col_var = detect_variable_col(df, tuple(var_map.values()))
            if col_var:
                # alinhar índices: pega apenas as linhas selecionadas em 'out'
                out = out.reset_index(drop=True)
                # traz os rótulos na mesma ordem do 'out'
                out["variavel"] = df.loc[df[col_serie].isin(out["serie"]) & df[col_periodo].isin(out["periodo"]), col_var].astype(str).reset_index(drop=True)
                # fallback caso o alinhamento acima não bata pela ordem
                if out["variavel"].isna().any():
                    out["variavel"] = out["variavel"].fillna("Variável")
            else:
                out["variavel"] = "Variável"
        else:
            out["variavel"] = var_name

        return out

    frames: list[pd.DataFrame] = []

    # --- 1) Tenta tudo numa chamada só ---
    all_codes = ",".join(var_map.keys())
    url_all = url_base.format(vars=all_codes)
    try:
        data_all = fetch_json(url_all)
        if isinstance(data_all, list) and len(data_all) >= 2:
            block = _parse_block(data_all, var_name=None)
            if block.empty or "variavel" not in block.columns or block["variavel"].isna().all():
                # se veio sem rótulo claro de variável, cai para o fallback por variável
                raise ValueError("Batch sem rótulo adequado; usando chamadas por variável.")
            frames.append(block)
        else:
            raise ValueError("Resposta inesperada no batch; usando fallback por variável.")
    except Exception:
        # --- 2) Fallback: uma chamada por variável ---
        frames.clear()
        for code, name in var_map.items():
            url = url_base.format(vars=code)
            try:
                data = fetch_json(url)
                if isinstance(data, list) and len(data) >= 2:
                    frames.append(_parse_block(data, var_name=name))
            except Exception:
                # ignora variável que falhou; segue com as demais
                continue

    if not frames:
        raise RuntimeError("Não foi possível montar os dados da 6960 (verifique conexão/endpoint).")

    tidy = pd.concat(frames, ignore_index=True)
    # agrega por período/série/variável (somando sobre demais classificações)
    grp = (tidy.groupby(["periodo", "ano", "serie", "variavel"], as_index=False)["valor"]
                .sum())

    # fica com o último ano disponível (em geral, 2017)
    if grp["ano"].notna().any():
        last_year = int(grp["ano"].dropna().max())
        grp = grp[grp["ano"] == last_year].copy()

    # ordenações amigáveis
    grp["variavel"] = pd.Categorical(grp["variavel"], categories=list(var_map.values()), ordered=True)
    grp["serie"]    = pd.Categorical(grp["serie"], categories=list(SERIES_TARGETS), ordered=True)
    grp = grp.sort_values(["variavel", "serie"]).reset_index(drop=True)
    return grp  # columns: periodo, ano, serie, variavel, valor


def plot_6960_grouped_bars(df: pd.DataFrame, outfile: Path) -> None:
    """
    Barras agrupadas por variável (x) e série (cores: BR/NE/PB).
    Salva PNG na mesma pasta do script.
    """
    var_order = list(VARS_6960.values())
    pivot = df.pivot_table(index="variavel", columns="serie", values="valor", aggfunc="first").reindex(var_order)
    variables = pivot.index.to_list()
    series = [s for s in SERIES_TARGETS if s in pivot.columns]
    X = np.arange(len(variables), dtype=float)

    # Config de barras
    width = 0.22
    offsets = np.linspace(-width, width, num=len(series), endpoint=True)

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for i, s in enumerate(series):
        ax.bar(X + offsets[i], pivot[s].values, width=width, label=s)

    ax.set_xticks(X)
    ax.set_xticklabels(variables, rotation=0, ha="center")
    ax.set_ylabel("Valor (unidade conforme variável)")

    # Título com ano
    if df["ano"].notna().any():
        year = int(df["ano"].dropna().max())
        ax.set_title(f"Censo Agro (tabela 6960) — PB × NE × BR — {year}")
    else:
        ax.set_title("Censo Agro (tabela 6960) — PB × NE × BR")

    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(str(outfile), dpi=120, bbox_inches="tight")


# =============================================================================
# 6) MAIN — executa tudo e salva na pasta do script
# =============================================================================

def main() -> None:
    # --- PNADC: animação trimestral ---
    tab_pnadc = load_pnadc_table(URL_PNADC)
    animate_pnadc(tab_pnadc, outfile=BASE_DIR / "pnadc_pb_ne_br_TRIMESTRAL.gif")
    print("✔ PNADC: salvos GIF + PNG na pasta do script.")

    # --- 6960: barras agrupadas (ano mais recente — tipicamente 2017) ---
    df6960 = load_6960(URL_6960_BASE, VARS_6960)
    plot_6960_grouped_bars(df6960, outfile=BASE_DIR / "agro6960_pb_ne_br.png")
    print("✔ 6960: salvo PNG na pasta do script.")


if __name__ == "__main__":
    main()
