#!/usr/bin/env python3

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


URL_ALLIANZ = (
    "https://tools.morningstar.it/api/rest.svc/timeseries_price/jbyiq3rhyf"
    "?currencyId=EUR&idtype=Morningstar&frequency=daily&startDate=2000-01-01"
    "&outputType=COMPACTJSON&id=0P0000CWZR%5D22%5D0%5DDETEXG%24XLON"
)

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_timeseries(url: str) -> pd.DataFrame:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://tools.morningstar.it/",
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    data = response.json()

    if not isinstance(data, list) or not data:
        raise ValueError("Formato dati inatteso: attesa lista di [timestamp, valore]")

    frame = pd.DataFrame(data, columns=["timestamp_ms", "nav"])
    frame["date"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True).dt.tz_localize(None)
    frame = frame.dropna(subset=["nav"]).sort_values("date").reset_index(drop=True)
    frame = frame[["date", "nav"]]
    return frame


def compute_drawdown(nav_series: pd.Series) -> Tuple[pd.Series, float]:
    running_max = nav_series.cummax()
    drawdown = nav_series / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else float("nan")
    return drawdown, max_drawdown


def compute_metrics(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        raise ValueError("Serie vuota, impossibile calcolare le metriche")

    nav = df["nav"].astype(float)
    daily_returns = nav.pct_change().dropna()

    if len(daily_returns) == 0:
        raise ValueError("Serie di rendimenti vuota")

    ann_factor = 252.0

    start_date = df["date"].iloc[0]
    end_date = df["date"].iloc[-1]
    num_years = max((end_date - start_date).days / 365.25, 1e-9)

    cumulative_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / num_years) - 1.0)

    mean_daily = float(daily_returns.mean())
    std_daily = float(daily_returns.std(ddof=1))
    ann_return = mean_daily * ann_factor
    ann_volatility = std_daily * math.sqrt(ann_factor)

    downside = daily_returns[daily_returns < 0.0]
    downside_std_daily = float(downside.std(ddof=1)) if len(downside) > 1 else float("nan")
    downside_dev_ann = (
        downside_std_daily * math.sqrt(ann_factor) if not math.isnan(downside_std_daily) else float("nan")
    )

    sharpe = ann_return / ann_volatility if ann_volatility > 0 else float("nan")
    sortino = ann_return / downside_dev_ann if downside_dev_ann > 0 else float("nan")

    drawdown_series, max_drawdown = compute_drawdown(nav)
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else float("nan")

    var_95_daily = float(np.nanpercentile(daily_returns, 5))

    metrics = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "observations": int(len(df)),
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "ann_return": ann_return,
        "ann_volatility": ann_volatility,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar,
        "var_95_daily": var_95_daily,
    }
    return metrics


def calendar_year_returns(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.set_index("date").copy()
    annual = temp.resample("Y").last()
    annual["prev"] = annual["nav"].shift(1)
    annual["calendar_return"] = annual["nav"] / annual["prev"] - 1.0
    annual.index = annual.index.year
    return annual[["calendar_return"]].dropna()


def rolling_series(df: pd.DataFrame) -> Dict[str, pd.Series]:
    temp = df.set_index("date").copy()
    nav = temp["nav"].astype(float)
    daily_returns = nav.pct_change()

    roll_12m_return = nav.pct_change(252)
    roll_36m_vol = daily_returns.rolling(252 * 3).std() * math.sqrt(252)
    return {
        "roll_12m_return": roll_12m_return.dropna(),
        "roll_36m_vol": roll_36m_vol.dropna(),
    }


def format_pct(x: float) -> str:
    if x is None or math.isnan(x):
        return "n/d"
    return f"{x*100:.2f}%"


def plot_price(df: pd.DataFrame, out_path: str) -> None:
    plt.figure(figsize=(11, 5))
    sns.lineplot(x="date", y="nav", data=df, linewidth=1.2, color="#1f77b4")
    plt.title("Allianz Insieme – Linea Azionaria: Valore quota")
    plt.xlabel("Data")
    plt.ylabel("NAV (EUR)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_drawdown(df: pd.DataFrame, out_path: str) -> None:
    nav = df["nav"].astype(float)
    drawdown, _ = compute_drawdown(nav)
    dd_df = pd.DataFrame({"date": df["date"], "drawdown": drawdown.values})

    plt.figure(figsize=(11, 4))
    sns.lineplot(x="date", y="drawdown", data=dd_df, color="#d62728", linewidth=1.0)
    plt.fill_between(dd_df["date"], dd_df["drawdown"], 0, color="#d62728", alpha=0.2)
    plt.title("Drawdown")
    plt.xlabel("Data")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def plot_rolling(roll: Dict[str, pd.Series], out_returns: str, out_vol: str) -> None:
    plt.figure(figsize=(11, 4))
    sns.lineplot(x=roll["roll_12m_return"].index, y=roll["roll_12m_return"].values, color="#2ca02c")
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.title("Rendimento rolling 12 mesi")
    plt.xlabel("Data")
    plt.ylabel("Rendimento 12m")
    plt.tight_layout()
    plt.savefig(out_returns, dpi=130)
    plt.close()

    plt.figure(figsize=(11, 4))
    sns.lineplot(x=roll["roll_36m_vol"].index, y=roll["roll_36m_vol"].values, color="#9467bd")
    plt.title("Volatilità rolling 36 mesi (ann.)")
    plt.xlabel("Data")
    plt.ylabel("Volatilità ann.")
    plt.tight_layout()
    plt.savefig(out_vol, dpi=130)
    plt.close()


def save_report(metrics: Dict[str, float], cal_ret: pd.DataFrame, paths: Dict[str, str]) -> None:
    report_path = paths["report"]

    lines = []
    lines.append("# Allianz Insieme – Linea Azionaria: Analisi\n")
    lines.append(f"Periodo: {metrics['start_date']} → {metrics['end_date']}  ")
    lines.append(f"Osservazioni: {metrics['observations']}\n")

    lines.append("## Metriche principali\n")
    lines.append(f"- Rendimento cumulato: {format_pct(metrics['cumulative_return'])}")
    lines.append(f"- CAGR: {format_pct(metrics['cagr'])}")
    lines.append(f"- Rendimento ann.: {format_pct(metrics['ann_return'])}")
    lines.append(f"- Volatilità ann.: {format_pct(metrics['ann_volatility'])}")
    lines.append(f"- Sharpe (rf=0): {metrics['sharpe_ratio']:.2f}" if not math.isnan(metrics['sharpe_ratio']) else "- Sharpe (rf=0): n/d")
    lines.append(f"- Sortino (rf=0): {metrics['sortino_ratio']:.2f}" if not math.isnan(metrics['sortino_ratio']) else "- Sortino (rf=0): n/d")
    lines.append(f"- Max drawdown: {format_pct(metrics['max_drawdown'])}")
    lines.append(f"- Calmar: {metrics['calmar_ratio']:.2f}" if not math.isnan(metrics['calmar_ratio']) else "- Calmar: n/d")
    lines.append(f"- VaR(95%) giornaliero: {format_pct(metrics['var_95_daily'])}\n")

    lines.append("## Grafici\n")
    lines.append(f"![NAV]({os.path.basename(paths['price'])})")
    lines.append("")
    lines.append(f"![Drawdown]({os.path.basename(paths['drawdown'])})")
    lines.append("")
    lines.append(f"![Rolling 12m return]({os.path.basename(paths['roll_ret'])})")
    lines.append("")
    lines.append(f"![Rolling 36m vol]({os.path.basename(paths['roll_vol'])})\n")

    if not cal_ret.empty:
        lines.append("## Rendimento per anno\n")
        lines.append("Anno | Rendimento")
        lines.append("---|---")
        for year, row in cal_ret.iterrows():
            lines.append(f"{int(year)} | {format_pct(float(row['calendar_return']))}")
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def main() -> None:
    print("Scarico i dati da Morningstar…")
    df = fetch_timeseries(URL_ALLIANZ)

    data_csv_path = os.path.join(OUTPUT_DIR, "allianz_insieme_azionaria_nav.csv")
    df.to_csv(data_csv_path, index=False)

    print(f"Serie storica salvata in: {data_csv_path}")

    print("Calcolo metriche…")
    metrics = compute_metrics(df)

    print("Costruisco serie rolling e rendimenti per anno…")
    cal_ret = calendar_year_returns(df)
    roll = rolling_series(df)

    print("Genero i grafici…")
    paths = {
        "price": os.path.join(OUTPUT_DIR, "price.png"),
        "drawdown": os.path.join(OUTPUT_DIR, "drawdown.png"),
        "roll_ret": os.path.join(OUTPUT_DIR, "rolling_12m_return.png"),
        "roll_vol": os.path.join(OUTPUT_DIR, "rolling_36m_vol.png"),
        "report": os.path.join(OUTPUT_DIR, "report.md"),
    }

    plot_price(df, paths["price"])
    plot_drawdown(df, paths["drawdown"])
    plot_rolling(roll, paths["roll_ret"], paths["roll_vol"])

    print("Scrivo il report…")
    save_report(metrics, cal_ret, paths)

    print("Fatto. Vedi la cartella 'output' per CSV, immagini e report.md")


if __name__ == "__main__":
    main()