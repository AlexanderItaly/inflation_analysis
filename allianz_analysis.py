#!/usr/bin/env python3

import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

URL_ALLIANZ = (
    "https://tools.morningstar.it/api/rest.svc/timeseries_price/jbyiq3rhyf"
    "?currencyId=EUR&idtype=Morningstar&frequency=daily&startDate=2000-01-01"
    "&outputType=COMPACTJSON&id=0P0000CWZR%5D22%5D0%5DDETEXG%24XLON"
)

OUTPUT_DIR = os.path.join(os.getcwd(), "output")


def http_get_json(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://tools.morningstar.it/",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def fetch_timeseries(url: str) -> List[Tuple[datetime, float]]:
    data = http_get_json(url)
    if not isinstance(data, list) or not data:
        raise ValueError("Formato dati inatteso: attesa lista di [timestamp, valore]")
    series: List[Tuple[datetime, float]] = []
    for ts_ms, nav in data:
        if nav is None:
            continue
        dt = datetime.utcfromtimestamp(int(ts_ms) / 1000.0)
        series.append((dt, float(nav)))
    series.sort(key=lambda x: x[0])
    return series


def compute_drawdown(nav: List[float]) -> Tuple[List[float], float]:
    peak = -float("inf")
    dd: List[float] = []
    for v in nav:
        if v > peak:
            peak = v
        dd.append(v / peak - 1.0 if peak > 0 else 0.0)
    max_dd = min(dd) if dd else float("nan")
    return dd, max_dd


def time_weighted_stats(dates: List[datetime], nav: List[float]) -> Tuple[float, float]:
    if len(nav) < 2:
        return float("nan"), float("nan")
    log_rets: List[float] = []
    dt_years: List[float] = []
    for i in range(1, len(nav)):
        if nav[i - 1] <= 0 or nav[i] <= 0:
            continue
        lr = math.log(nav[i] / nav[i - 1])
        days = (dates[i] - dates[i - 1]).days + (dates[i] - dates[i - 1]).seconds / 86400.0
        if days <= 0:
            continue
        log_rets.append(lr)
        dt_years.append(days / 365.25)
    if not dt_years:
        return float("nan"), float("nan")
    total_years = sum(dt_years)
    mu = sum(log_rets) / total_years
    # variance of log-return rate per year using weighted LS residuals
    resid_sq_sum = 0.0
    for lr, dt in zip(log_rets, dt_years):
        resid = lr - mu * dt
        resid_sq_sum += resid * resid
    var_ann = resid_sq_sum / total_years
    sigma = math.sqrt(max(0.0, var_ann))
    return mu, sigma


def daily_var_from_irregular(dates: List[datetime], nav: List[float], q: float = 0.05) -> float:
    per_day_logs: List[float] = []
    for i in range(1, len(nav)):
        if nav[i - 1] <= 0 or nav[i] <= 0:
            continue
        lr = math.log(nav[i] / nav[i - 1])
        days = (dates[i] - dates[i - 1]).days + (dates[i] - dates[i - 1]).seconds / 86400.0
        if days <= 0:
            continue
        per_day_logs.append(lr / days)
    if not per_day_logs:
        return float("nan")
    per_day_logs.sort()
    k = max(0, min(len(per_day_logs) - 1, int(q * len(per_day_logs)) - 1))
    return math.expm1(per_day_logs[k])


def calendar_year_returns(dates: List[datetime], nav: List[float]) -> List[Tuple[int, float]]:
    # pick last observation per year
    last_by_year: Dict[int, Tuple[datetime, float]] = {}
    for d, v in zip(dates, nav):
        y = d.year
        tup = last_by_year.get(y)
        if tup is None or d > tup[0]:
            last_by_year[y] = (d, v)
    years = sorted(last_by_year.keys())
    out: List[Tuple[int, float]] = []
    for i in range(1, len(years)):
        y = years[i]
        y_prev = years[i - 1]
        nav_now = last_by_year[y][1]
        nav_prev = last_by_year[y_prev][1]
        if nav_prev > 0:
            out.append((y, nav_now / nav_prev - 1.0))
    return out


def trailing_return(dates: List[datetime], nav: List[float], years: int) -> Tuple[float, float]:
    if not dates:
        return float("nan"), float("nan")
    end_date = dates[-1]
    lookup = end_date - timedelta(days=int(365.25 * years))
    # find last date <= lookup
    idx = 0
    for i, d in enumerate(dates):
        if d <= lookup:
            idx = i
        else:
            break
    if dates[idx] > lookup or idx >= len(nav) - 1:
        return float("nan"), float("nan")
    nav_prev = nav[idx]
    nav_now = nav[-1]
    if nav_prev <= 0:
        return float("nan"), float("nan")
    total = nav_now / nav_prev - 1.0
    cagr = (nav_now / nav_prev) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return total, cagr


def min_max(vals: List[float]) -> Tuple[float, float]:
    mn = float("inf")
    mx = -float("inf")
    for v in vals:
        if v < mn:
            mn = v
        if v > mx:
            mx = v
    if mn == float("inf"):
        mn = 0.0
    if mx == -float("inf"):
        mx = 1.0
    return mn, mx


def scale_points(xs: List[float], ys: List[float], width: int, height: int, pad: int = 20) -> List[Tuple[int, int]]:
    x_min, x_max = min_max(xs)
    y_min, y_max = min_max(ys)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0
    def sx(x: float) -> int:
        return int(pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad))
    def sy(y: float) -> int:
        # invert y (SVG origin is top-left)
        return int(height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad))
    return [(sx(x), sy(y)) for x, y in zip(xs, ys)]


def svg_polyline(points: List[Tuple[int, int]], stroke: str) -> str:
    if not points:
        return ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f"<polyline fill='none' stroke='{stroke}' stroke-width='1.5' points='{pts}'/>"


def svg_axes(width: int, height: int) -> str:
    return f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>"


def write_svg_price(dates: List[datetime], nav: List[float], out_path: str) -> None:
    width, height = 900, 420
    xs = [d.timestamp() for d in dates]
    ys = nav
    pts = scale_points(xs, ys, width, height)
    svg = ["<svg xmlns='http://www.w3.org/2000/svg' width='900' height='420'>"]
    svg.append(svg_axes(width, height))
    svg.append(svg_polyline(pts, "#1f77b4"))
    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def write_svg_drawdown(dates: List[datetime], nav: List[float], out_path: str) -> None:
    width, height = 900, 320
    dd, _ = compute_drawdown(nav)
    xs = [d.timestamp() for d in dates]
    ys = dd
    pts = scale_points(xs, ys, width, height)
    svg = ["<svg xmlns='http://www.w3.org/2000/svg' width='900' height='320'>"]
    svg.append(svg_axes(width, height))
    svg.append(svg_polyline(pts, "#d62728"))
    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def write_svg_series(series: List[Tuple[datetime, float]], color: str, width: int, height: int, out_path: str) -> None:
    xs = [d.timestamp() for d, _ in series]
    ys = [v for _, v in series]
    pts = scale_points(xs, ys, width, height)
    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg.append(svg_axes(width, height))
    svg.append(svg_polyline(pts, color))
    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def rolling_12m_return(dates: List[datetime], nav: List[float]) -> List[Tuple[datetime, float]]:
    out: List[Tuple[datetime, float]] = []
    for i in range(len(nav)):
        d = dates[i]
        lookup = d - timedelta(days=365)
        # find last j with dates[j] <= lookup
        j = None
        for k in range(i, -1, -1):
            if dates[k] <= lookup:
                j = k
                break
        if j is None:
            continue
        if nav[j] > 0:
            out.append((d, nav[i] / nav[j] - 1.0))
    return out


def rolling_36m_vol(dates: List[datetime], nav: List[float]) -> List[Tuple[datetime, float]]:
    out: List[Tuple[datetime, float]] = []
    for i in range(1, len(nav)):
        end_d = dates[i]
        start_cut = end_d - timedelta(days=int(365.25 * 3))
        log_rets: List[float] = []
        dt_years: List[float] = []
        for k in range(i, 0, -1):
            if dates[k] <= start_cut:
                break
            if nav[k - 1] <= 0 or nav[k] <= 0:
                continue
            lr = math.log(nav[k] / nav[k - 1])
            days = (dates[k] - dates[k - 1]).days + (dates[k] - dates[k - 1]).seconds / 86400.0
            if days <= 0:
                continue
            log_rets.append(lr)
            dt_years.append(days / 365.25)
        if not dt_years:
            continue
        total_years = sum(dt_years)
        mu = sum(log_rets) / total_years
        resid_sq_sum = 0.0
        for lr, dt in zip(log_rets, dt_years):
            resid_sq_sum += (lr - mu * dt) ** 2
        sigma = math.sqrt(max(0.0, resid_sq_sum / total_years))
        out.append((end_d, sigma))
    return out


def format_pct(x: float) -> str:
    if x != x or x is None:  # NaN check
        return "n/d"
    return f"{x*100:.2f}%"


def save_report(metrics: Dict[str, float], cal_ret: List[Tuple[int, float]], trailing: Dict[str, Tuple[float, float]], paths: Dict[str, str]) -> None:
    os.makedirs(os.path.dirname(paths["report"]), exist_ok=True)
    lines: List[str] = []
    lines.append("# Allianz Insieme – Linea Azionaria: Analisi\n")
    lines.append(f"Periodo: {metrics['start_date']} → {metrics['end_date']}  ")
    lines.append(f"Osservazioni: {int(metrics['observations'])}\n")

    lines.append("## Metriche principali\n")
    lines.append(f"- Rendimento cumulato: {format_pct(metrics['cumulative_return'])}")
    lines.append(f"- CAGR: {format_pct(metrics['cagr'])}")
    lines.append(f"- Rendimento ann. (approx aritmetico): {format_pct(metrics['ann_return'])}")
    lines.append(f"- Rendimento ann. (comp. continuo): {format_pct(metrics['ann_return_log'])}")
    lines.append(f"- Volatilità ann.: {format_pct(metrics['ann_volatility'])}")
    lines.append(f"- Sharpe (rf=0): {metrics['sharpe_ratio']:.2f}" if not math.isnan(metrics['sharpe_ratio']) else "- Sharpe (rf=0): n/d")
    lines.append(f"- Sortino (rf=0): {metrics['sortino_ratio']:.2f}" if not math.isnan(metrics['sortino_ratio']) else "- Sortino (rf=0): n/d")
    lines.append(f"- Max drawdown: {format_pct(metrics['max_drawdown'])}")
    lines.append(f"- Calmar: {metrics['calmar_ratio']:.2f}" if not math.isnan(metrics['calmar_ratio']) else "- Calmar: n/d")
    lines.append(f"- VaR(95%) giornaliero: {format_pct(metrics['var_95_daily'])}\n")
    lines.append("Nota: la frequenza dei dati è mista (giornaliera → settimanale). Le metriche sono calcolate con stime time-weighted robuste a campionamenti irregolari.\n")

    lines.append("## Rendimenti trailing\n")
    lines.append("Periodo | Totale | CAGR")
    lines.append("---|---|---")
    for y in [1, 3, 5, 10]:
        tot, cagr = trailing.get(f"{y}y", (float("nan"), float("nan")))
        lines.append(f"{y} anni | {format_pct(tot)} | {format_pct(cagr)}")
    lines.append("")

    lines.append("## Rendimento per anno\n")
    lines.append("Anno | Rendimento")
    lines.append("---|---")
    for y, r in cal_ret:
        lines.append(f"{y} | {format_pct(r)}")
    lines.append("")

    lines.append("## Grafici\n")
    lines.append(f"![NAV]({os.path.basename(paths['price'])})")
    lines.append("")
    lines.append(f"![Drawdown]({os.path.basename(paths['drawdown'])})")
    lines.append("")
    lines.append(f"![Rolling 12m return]({os.path.basename(paths['roll_ret'])})")
    lines.append("")
    lines.append(f"![Rolling 36m vol]({os.path.basename(paths['roll_vol'])})\n")

    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Scarico i dati da Morningstar…")
    series = fetch_timeseries(URL_ALLIANZ)
    dates = [d for d, _ in series]
    nav = [v for _, v in series]

    csv_path = os.path.join(OUTPUT_DIR, "allianz_insieme_azionaria_nav.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("date,nav\n")
        for d, v in series:
            f.write(f"{d.strftime('%Y-%m-%d')},{v}\n")

    print(f"Serie storica salvata in: {csv_path}")

    print("Calcolo metriche…")
    mu, sigma = time_weighted_stats(dates, nav)

    start_date = dates[0]
    end_date = dates[-1]
    n_years = max((end_date - start_date).days / 365.25, 1e-9)
    cumulative_return = nav[-1] / nav[0] - 1.0
    cagr = (nav[-1] / nav[0]) ** (1.0 / n_years) - 1.0

    # Arithmetic approx annual return from log moments
    ann_return_log = math.expm1(mu) if mu == mu else float("nan")
    ann_return = mu + 0.5 * (sigma * sigma) if (mu == mu and sigma == sigma) else float("nan")

    # Sortino downside (time-weighted) using negative interval returns
    neg_log: List[float] = []
    neg_dt_years: List[float] = []
    for i in range(1, len(nav)):
        r = nav[i] / nav[i - 1] - 1.0
        if r < 0 and nav[i - 1] > 0 and nav[i] > 0:
            lr = math.log(nav[i] / nav[i - 1])
            days = (dates[i] - dates[i - 1]).days + (dates[i] - dates[i - 1]).seconds / 86400.0
            if days > 0:
                neg_log.append(lr)
                neg_dt_years.append(days / 365.25)
    if neg_dt_years:
        mu_d = sum(neg_log) / sum(neg_dt_years)
        resid_sq = 0.0
        for lr, dt in zip(neg_log, neg_dt_years):
            resid_sq += (lr - mu_d * dt) ** 2
        downside_dev_ann = math.sqrt(max(0.0, resid_sq / sum(neg_dt_years)))
    else:
        downside_dev_ann = float("nan")

    dd, max_dd = compute_drawdown(nav)
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    var_95_daily = daily_var_from_irregular(dates, nav, q=0.05)

    sharpe = (ann_return / sigma) if (sigma == sigma and sigma > 0 and ann_return == ann_return) else float("nan")
    sortino = (ann_return / downside_dev_ann) if (downside_dev_ann == downside_dev_ann and downside_dev_ann > 0 and ann_return == ann_return) else float("nan")

    metrics = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "observations": float(len(nav)),
        "cumulative_return": cumulative_return,
        "cagr": cagr,
        "ann_return": ann_return,
        "ann_return_log": ann_return_log,
        "ann_volatility": sigma,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "var_95_daily": var_95_daily,
    }

    print("Calcolo rolling e trailing…")
    roll12 = rolling_12m_return(dates, nav)
    roll36v = rolling_36m_vol(dates, nav)

    trailing = {
        "1y": trailing_return(dates, nav, 1),
        "3y": trailing_return(dates, nav, 3),
        "5y": trailing_return(dates, nav, 5),
        "10y": trailing_return(dates, nav, 10),
    }

    print("Genero grafici SVG…")
    paths = {
        "price": os.path.join(OUTPUT_DIR, "price.svg"),
        "drawdown": os.path.join(OUTPUT_DIR, "drawdown.svg"),
        "roll_ret": os.path.join(OUTPUT_DIR, "rolling_12m_return.svg"),
        "roll_vol": os.path.join(OUTPUT_DIR, "rolling_36m_vol.svg"),
        "report": os.path.join(OUTPUT_DIR, "report.md"),
    }

    write_svg_price(dates, nav, paths["price"])
    write_svg_drawdown(dates, nav, paths["drawdown"])
    write_svg_series(roll12, "#2ca02c", 900, 320, paths["roll_ret"])
    write_svg_series(roll36v, "#9467bd", 900, 320, paths["roll_vol"])

    print("Scrivo report…")
    save_report(metrics, calendar_year_returns(dates, nav), trailing, paths)

    print("Fatto. Vedi la cartella 'output' per CSV, SVG e report.md")


if __name__ == "__main__":
    main()