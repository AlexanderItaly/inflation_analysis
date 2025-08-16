#!/usr/bin/env python3

import json
import math
import os
import sys
import urllib.request
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from zipfile import ZipFile, ZIP_DEFLATED

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


def trailing_return_asof(dates: List[datetime], nav: List[float], years: int, asof_date: datetime) -> Tuple[float, float]:
    if not dates:
        return float("nan"), float("nan")
    # find last observation <= asof_date
    end_idx = None
    for i in range(len(dates) - 1, -1, -1):
        if dates[i] <= asof_date:
            end_idx = i
            break
    if end_idx is None:
        return float("nan"), float("nan")
    nav_now = nav[end_idx]
    lookup = asof_date - timedelta(days=int(365.25 * years))
    prev_idx = None
    for j in range(end_idx, -1, -1):
        if dates[j] <= lookup:
            prev_idx = j
            break
    if prev_idx is None or nav[prev_idx] <= 0:
        return float("nan"), float("nan")
    total = nav_now / nav[prev_idx] - 1.0
    cagr = (nav_now / nav[prev_idx]) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    return total, cagr


def compute_trailing_returns_asof(dates: List[datetime], nav: List[float], asof_date: datetime) -> Dict[str, Tuple[float, float]]:
    tr: Dict[str, Tuple[float, float]] = {}
    for y in [1, 3, 5, 10, 20]:
        tr[f"{y}y"] = trailing_return_asof(dates, nav, y, asof_date)
    return tr


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


def scale_points(xs: List[float], ys: List[float], width: int, height: int, pad: int = 40) -> Tuple[List[Tuple[int, int]], Tuple[float,float,float,float]]:
    x_min = min(xs) if xs else 0.0
    x_max = max(xs) if xs else 1.0
    y_min = min(ys) if ys else 0.0
    y_max = max(ys) if ys else 1.0
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0
    def sx(x: float) -> int:
        return int(pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad))
    def sy(y: float) -> int:
        return int(height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad))
    pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]
    return pts, (x_min, x_max, y_min, y_max)


def svg_polyline(points: List[Tuple[int, int]], stroke: str) -> str:
    if not points:
        return ""
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f"<polyline fill='none' stroke='{stroke}' stroke-width='1.5' points='{pts}'/>"


def svg_axes(width: int, height: int) -> str:
    return f"<rect x='0' y='0' width='{width}' height='{height}' fill='white' stroke='none'/>"


def _svg_grid_and_ticks(x_min: float, x_max: float, y_min: float, y_max: float, width: int, height: int, pad: int, x_is_time: bool, y_is_percent: bool, x_ticks: int = 12, y_ticks: int = 8) -> Tuple[str, callable, callable]:
    def sx(x: float) -> int:
        return int(pad + (x - x_min) / (x_max - x_min) * (width - 2 * pad))
    def sy(y: float) -> int:
        return int(height - pad - (y - y_min) / (y_max - y_min) * (height - 2 * pad))

    elements: List[str] = []
    # Axes lines
    elements.append(f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='#444' stroke-width='1' />")
    elements.append(f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='#444' stroke-width='1' />")

    # Grid + ticks + labels
    # Y ticks
    for i in range(y_ticks):
        y_val = y_min + i * (y_max - y_min) / (y_ticks - 1)
        y_pix = sy(y_val)
        elements.append(f"<line x1='{pad}' y1='{y_pix}' x2='{width-pad}' y2='{y_pix}' stroke='#eee' stroke-width='1' />")
        label = f"{y_val*100:.1f}%" if y_is_percent else f"{y_val:.2f}"
        elements.append(f"<text x='{pad-6}' y='{y_pix+4}' font-size='11' fill='#555' text-anchor='end' font-family='Arial, Helvetica, sans-serif'>{label}</text>")

    # X ticks
    for i in range(x_ticks):
        x_val = x_min + i * (x_max - x_min) / (x_ticks - 1)
        x_pix = sx(x_val)
        elements.append(f"<line x1='{x_pix}' y1='{height-pad}' x2='{x_pix}' y2='{pad}' stroke='#eee' stroke-width='1' />")
        if x_is_time:
            dt = datetime.utcfromtimestamp(x_val)
            label = dt.strftime('%Y')
        else:
            label = f"{x_val:.0f}"
        elements.append(f"<text x='{x_pix}' y='{height-pad+16}' font-size='11' fill='#555' text-anchor='middle' font-family='Arial, Helvetica, sans-serif'>{label}</text>")

    return "\n".join(elements), sx, sy


def write_svg_price(dates: List[datetime], nav: List[float], out_path: str) -> None:
    width, height, pad = 920, 460, 50
    xs = [d.timestamp() for d in dates]
    ys = nav
    _, (x_min, x_max, y_min, y_max) = scale_points(xs, ys, width, height, pad)

    grid, sx, sy = _svg_grid_and_ticks(x_min, x_max, y_min, y_max, width, height, pad, x_is_time=True, y_is_percent=False)
    pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]

    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg.append(svg_axes(width, height))
    svg.append(grid)
    svg.append(svg_polyline(pts, "#1f77b4"))
    # Labels
    svg.append(f"<text x='{width/2:.0f}' y='{20}' font-size='14' font-weight='bold' text-anchor='middle' font-family='Arial, Helvetica, sans-serif'>Allianz Insieme – Valore quota</text>")
    svg.append(f"<text x='{width/2:.0f}' y='{height-10}' font-size='12' text-anchor='middle' font-family='Arial, Helvetica, sans-serif'>Data</text>")
    svg.append(f"<text x='{16}' y='{height/2:.0f}' font-size='12' text-anchor='middle' transform='rotate(-90 16,{height/2:.0f})' font-family='Arial, Helvetica, sans-serif'>NAV (EUR)</text>")
    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def write_svg_drawdown(dates: List[datetime], nav: List[float], out_path: str) -> None:
    width, height, pad = 920, 360, 50
    dd, _ = compute_drawdown(nav)
    xs = [d.timestamp() for d in dates]
    ys = dd
    # ensure y_max includes 0 to see baseline
    y_min = min(ys) if ys else -0.5
    y_max = max(0.0, max(ys) if ys else 0.0)
    x_min = min(xs) if xs else 0.0
    x_max = max(xs) if xs else 1.0

    grid, sx, sy = _svg_grid_and_ticks(x_min, x_max, y_min, y_max, width, height, pad, x_is_time=True, y_is_percent=True)
    pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]

    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg.append(svg_axes(width, height))
    svg.append(grid)
    svg.append(svg_polyline(pts, "#d62728"))
    svg.append(f"<text x='{width/2:.0f}' y='{20}' font-size='14' font-weight='bold' text-anchor='middle' font-family='Arial, Helvetica, sans-serif'>Drawdown</text>")
    svg.append(f"<text x='{width/2:.0f}' y='{height-10}' font-size='12' text-anchor='middle' font-family='Arial, Helvetica, sans-serif'>Data</text>")
    svg.append(f"<text x='{16}' y='{height/2:.0f}' font-size='12' text-anchor='middle' transform='rotate(-90 16,{height/2:.0f})' font-family='Arial, Helvetica, sans-serif'>Drawdown</text>")
    svg.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def write_svg_series(series: List[Tuple[datetime, float]], color: str, width: int, height: int, out_path: str, title: str, ylabel: str, y_is_percent: bool, x_ticks: int = 12, y_ticks: int = 8) -> None:
    pad = 50
    xs = [d.timestamp() for d, _ in series]
    ys = [v for _, v in series]
    if not xs or not ys:
        xs, ys = [0.0, 1.0], [0.0, 1.0]
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    grid, sx, sy = _svg_grid_and_ticks(x_min, x_max, y_min, y_max, width, height, pad, x_is_time=True, y_is_percent=y_is_percent, x_ticks=x_ticks, y_ticks=y_ticks)
    pts = [(sx(x), sy(y)) for x, y in zip(xs, ys)]

    svg = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg.append(svg_axes(width, height))
    svg.append(grid)
    svg.append(svg_polyline(pts, color))
    svg.append(f"<text x='{width/2:.0f}' y='{20}' font-size='14' font-weight='bold' text-anchor='middle' font-family='Arial, Helvetica, sans-serif'>{title}</text>")
    svg.append(f"<text x='{width/2:.0f}' y='{height-10}' font-size='12' text-anchor='middle' font-family='Arial, Helvetica, sans-serif'>Data</text>")
    svg.append(f"<text x='{16}' y='{height/2:.0f}' font-size='12' text-anchor='middle' transform='rotate(-90 16,{height/2:.0f})' font-family='Arial, Helvetica, sans-serif'>{ylabel}</text>")
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


def rolling_36m_vol(dates: List[datetime], nav: List[float], min_coverage_ratio: float = 0.8) -> List[Tuple[datetime, float]]:
    out: List[Tuple[datetime, float]] = []
    target_years = 3.0
    min_years = target_years * min_coverage_ratio
    for i in range(1, len(nav)):
        end_d = dates[i]
        start_cut = end_d - timedelta(days=int(365.25 * target_years))
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
        coverage_years = sum(dt_years)
        if coverage_years < min_years:
            continue
        if not dt_years:
            continue
        mu = sum(log_rets) / coverage_years
        resid_sq_sum = 0.0
        for lr, dt in zip(log_rets, dt_years):
            resid_sq_sum += (lr - mu * dt) ** 2
        sigma = math.sqrt(max(0.0, resid_sq_sum / coverage_years))
        out.append((end_d, sigma))
    return out


def rolling_return_months(dates: List[datetime], nav: List[float], months: int) -> List[Tuple[datetime, float]]:
    days = int(round(365.25 * months / 12.0))
    out: List[Tuple[datetime, float]] = []
    for i in range(len(nav)):
        d = dates[i]
        lookup = d - timedelta(days=days)
        j = None
        for k in range(i, -1, -1):
            if dates[k] <= lookup:
                j = k
                break
        if j is None:
            continue
        if nav[j] > 0:
            ratio = nav[i] / nav[j]
            # annualized return over 'months' months
            ann = ratio ** (12.0 / months) - 1.0
            out.append((d, ann))
    return out


def rolling_vol_months(dates: List[datetime], nav: List[float], months: int, min_coverage_ratio: float = 0.8) -> List[Tuple[datetime, float]]:
    window_days = int(round(365.25 * months / 12.0))
    target_years = months / 12.0
    min_years = target_years * min_coverage_ratio
    out: List[Tuple[datetime, float]] = []
    for i in range(1, len(nav)):
        end_d = dates[i]
        start_cut = end_d - timedelta(days=window_days)
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
        coverage_years = sum(dt_years)
        if coverage_years < min_years:
            continue
        if not dt_years:
            continue
        mu = sum(log_rets) / coverage_years
        resid_sq_sum = 0.0
        for lr, dt in zip(log_rets, dt_years):
            resid_sq_sum += (lr - mu * dt) ** 2
        sigma = math.sqrt(max(0.0, resid_sq_sum / coverage_years))
        out.append((end_d, sigma))
    return out


def success_probabilities(dates: List[datetime], nav: List[float], years_list: List[int], threshold: float = 0.0) -> Dict[int, Tuple[int, int, float]]:
    result: Dict[int, Tuple[int, int, float]] = {}
    for y in years_list:
        days = int(round(365.25 * y))
        total = 0
        success = 0
        for i in range(len(nav)):
            d = dates[i]
            lookup = d - timedelta(days=days)
            j = None
            for k in range(i, -1, -1):
                if dates[k] <= lookup:
                    j = k
                    break
            if j is None:
                continue
            if nav[j] <= 0:
                continue
            r = nav[i] / nav[j] - 1.0
            total += 1
            if r > threshold:
                success += 1
        pct = (success / total * 100.0) if total > 0 else float("nan")
        result[y] = (total, success, pct)
    return result


def format_pct(x: float) -> str:
    if x != x or x is None:  # NaN check
        return "n/d"
    return f"{x*100:.2f}%"


def save_report(metrics: Dict[str, float], cal_ret: List[Tuple[int, float]], trailing: Dict[str, Tuple[float, float]], trailing_asof: Dict[str, Tuple[float, float]], asof_label: str, paths: Dict[str, str], success: Dict[int, Tuple[int, int, float]]) -> None:
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

    lines.append("## Rendimenti trailing (alla data più recente)\n")
    lines.append("Periodo | Totale | CAGR")
    lines.append("---|---|---")
    for y in [1, 3, 5, 10, 20]:
        key = f"{y}y"
        tot, cagr = trailing.get(key, (float("nan"), float("nan")))
        lines.append(f"{y} anni | {format_pct(tot)} | {format_pct(cagr)}")
    lines.append("")

    lines.append(f"## Rendimenti trailing al {asof_label}\n")
    lines.append("Periodo | Totale | CAGR")
    lines.append("---|---|---")
    for y in [1, 3, 5, 10, 20]:
        key = f"{y}y"
        tot, cagr = trailing_asof.get(key, (float("nan"), float("nan")))
        lines.append(f"{y} anni | {format_pct(tot)} | {format_pct(cagr)}")
    lines.append("")

    lines.append("## Probabilità di successo (rendimento > 0)\n")
    lines.append("Periodo | Finestre | Successi | Probabilità")
    lines.append("---|---:|---:|---:")
    for y in [1, 3, 5, 10, 15, 20]:
        if y in success:
            total, succ, pct = success[y]
            pct_str = f"{pct:.1f}%" if pct == pct else "n/d"
            lines.append(f"{y} anni | {total} | {succ} | {pct_str}")
    lines.append("")

    lines.append("## Grafici\n")
    lines.append(f"![NAV]({os.path.basename(paths['price'])})")
    lines.append("")
    lines.append(f"![Drawdown]({os.path.basename(paths['drawdown'])})")
    lines.append("")
    lines.append("### Rendimenti rolling (annualizzati)")
    lines.append(f"![Rolling 12m return]({os.path.basename(paths['roll_ret'])})")
    lines.append("")
    lines.append(f"![Rolling 36m return]({os.path.basename(paths['roll_ret_36m'])})")
    lines.append("")
    lines.append(f"![Rolling 60m return]({os.path.basename(paths['roll_ret_60m'])})")
    lines.append("")
    lines.append(f"![Rolling 120m return]({os.path.basename(paths['roll_ret_120m'])})")
    lines.append("")
    lines.append(f"![Rolling 180m return]({os.path.basename(paths['roll_ret_180m'])})")
    lines.append("")
    lines.append("### Volatilità rolling (ann.)")
    lines.append(f"![Rolling 36m vol]({os.path.basename(paths['roll_vol'])})")
    lines.append("")
    lines.append(f"![Rolling 120m vol]({os.path.basename(paths['roll_vol_120m'])})\n")

    lines.append("## Rendimento per anno\n")
    lines.append("Anno | Rendimento")
    lines.append("---|---")
    for y, r in cal_ret:
        lines.append(f"{y} | {format_pct(r)}")
    lines.append("")

    with open(paths["report"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _pdf_escape_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_text(x: float, y: float, size: float, text: str, align: str = "left") -> str:
    safe = _pdf_escape_text(text)
    # alignment handled crudely by shifting x for center/right outside (no width calc)
    return f"BT /F1 {size:.2f} Tf {x:.2f} {y:.2f} Td ({safe}) Tj ET\n"


def _pdf_line(x1: float, y1: float, x2: float, y2: float, width: float = 1.0) -> str:
    return f"{width:.2f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S\n"


def _pdf_rgb(r: float, g: float, b: float) -> str:
    return f"{r:.3f} {g:.3f} {b:.3f} RG\n"


def _pdf_fill_rgb(r: float, g: float, b: float) -> str:
    return f"{r:.3f} {g:.3f} {b:.3f} rg\n"


def _pdf_polyline(points: List[Tuple[float, float]], width: float = 1.2) -> str:
    if not points:
        return ""
    cmds = [f"{width:.2f} w {points[0][0]:.2f} {points[0][1]:.2f} m\n"]
    for x, y in points[1:]:
        cmds.append(f"{x:.2f} {y:.2f} l\n")
    cmds.append("S\n")
    return "".join(cmds)


def _pdf_rect(x: float, y: float, w: float, h: float, stroke: bool = True, fill: bool = False) -> str:
    op = "S" if stroke and not fill else ("f" if fill and not stroke else "B")
    return f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}\n"


def _pdf_table(x: float, y: float, col_widths: List[float], rows: List[List[str]], row_h: float, header_rows: int = 1) -> str:
    cmds: List[str] = []
    total_w = sum(col_widths)
    # Draw header background
    if header_rows > 0:
        for r in range(header_rows):
            cmds.append(_pdf_fill_rgb(0.94, 0.94, 0.94))
            cmds.append(_pdf_rect(x, y - (r + 1) * row_h, total_w, row_h, stroke=False, fill=True))
    # Grid
    rows_count = len(rows)
    cmds.append(_pdf_rgb(0.2, 0.2, 0.2))
    # Outer border
    cmds.append(_pdf_rect(x, y - rows_count * row_h, total_w, rows_count * row_h, stroke=True, fill=False))
    # Vertical lines
    cx = x
    for w in col_widths[:-1]:
        cx += w
        cmds.append(_pdf_line(cx, y, cx, y - rows_count * row_h, 0.8))
    # Horizontal lines
    for r in range(1, rows_count):
        yy = y - r * row_h
        cmds.append(_pdf_line(x, yy, x + total_w, yy, 0.8))
    # Text
    for r, row in enumerate(rows):
        tx = x + 6
        ty = y - r * row_h - (row_h * 0.7)
        for c, cell in enumerate(row):
            cmds.append(_pdf_rgb(0, 0, 0))
            cmds.append(_pdf_text(tx, ty, 10, str(cell)))
            tx += col_widths[c]
    return "".join(cmds)


def write_pdf_report(dates: List[datetime], nav: List[float], metrics: Dict[str, float], trailing: Dict[str, Tuple[float, float]], trailing_asof: Dict[str, Tuple[float, float]], asof_label: str, success: Dict[int, Tuple[int, int, float]], roll12: List[Tuple[datetime, float]], roll36: List[Tuple[datetime, float]], roll60: List[Tuple[datetime, float]], roll120: List[Tuple[datetime, float]], roll180: List[Tuple[datetime, float]], roll36v: List[Tuple[datetime, float]], roll120v: List[Tuple[datetime, float]], out_path: str) -> None:
    # A4 landscape
    W, H = 842.0, 595.0

    pages: List[bytes] = []

    # Page 1: Tables and summary
    content: List[str] = []
    content.append(_pdf_text(40, H - 40, 18, "Allianz Insieme – Linea Azionaria: Analisi"))
    content.append(_pdf_text(40, H - 60, 11, f"Periodo: {metrics['start_date']} → {metrics['end_date']}"))
    content.append(_pdf_text(40, H - 75, 11, f"Osservazioni: {int(metrics['observations'])}"))

    # Metrics table
    content.append(_pdf_text(40, H - 100, 13, "Metriche principali"))
    metric_rows = [
        ["Rendimento cumulato", format_pct(metrics['cumulative_return'])],
        ["CAGR", format_pct(metrics['cagr'])],
        ["Rendimento ann. (approx aritmetico)", format_pct(metrics['ann_return'])],
        ["Rendimento ann. (comp. continuo)", format_pct(metrics['ann_return_log'])],
        ["Volatilità ann.", format_pct(metrics['ann_volatility'])],
        ["Sharpe (rf=0)", f"{metrics['sharpe_ratio']:.2f}" if not math.isnan(metrics['sharpe_ratio']) else "n/d"],
        ["Sortino (rf=0)", f"{metrics['sortino_ratio']:.2f}" if not math.isnan(metrics['sortino_ratio']) else "n/d"],
        ["Max drawdown", format_pct(metrics['max_drawdown'])],
        ["Calmar", f"{metrics['calmar_ratio']:.2f}" if not math.isnan(metrics['calmar_ratio']) else "n/d"],
        ["VaR(95%) giornaliero", format_pct(metrics['var_95_daily'])],
    ]
    metric_rows = [["Voce", "Valore"]] + metric_rows
    content.append(_pdf_table(40, H - 120, [260, 140], metric_rows, 18, header_rows=1))

    # Trailing latest
    content.append(_pdf_text(440, H - 100, 13, "Rendimenti trailing (più recente)"))
    tr_rows = [["Periodo", "Totale", "CAGR"]]
    for yv in [1, 3, 5, 10, 20]:
        tot, c = trailing.get(f"{yv}y", (float('nan'), float('nan')))
        tr_rows.append([f"{yv} anni", format_pct(tot), format_pct(c)])
    content.append(_pdf_table(440, H - 120, [120, 100, 100], tr_rows, 18, header_rows=1))

    # Trailing as-of
    base_y = H - 120 - (len(tr_rows) * 18) - 20
    content.append(_pdf_text(440, base_y, 13, f"Rendimenti trailing al {asof_label}"))
    tr2_rows = [["Periodo", "Totale", "CAGR"]]
    for yv in [1, 3, 5, 10, 20]:
        tot, c = trailing_asof.get(f"{yv}y", (float('nan'), float('nan')))
        tr2_rows.append([f"{yv} anni", format_pct(tot), format_pct(c)])
    content.append(_pdf_table(440, base_y - 20, [120, 100, 100], tr2_rows, 18, header_rows=1))

    # Success table spanning width
    succ_y = base_y - 20 - (len(tr2_rows) * 18) - 20
    content.append(_pdf_text(40, succ_y, 13, "Probabilità di successo (rendimento > 0)"))
    succ_rows = [["Periodo", "Finestre", "Successi", "Probabilità"]]
    for yv in [1, 3, 5, 10, 15, 20]:
        tot, suc, pct = success.get(yv, (0, 0, float('nan')))
        pct_s = f"{pct:.1f}%" if pct == pct else "n/d"
        succ_rows.append([f"{yv} anni", str(tot), str(suc), pct_s])
    content.append(_pdf_table(40, succ_y - 20, [120, 120, 120, 140], succ_rows, 18, header_rows=1))

    # Footer note
    content.append(_pdf_text(40, 20, 9, "Nota: dati a frequenza mista (giornaliera/settimanale); metriche time-weighted."))

    pages.append(("".join(content)).encode("utf-8"))

    # Chart block helper (draw inside bbox)
    def chart_block(series: List[Tuple[datetime, float]], title: str, ylabel: str, y_is_percent: bool, bx: float, by: float, bw: float, bh: float) -> bytes:
        pad_left = 60.0
        pad_bottom = 50.0
        pad_right = 20.0
        pad_top = 30.0
        inner_x = bx + pad_left
        inner_y = by + pad_bottom
        inner_w = bw - (pad_left + pad_right)
        inner_h = bh - (pad_top + pad_bottom)
        xs = [d.timestamp() for d, _ in series]
        ys = [v for _, v in series]
        if not xs:
            xs = [0.0, 1.0]
            ys = [0.0, 1.0]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        if y_min == y_max:
            y_min -= 0.5
            y_max += 0.5
        def sx(xv: float) -> float:
            return inner_x + (xv - x_min) / (x_max - x_min) * inner_w
        def sy(yv: float) -> float:
            return inner_y + (yv - y_min) / (y_max - y_min) * inner_h
        cmds: List[str] = []
        # frame
        cmds.append(_pdf_rgb(0.2, 0.2, 0.2))
        cmds.append(_pdf_rect(bx, by, bw, bh, stroke=True, fill=False))
        # axes
        cmds.append(_pdf_rgb(0.27, 0.27, 0.27))
        cmds.append(_pdf_line(inner_x, inner_y, inner_x + inner_w, inner_y, 1.0))
        cmds.append(_pdf_line(inner_x, inner_y, inner_x, inner_y + inner_h, 1.0))
        # grid
        x_ticks = 8
        y_ticks = 6
        for i in range(y_ticks):
            yv = y_min + i * (y_max - y_min) / (y_ticks - 1)
            yy = sy(yv)
            cmds.append(_pdf_rgb(0.9, 0.9, 0.9))
            cmds.append(_pdf_line(inner_x, yy, inner_x + inner_w, yy, 0.6))
            label = f"{yv*100:.1f}%" if y_is_percent else f"{yv:.2f}"
            cmds.append(_pdf_rgb(0.33, 0.33, 0.33))
            cmds.append(_pdf_text(inner_x - 40, yy - 4, 9, label))
        for i in range(x_ticks):
            xv = x_min + i * (x_max - x_min) / (x_ticks - 1)
            xx = sx(xv)
            cmds.append(_pdf_rgb(0.9, 0.9, 0.9))
            cmds.append(_pdf_line(xx, inner_y, xx, inner_y + inner_h, 0.6))
            dt = datetime.utcfromtimestamp(xv)
            cmds.append(_pdf_rgb(0.33, 0.33, 0.33))
            cmds.append(_pdf_text(xx - 12, inner_y - 16, 9, dt.strftime('%Y'))) 
        # series
        pts = [(sx(xv), sy(yv)) for xv, yv in zip(xs, ys)]
        cmds.append(_pdf_rgb(0.12, 0.47, 0.71))
        cmds.append(_pdf_polyline(pts, 1.2))
        # titles
        cmds.append(_pdf_text(bx + 10, by + bh - 18, 12, title))
        cmds.append(_pdf_text(bx + bw/2 - 20, by + 10, 10, "Data"))
        cmds.append(_pdf_text(bx + 10, by + bh/2, 10, ylabel))
        return ("".join(cmds)).encode("utf-8")

    # Charts arranged two per page (stacked)
    chart_pages: List[List[Tuple[List[Tuple[datetime, float]], str, str, bool]]] = [
        [ (list(zip(dates, nav)), "Valore quota (NAV)", "NAV (EUR)", False),
          ([(d, v) for d, v in zip(dates, compute_drawdown(nav)[0])], "Drawdown", "Drawdown", True) ],
        [ (roll12, "Rendimento rolling 12 mesi (ann.)", "Rendimento ann.", True),
          (roll36, "Rendimento rolling 36 mesi (ann.)", "Rendimento ann.", True) ],
        [ (roll60, "Rendimento rolling 60 mesi (ann.)", "Rendimento ann.", True),
          (roll120, "Rendimento rolling 120 mesi (ann.)", "Rendimento ann.", True) ],
        [ (roll180, "Rendimento rolling 180 mesi (ann.)", "Rendimento ann.", True),
          (roll36v, "Volatilità rolling 36 mesi (ann.)", "Volatilità ann.", True) ],
        [ (roll120v, "Volatilità rolling 120 mesi (ann.)", "Volatilità ann.", True) ],
    ]

    for page_blocks in chart_pages:
        cmds: List[bytes] = []
        top_box = (40.0, H - 40.0 - 240.0, W - 80.0, 240.0)
        bottom_box = (40.0, 40.0, W - 80.0, 240.0)
        # First block
        series, title, ylabel, yperc = page_blocks[0]
        cmds.append(chart_block(series, title, ylabel, yperc, *top_box))
        # Second block if exists
        if len(page_blocks) > 1:
            series2, title2, ylabel2, yperc2 = page_blocks[1]
            cmds.append(chart_block(series2, title2, ylabel2, yperc2, *bottom_box))
        pages.append(b"".join(cmds))

    # Assemble PDF (same as before)
    objects: List[bytes] = []
    xref: List[int] = []

    def add_object(obj_str: str) -> int:
        xref.append(sum(len(o) for o in objects))
        objects.append(obj_str.encode("utf-8"))
        return len(objects)

    # Font object
    font_obj_num = add_object("1 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    # Page content streams
    content_obj_nums: List[int] = []
    for p in pages:
        stream = p
        header = f"<< /Length {len(stream)} >>\nstream\n".encode("utf-8")
        footer = b"endstream\nendobj\n"
        xref.append(sum(len(o) for o in objects))
        objects.append(b"")
        obj_index = len(objects)
        objects[obj_index - 1] = f"{obj_index} 0 obj\n".encode("utf-8") + header + stream + footer
        content_obj_nums.append(obj_index)

    # Pages objects
    page_obj_nums: List[int] = []
    for content_num in content_obj_nums:
        page_index = len(objects) + 1
        xref.append(sum(len(o) for o in objects))
        page_obj = (
            f"{page_index} 0 obj\n"
            f"<< /Type /Page /Parent 0 0 R /MediaBox [0 0 {W:.0f} {H:.0f}] "
            f"/Resources << /Font << /F1 {font_obj_num} 0 R >> >> /Contents {content_num} 0 R >>\n"
            f"endobj\n"
        )
        objects.append(page_obj.encode("utf-8"))
        page_obj_nums.append(page_index)

    # Pages root
    pages_root_index = len(objects) + 1
    kids_ref = " ".join([f"{n} 0 R" for n in page_obj_nums])
    xref.append(sum(len(o) for o in objects))
    pages_root_obj = (
        f"{pages_root_index} 0 obj\n"
        f"<< /Type /Pages /Kids [ {kids_ref} ] /Count {len(page_obj_nums)} >>\n"
        f"endobj\n"
    )
    objects.append(pages_root_obj.encode("utf-8"))

    # Catalog
    catalog_index = len(objects) + 1
    xref.append(sum(len(o) for o in objects))
    catalog_obj = (
        f"{catalog_index} 0 obj\n"
        f"<< /Type /Catalog /Pages {pages_root_index} 0 R >>\n"
        f"endobj\n"
    )
    objects.append(catalog_obj.encode("utf-8"))

    # Fix Parent refs
    fixed_objects: List[bytes] = []
    offset = 0
    offsets: List[int] = []
    for i, obj in enumerate(objects, start=1):
        if i in page_obj_nums:
            s = obj.decode("utf-8").replace("/Parent 0 0 R", f"/Parent {pages_root_index} 0 R").encode("utf-8")
            fixed_objects.append(s)
        else:
            fixed_objects.append(obj)
    # Write PDF
    with open(out_path, "wb") as f:
        f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        byte_count = 0
        for obj in fixed_objects:
            offsets.append(byte_count)
            f.write(obj)
            byte_count += len(obj)
        xref_start = byte_count
        f.write(f"xref\n0 {len(fixed_objects)+1}\n".encode("utf-8"))
        f.write(b"0000000000 65535 f \n")
        for off in offsets:
            f.write(f"{off:010d} 00000 n \n".encode("utf-8"))
        f.write(b"trailer\n")
        f.write(f"<< /Size {len(fixed_objects)+1} /Root {catalog_index} 0 R >>\n".encode("utf-8"))
        f.write(f"startxref\n{xref_start}\n%%EOF\n".encode("utf-8"))


def _xml_escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace("\"", "&quot;")
         .replace("'", "&apos;")
    )


def _col_letter(n: int) -> str:
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s or "A"


def _worksheet_xml(headers: List[str], rows: List[List[object]]) -> str:
    parts: List[str] = []
    parts.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>")
    parts.append("<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\">")
    parts.append("<sheetData>")
    # Header row
    parts.append("<row r=\"1\">")
    for ci, h in enumerate(headers, start=1):
        cell_ref = f"{_col_letter(ci)}1"
        parts.append(f"<c r=\"{cell_ref}\" t=\"inlineStr\"><is><t>{_xml_escape(str(h))}</t></is></c>")
    parts.append("</row>")
    # Data rows
    for ri, row in enumerate(rows, start=2):
        parts.append(f"<row r=\"{ri}\">")
        for ci, val in enumerate(row, start=1):
            cell_ref = f"{_col_letter(ci)}{ri}"
            if val is None:
                parts.append(f"<c r=\"{cell_ref}\"/>")
            elif isinstance(val, (int, float)) and not (isinstance(val, float) and (val != val)):
                parts.append(f"<c r=\"{cell_ref}\"><v>{val}</v></c>")
            else:
                parts.append(f"<c r=\"{cell_ref}\" t=\"inlineStr\"><is><t>{_xml_escape(str(val))}</t></is></c>")
        parts.append("</row>")
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def write_excel_report(dates: List[datetime], nav: List[float], metrics: Dict[str, float], cal_ret: List[Tuple[int, float]], trailing: Dict[str, Tuple[float, float]], trailing_asof: Dict[str, Tuple[float, float]], asof_label: str, success: Dict[int, Tuple[int, int, float]], roll12: List[Tuple[datetime, float]], roll36: List[Tuple[datetime, float]], roll60: List[Tuple[datetime, float]], roll120: List[Tuple[datetime, float]], roll180: List[Tuple[datetime, float]], roll36v: List[Tuple[datetime, float]], roll120v: List[Tuple[datetime, float]], out_path: str, image_paths: Dict[str, str]) -> None:
    sheets: List[Tuple[str, str]] = []
    # NAV sheet
    nav_rows = [[d.strftime('%Y-%m-%d'), v] for d, v in zip(dates, nav)]
    sheets.append(("NAV", _worksheet_xml(["Data", "NAV"], nav_rows)))
    # Metrics sheet
    met_rows = [["Rendimento cumulato", format_pct(metrics['cumulative_return'])],
                ["CAGR", format_pct(metrics['cagr'])],
                ["Rendimento ann. (approx aritmetico)", format_pct(metrics['ann_return'])],
                ["Rendimento ann. (comp. continuo)", format_pct(metrics['ann_return_log'])],
                ["Volatilità ann.", format_pct(metrics['ann_volatility'])],
                ["Sharpe (rf=0)", f"{metrics['sharpe_ratio']:.2f}" if not math.isnan(metrics['sharpe_ratio']) else "n/d"],
                ["Sortino (rf=0)", f"{metrics['sortino_ratio']:.2f}" if not math.isnan(metrics['sortino_ratio']) else "n/d"],
                ["Max drawdown", format_pct(metrics['max_drawdown'])],
                ["Calmar", f"{metrics['calmar_ratio']:.2f}" if not math.isnan(metrics['calmar_ratio']) else "n/d"],
                ["VaR(95%) giornaliero", format_pct(metrics['var_95_daily'])]]
    sheets.append(("Metriche", _worksheet_xml(["Voce", "Valore"], met_rows)))
    # Calendar returns
    cal_rows = [[y, float(r)] for y, r in cal_ret]
    sheets.append(("Rend_Anno", _worksheet_xml(["Anno", "Rendimento"], cal_rows)))
    # Trailing latest
    tr_rows = [[f"{y} anni", format_pct(trailing.get(f"{y}y", (float('nan'),))[0]), format_pct(trailing.get(f"{y}y", (0, float('nan')))[1])] for y in [1,3,5,10,20]]
    sheets.append(("Trailing_Recent", _worksheet_xml(["Periodo", "Totale", "CAGR"], tr_rows)))
    # Trailing as-of
    tr2_rows = [[f"{y} anni", format_pct(trailing_asof.get(f"{y}y", (float('nan'),))[0]), format_pct(trailing_asof.get(f"{y}y", (0, float('nan')))[1])] for y in [1,3,5,10,20]]
    sheets.append(("Trailing_2024-12-31", _worksheet_xml(["Periodo", "Totale", "CAGR"], tr2_rows)))
    # Success probabilities
    succ_rows = [[y, total, succ, (f"{pct:.1f}%" if pct == pct else "n/d")] for y,(total,succ,pct) in success.items()]
    succ_rows.sort(key=lambda x: x[0])
    sheets.append(("Successo", _worksheet_xml(["Periodo(anni)", "Finestre", "Successi", "Probabilità"], succ_rows)))
    # Rolling returns sheet (aligned)
    ret_map: Dict[str, Dict[str, float]] = {}
    for label, series in [("12m", roll12), ("36m", roll36), ("60m", roll60), ("120m", roll120), ("180m", roll180)]:
        for d, v in series:
            key = d.strftime('%Y-%m-%d')
            ret_map.setdefault(key, {})[label] = float(v)
    dates_sorted = sorted(ret_map.keys())
    rows_ret: List[List[object]] = []
    for ds in dates_sorted:
        row = [ds]
        for label in ["12m","36m","60m","120m","180m"]:
            row.append(ret_map[ds].get(label))
        rows_ret.append(row)
    sheets.append(("Rolling_Returns", _worksheet_xml(["Data","12m(ann.)","36m(ann.)","60m(ann.)","120m(ann.)","180m(ann.)"], rows_ret)))
    # Rolling vol sheet
    vol_map: Dict[str, Dict[str, float]] = {}
    for label, series in [("36m", roll36v), ("120m", roll120v)]:
        for d, v in series:
            key = d.strftime('%Y-%m-%d')
            vol_map.setdefault(key, {})[label] = float(v)
    dsorted = sorted(vol_map.keys())
    rows_vol: List[List[object]] = []
    for ds in dsorted:
        rows_vol.append([ds, vol_map[ds].get("36m"), vol_map[ds].get("120m")])
    sheets.append(("Rolling_Vol", _worksheet_xml(["Data","Vol36m(ann.)","Vol120m(ann.)"], rows_vol)))

    # Charts sheet with images
    chart_keys = [
        "price","drawdown","roll_ret","roll_ret_36m","roll_ret_60m",
        "roll_ret_120m","roll_ret_180m","roll_vol","roll_vol_120m"
    ]
    images: List[Tuple[str, bytes]] = []
    for idx, key in enumerate(chart_keys, start=1):
        p = image_paths.get(key)
        if not p:
            continue
        # Prefer PNG in output/png
        base = os.path.splitext(os.path.basename(p))[0]
        png_path = os.path.join(os.path.dirname(p), "png", f"{base}.png")
        use_path = png_path if os.path.exists(png_path) else p
        with open(use_path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(use_path)[1].lower()
        fname = f"image{idx}{ext}"
        images.append((fname, data))
    # Minimal charts sheet XML with drawing rel (rId1)
    charts_sheet_xml = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
        "<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
        "<sheetData/></worksheet>"
    )
    charts_sheet_name = "Charts"
    sheets.append((charts_sheet_name, charts_sheet_xml))
    charts_sheet_index = len(sheets)  # 1-based index in workbook order

    # Build XLSX package
    with ZipFile(out_path, 'w', ZIP_DEFLATED) as z:
        # [Content_Types].xml
        ct = [
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">",
            "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>",
            "<Default Extension=\"xml\" ContentType=\"application/xml\"/>",
            "<Default Extension=\"png\" ContentType=\"image/png\"/>",
            "<Default Extension=\"svg\" ContentType=\"image/svg+xml\"/>",
            "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>",
            "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>",
            "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>",
            "<Override PartName=\"/xl/drawings/drawing1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.drawing+xml\"/>",
        ]
        for i in range(len(sheets)):
            ct.append(f"<Override PartName=\"/xl/worksheets/sheet{i+1}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>")
        ct.append("</Types>")
        z.writestr("[Content_Types].xml", "".join(ct))
        # _rels/.rels
        z.writestr("_rels/.rels", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>",
            "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>",
            "<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/>",
            "</Relationships>"
        ]))
        # docProps/core.xml
        z.writestr("docProps/core.xml", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">",
            "<dc:title>Allianz Insieme – Analisi</dc:title>",
            f"<dc:creator>AutoReport</dc:creator>",
            "</cp:coreProperties>"
        ]))
        # docProps/app.xml
        z.writestr("docProps/app.xml", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">",
            "<Application>AutoReport</Application>",
            "</Properties>"
        ]))
        # xl/workbook.xml and rels
        sheets_xml = []
        rels_xml = []
        for i, (name, _) in enumerate(sheets, start=1):
            sheets_xml.append(f"<sheet name=\"{_xml_escape(name)}\" sheetId=\"{i}\" r:id=\"rId{i}\"/>")
            rels_xml.append(f"<Relationship Id=\"rId{i}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet{i}.xml\"/>")
        z.writestr("xl/workbook.xml", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">",
            "<sheets>", "".join(sheets_xml), "</sheets>", "</workbook>"
        ]))
        z.writestr("xl/_rels/workbook.xml.rels", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
            "".join(rels_xml),
            "</Relationships>"
        ]))
        # xl/worksheets/sheetN.xml
        for i, (_, ws_xml) in enumerate(sheets, start=1):
            if i == charts_sheet_index:
                # inject drawing tag before closing worksheet
                ws_xml = ws_xml.replace("</worksheet>", "<drawing r:id=\"rId1\"/></worksheet>")
            z.writestr(f"xl/worksheets/sheet{i}.xml", ws_xml)
        # Rel for charts sheet -> drawing1.xml
        z.writestr(f"xl/worksheets/_rels/sheet{charts_sheet_index}.xml.rels", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing\" Target=\"../drawings/drawing1.xml\"/>",
            "</Relationships>"
        ]))
        # xl/drawings/drawing1.xml and rels
        anchors: List[str] = []
        rels: List[str] = []
        row_cursor = 0
        for i, (fname, _) in enumerate(images, start=1):
            from_col, to_col = 0, 10
            from_row = row_cursor
            to_row = row_cursor + 18
            row_cursor += 20
            anchors.append("".join([
                "<xdr:twoCellAnchor>",
                "<xdr:from><xdr:col>", str(from_col), "</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>", str(from_row), "</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:from>",
                "<xdr:to><xdr:col>", str(to_col), "</xdr:col><xdr:colOff>0</xdr:colOff><xdr:row>", str(to_row), "</xdr:row><xdr:rowOff>0</xdr:rowOff></xdr:to>",
                "<xdr:pic>",
                f"<xdr:nvPicPr><xdr:cNvPr id=\"{i}\" name=\"Picture {i}\"/></xdr:nvPicPr>",
                "<xdr:blipFill>",
                f"<a:blip r:embed=\"rId{i}\"/>",
                "<a:stretch><a:fillRect/></a:stretch>",
                "</xdr:blipFill>",
                "<xdr:spPr><a:prstGeom prst=\"rect\"><a:avLst/></a:prstGeom></xdr:spPr>",
                "</xdr:pic>",
                "<xdr:clientData/>",
                "</xdr:twoCellAnchor>"
            ]))
            rels.append(f"<Relationship Id=\"rId{i}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"../media/{fname}\"/>")
        drawing_xml = "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<xdr:wsDr xmlns:xdr=\"http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing\" xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">",
            "".join(anchors),
            "</xdr:wsDr>"
        ])
        z.writestr("xl/drawings/drawing1.xml", drawing_xml)
        z.writestr("xl/drawings/_rels/drawing1.xml.rels", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
            "".join(rels),
            "</Relationships>"
        ]))
        # Media files
        for fname, data in images:
            z.writestr(f"xl/media/{fname}", data)


def _excel_serial_date(dt: datetime) -> int:
    epoch = datetime(1899, 12, 30)
    return int((dt - epoch).total_seconds() // 86400)


def _worksheet_xml_cells(rows: List[List[Dict[str, object]]]) -> str:
    parts: List[str] = []
    parts.append("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>")
    parts.append("<worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">")
    parts.append("<sheetData>")
    for ri, row in enumerate(rows, start=1):
        parts.append(f"<row r=\"{ri}\">")
        for ci, cell in enumerate(row, start=1):
            ref = f"{_col_letter(ci)}{ri}"
            ctype = cell.get("t")
            formula = cell.get("f")
            value = cell.get("v")
            if formula:
                parts.append(f"<c r=\"{ref}\"><f>{_xml_escape(str(formula))}</f>")
                if value is not None:
                    parts.append(f"<v>{value}</v>")
                parts.append("</c>")
            elif ctype == "s":
                parts.append(f"<c r=\"{ref}\" t=\"inlineStr\"><is><t>{_xml_escape(str(value) if value is not None else '')}</t></is></c>")
            elif value is None:
                parts.append(f"<c r=\"{ref}\"/>")
            else:
                parts.append(f"<c r=\"{ref}\"><v>{value}</v></c>")
        parts.append("</row>")
    parts.append("</sheetData></worksheet>")
    return "".join(parts)


def write_excel_report_with_formulas(dates: List[datetime], nav: List[float], out_path: str) -> None:
    n = len(dates)
    last_row = n + 1  # because header at row 1

    # NAV sheet: numeric date serials and nav values
    nav_rows: List[List[Dict[str, object]]] = []
    nav_rows.append([{"t": "s", "v": "Data"}, {"t": "s", "v": "NAV"}])
    for i in range(n):
        nav_rows.append([{"v": _excel_serial_date(dates[i])}, {"v": float(nav[i])}])
    nav_xml = _worksheet_xml_cells(nav_rows)

    # Calc sheet with formulas referencing NAV
    calc_rows: List[List[Dict[str, object]]] = []
    calc_rows.append([
        {"t": "s", "v": "Data"}, {"t": "s", "v": "NAV"}, {"t": "s", "v": "log_r"},
        {"t": "s", "v": "dt_days"}, {"t": "s", "v": "dt_years"}, {"t": "s", "v": ""},
        {"t": "s", "v": ""}, {"t": "s", "v": ""}, {"t": "s", "v": "per_day_log"},
        {"t": "s", "v": "r_simple"}, {"t": "s", "v": "runmax"}, {"t": "s", "v": "drawdown"}
    ])
    # Row 2 (first data row)
    calc_rows.append([
        {"f": "NAV!A2"}, {"f": "NAV!B2"}, {"t": "s", "v": ""}, {"t": "s", "v": ""}, {"t": "s", "v": ""},
        {"t": "s", "v": ""}, {"t": "s", "v": ""}, {"t": "s", "v": ""}, {"t": "s", "v": ""},
        {"t": "s", "v": ""}, {"f": "B2"}, {"f": "IF(B2>0,B2/K2-1,\"\")"}
    ])
    # Rows 3..last_row
    for r in range(3, last_row + 1):
        calc_rows.append([
            {"f": f"NAV!A{r}"},
            {"f": f"NAV!B{r}"},
            {"f": f"IFERROR(LN(B{r}/B{r-1}),\"\")"},
            {"f": f"IFERROR(A{r}-A{r-1},\"\")"},
            {"f": f"IFERROR(D{r}/365.25,\"\")"},
            {"t": "s", "v": ""},
            {"t": "s", "v": ""},
            {"t": "s", "v": ""},
            {"f": f"IF(D{r}>0,C{r}/D{r},\"\")"},
            {"f": f"IFERROR(B{r}/B{r-1}-1,\"\")"},
            {"f": f"MAX(K{r-1},B{r})"},
            {"f": f"IF(K{r}>0,B{r}/K{r}-1,\"\")"},
        ])
    calc_xml = _worksheet_xml_cells(calc_rows)

    # Summary sheet with formulas
    c_start = 3
    c_end = last_row
    summary_rows: List[List[Dict[str, object]]] = []
    summary_rows.append([{"t": "s", "v": "Voce"}, {"t": "s", "v": "Valore"}])
    summary_rows.append([{ "t": "s", "v": "First NAV"}, {"f": "NAV!B2"}])
    summary_rows.append([{ "t": "s", "v": "Last NAV"}, {"f": "LOOKUP(2,1/(NAV!B:B<>\"\"),NAV!B:B)"}])
    summary_rows.append([{ "t": "s", "v": "First Date"}, {"f": "NAV!A2"}])
    summary_rows.append([{ "t": "s", "v": "Last Date"}, {"f": "LOOKUP(2,1/(NAV!A:A<>\"\"),NAV!A:A)"}])
    summary_rows.append([{ "t": "s", "v": "Rendimento cumulato"}, {"f": "B3/B2-1"}])
    summary_rows.append([{ "t": "s", "v": "CAGR"}, {"f": "POWER(B3/B2,365.25/(B5-B4))-1"}])
    summary_rows.append([{ "t": "s", "v": "mu (log/yr)"}, {"f": f"SUM(Calc!C{c_start}:C{c_end})/SUM(Calc!E{c_start}:E{c_end})" }])
    summary_rows.append([{ "t": "s", "v": "sigma (ann.)"}, {"f": f"SQRT(SUMPRODUCT((Calc!C{c_start}:C{c_end}-$B$8*Calc!E{c_start}:E{c_end})^2)/SUM(Calc!E{c_start}:E{c_end}))" }])
    summary_rows.append([{ "t": "s", "v": "Rendimento ann. (approx)"}, {"f": "$B$8+0.5*($B$9^2)" }])
    summary_rows.append([{ "t": "s", "v": "Sharpe (rf=0)"}, {"f": "IF($B$9>0,$B$10/$B$9,NA())" }])
    summary_rows.append([{ "t": "s", "v": "Downside dev (ann.)"}, {"f": f"SQRT(SUMPRODUCT((Calc!C{c_start}:C{c_end}-$B$8*Calc!E{c_start}:E{c_end})^2*(Calc!J{c_start}:J{c_end}<0))/SUMPRODUCT(Calc!E{c_start}:E{c_end}*(Calc!J{c_start}:J{c_end}<0)))" }])
    summary_rows.append([{ "t": "s", "v": "Sortino (rf=0)"}, {"f": "IF($B$12>0,$B$10/$B$12,NA())" }])
    summary_rows.append([{ "t": "s", "v": "Max drawdown"}, {"f": f"MIN(Calc!L2:L{c_end})" }])
    summary_rows.append([{ "t": "s", "v": "Calmar"}, {"f": "IF(B14<0,$B$7/ABS(B14),NA())" }])
    summary_rows.append([{ "t": "s", "v": "VaR(95%) giornaliero"}, {"f": f"EXP(PERCENTILE.INC(Calc!I{c_start}:I{c_end},0.05))-1" }])

    # Trailing latest block
    summary_rows.append([{ "t": "s", "v": "Trailing (più recente)"}, {"t": "s", "v": "" }])
    for y in [1,3,5,10,20]:
        summary_rows.append([
            {"t": "s", "v": f"{y} anni Tot"},
            {"f": f"B3/LOOKUP(EDATE($B$5,-{12*y}),NAV!A:A,NAV!B:B)-1"}
        ])
        summary_rows.append([
            {"t": "s", "v": f"{y} anni CAGR"},
            {"f": f"POWER(B3/LOOKUP(EDATE($B$5,-{12*y}),NAV!A:A,NAV!B:B),1/{y})-1"}
        ])

    # Trailing as-of 31/12/2024
    summary_rows.append([{ "t": "s", "v": f"As-of"}, {"f": "DATE(2024,12,31)" }])
    summary_rows.append([{ "t": "s", "v": "NAV as-of"}, {"f": "LOOKUP($B$22,NAV!A:A,NAV!B:B)" }])
    for y in [1,3,5,10,20]:
        summary_rows.append([
            {"t": "s", "v": f"{y} anni Tot (as-of)"},
            {"f": f"$B$23/LOOKUP(EDATE($B$22,-{12*y}),NAV!A:A,NAV!B:B)-1"}
        ])
        summary_rows.append([
            {"t": "s", "v": f"{y} anni CAGR (as-of)"},
            {"f": f"POWER($B$23/LOOKUP(EDATE($B$22,-{12*y}),NAV!A:A,NAV!B:B),1/{y})-1"}
        ])

    summary_xml = _worksheet_xml_cells(summary_rows)

    # Rolling returns formulas sheet
    rr_rows: List[List[Dict[str, object]]] = []
    rr_rows.append([
        {"t": "s", "v": "Data"}, {"t": "s", "v": "12m(ann.)"}, {"t": "s", "v": "36m(ann.)"},
        {"t": "s", "v": "60m(ann.)"}, {"t": "s", "v": "120m(ann.)"}, {"t": "s", "v": "180m(ann.)"}
    ])
    for r in range(2, last_row + 1):
        row: List[Dict[str, object]] = []
        row.append({"f": f"NAV!A{r}"})
        for m in [12,36,60,120,180]:
            row.append({"f": f"IFERROR(POWER(NAV!B{r}/LOOKUP(EDATE(NAV!A{r},-{m}),NAV!A:A,NAV!B:B),{12/m})-1,\"\")"})
        rr_rows.append(row)
    rr_xml = _worksheet_xml_cells(rr_rows)

    # Rolling volatility formulas sheet (coverage-check 80%)
    rv_rows: List[List[Dict[str, object]]] = []
    rv_rows.append([
        {"t": "s", "v": "Data"}, {"t": "s", "v": "Vol36m(ann.)"}, {"t": "s", "v": "Vol120m(ann.)"}
    ])
    for r in range(2, last_row + 1):
        date_ref = f"NAV!A{r}"
        # 36m: need >= 2.4 years
        f36 = (
            f"LET(d,{date_ref},s,EDATE(d,-36),inc,(Calc!A:A>s)*(Calc!A:A<=d),"
            f"arr,FILTER(Calc!I:I,inc),yrs,SUM(FILTER(Calc!E:E,inc)),"
            f"IF(OR(ROWS(arr)<2,yrs<2.4),\"\",STDEV.S(arr)*SQRT(365.25)))"
        )
        # 120m: need >= 8 years
        f120 = (
            f"LET(d,{date_ref},s,EDATE(d,-120),inc,(Calc!A:A>s)*(Calc!A:A<=d),"
            f"arr,FILTER(Calc!I:I,inc),yrs,SUM(FILTER(Calc!E:E,inc)),"
            f"IF(OR(ROWS(arr)<2,yrs<8),\"\",STDEV.S(arr)*SQRT(365.25)))"
        )
        rv_rows.append([{"f": date_ref}, {"f": f36}, {"f": f120}])
    rv_xml = _worksheet_xml_cells(rv_rows)

    sheets: List[Tuple[str, str]] = [
        ("NAV", nav_xml), ("Calc", calc_xml), ("Summary_Formulas", summary_xml),
        ("Rolling_Returns_Formulas", rr_xml), ("Rolling_Vol_Formulas", rv_xml)
    ]

    with ZipFile(out_path, 'w', ZIP_DEFLATED) as z:
        # [Content_Types]
        ct = [
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">",
            "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>",
            "<Default Extension=\"xml\" ContentType=\"application/xml\"/>",
            "<Override PartName=\"/xl/workbook.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml\"/>",
            "<Override PartName=\"/docProps/app.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.extended-properties+xml\"/>",
            "<Override PartName=\"/docProps/core.xml\" ContentType=\"application/vnd.openxmlformats-package.core-properties+xml\"/>",
        ]
        for i in range(len(sheets)):
            ct.append(f"<Override PartName=\"/xl/worksheets/sheet{i+1}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml\"/>")
        ct.append("</Types>")
        z.writestr("[Content_Types].xml", "".join(ct))
        # _rels
        z.writestr("_rels/.rels", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"xl/workbook.xml\"/>",
            "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties\" Target=\"docProps/core.xml\"/>",
            "<Relationship Id=\"rId3\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties\" Target=\"docProps/app.xml\"/>",
            "</Relationships>"
        ]))
        # docProps
        z.writestr("docProps/core.xml", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<cp:coreProperties xmlns:cp=\"http://schemas.openxmlformats.org/package/2006/metadata/core-properties\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:dcterms=\"http://purl.org/dc/terms/\" xmlns:dcmitype=\"http://purl.org/dc/dcmitype/\" xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\">",
            "<dc:title>Allianz Insieme – Analisi (Formule)</dc:title>",
            "<dc:creator>AutoReport</dc:creator>",
            "</cp:coreProperties>"
        ]))
        z.writestr("docProps/app.xml", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Properties xmlns=\"http://schemas.openxmlformats.org/officeDocument/2006/extended-properties\" xmlns:vt=\"http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes\">",
            "<Application>AutoReport</Application>",
            "</Properties>"
        ]))
        # workbook
        sheets_xml = []
        rels_xml = []
        for i, (name, _) in enumerate(sheets, start=1):
            sheets_xml.append(f"<sheet name=\"{_xml_escape(name)}\" sheetId=\"{i}\" r:id=\"rId{i}\"/>")
            rels_xml.append(f"<Relationship Id=\"rId{i}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\" Target=\"worksheets/sheet{i}.xml\"/>")
        z.writestr("xl/workbook.xml", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">",
            "<sheets>", "".join(sheets_xml), "</sheets>", "</workbook>"
        ]))
        z.writestr("xl/_rels/workbook.xml.rels", "".join([
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>",
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">",
            "".join(rels_xml),
            "</Relationships>"
        ]))
        # sheets content
        for i, (_, ws_xml) in enumerate(sheets, start=1):
            z.writestr(f"xl/worksheets/sheet{i}.xml", ws_xml)


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

    ann_return_log = math.expm1(mu) if mu == mu else float("nan")
    ann_return = mu + 0.5 * (sigma * sigma) if (mu == mu and sigma == sigma) else float("nan")

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
    roll12 = rolling_return_months(dates, nav, 12)
    roll36 = rolling_return_months(dates, nav, 36)
    roll60 = rolling_return_months(dates, nav, 60)
    roll120 = rolling_return_months(dates, nav, 120)
    roll180 = rolling_return_months(dates, nav, 180)
    roll36v = rolling_36m_vol(dates, nav, min_coverage_ratio=0.8)
    roll120v = rolling_vol_months(dates, nav, 120, min_coverage_ratio=0.8)

    trailing = {
        "1y": trailing_return(dates, nav, 1),
        "3y": trailing_return(dates, nav, 3),
        "5y": trailing_return(dates, nav, 5),
        "10y": trailing_return(dates, nav, 10),
        "20y": trailing_return(dates, nav, 20),
    }

    asof_date = datetime(2024, 12, 31)
    trailing_asof = compute_trailing_returns_asof(dates, nav, asof_date)

    success = success_probabilities(dates, nav, years_list=[1,3,5,10,15,20], threshold=0.0)

    print("Genero grafici SVG…")
    paths = {
        "price": os.path.join(OUTPUT_DIR, "price.svg"),
        "drawdown": os.path.join(OUTPUT_DIR, "drawdown.svg"),
        "roll_ret": os.path.join(OUTPUT_DIR, "rolling_12m_return.svg"),
        "roll_ret_36m": os.path.join(OUTPUT_DIR, "rolling_36m_return.svg"),
        "roll_ret_60m": os.path.join(OUTPUT_DIR, "rolling_60m_return.svg"),
        "roll_ret_120m": os.path.join(OUTPUT_DIR, "rolling_120m_return.svg"),
        "roll_ret_180m": os.path.join(OUTPUT_DIR, "rolling_180m_return.svg"),
        "roll_vol": os.path.join(OUTPUT_DIR, "rolling_36m_vol.svg"),
        "roll_vol_120m": os.path.join(OUTPUT_DIR, "rolling_120m_vol.svg"),
        "report": os.path.join(OUTPUT_DIR, "report.md"),
        "xlsx": os.path.join(OUTPUT_DIR, "report.xlsx"),
        "xlsx_formulas": os.path.join(OUTPUT_DIR, "report_formulas.xlsx"),
    }

    write_svg_price(dates, nav, paths["price"])
    write_svg_drawdown(dates, nav, paths["drawdown"])
    write_svg_series(roll12, "#2ca02c", 920, 360, paths["roll_ret"], title="Rendimento rolling 12 mesi (ann.)", ylabel="Rendimento ann.", y_is_percent=True, x_ticks=12, y_ticks=8)
    write_svg_series(roll36, "#2ca02c", 920, 360, paths["roll_ret_36m"], title="Rendimento rolling 36 mesi (ann.)", ylabel="Rendimento ann.", y_is_percent=True, x_ticks=12, y_ticks=8)
    write_svg_series(roll60, "#2ca02c", 920, 360, paths["roll_ret_60m"], title="Rendimento rolling 60 mesi (ann.)", ylabel="Rendimento ann.", y_is_percent=True, x_ticks=12, y_ticks=8)
    write_svg_series(roll120, "#2ca02c", 920, 360, paths["roll_ret_120m"], title="Rendimento rolling 120 mesi (ann.)", ylabel="Rendimento ann.", y_is_percent=True, x_ticks=12, y_ticks=8)
    write_svg_series(roll180, "#2ca02c", 920, 360, paths["roll_ret_180m"], title="Rendimento rolling 180 mesi (ann.)", ylabel="Rendimento ann.", y_is_percent=True, x_ticks=12, y_ticks=8)
    write_svg_series(roll36v, "#9467bd", 920, 360, paths["roll_vol"], title="Volatilità rolling 36 mesi (ann.)", ylabel="Volatilità ann.", y_is_percent=True, x_ticks=12, y_ticks=8)
    write_svg_series(roll120v, "#9467bd", 920, 360, paths["roll_vol_120m"], title="Volatilità rolling 120 mesi (ann.)", ylabel="Volatilità ann.", y_is_percent=True, x_ticks=12, y_ticks=8)

    print("Genero Excel…")
    write_excel_report(dates, nav, metrics, calendar_year_returns(dates, nav), trailing, trailing_asof, "31/12/2024", success, roll12, roll36, roll60, roll120, roll180, roll36v, roll120v, paths["xlsx"], paths)

    print("Genero Excel con formule…")
    write_excel_report_with_formulas(dates, nav, paths["xlsx_formulas"])

    print("Scrivo report…")
    save_report(metrics, calendar_year_returns(dates, nav), trailing, trailing_asof, "31/12/2024", paths, success)

    print("Fatto. Vedi la cartella 'output' per CSV, SVG, XLSX, XLSX (formule) e report.md")


if __name__ == "__main__":
    main()