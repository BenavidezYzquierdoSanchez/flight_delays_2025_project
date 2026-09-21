from __future__ import annotations

import os
import re
import math
import glob
import time
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------
# Configuración general
# -----------------------------
st.set_page_config(
    page_title="US Flight Delays 2025 | Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_SLUG = "a7madmostafa/us-flight-delays-2025-bts-on-time-performance"
BASE_DIR = Path(os.getenv("FLIGHT_PROJECT_DIR", "/content/flight_delays_2025"))
RAW_DIR = BASE_DIR / "raw"
OPT_FILE = BASE_DIR / "flights_2025_optimized.parquet"
RAW_DIR.mkdir(parents=True, exist_ok=True)
BASE_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "navy": "#0B1F33",
    "blue": "#1769AA",
    "cyan": "#0E7490",
    "teal": "#0F766E",
    "amber": "#F59E0B",
    "red": "#D64545",
    "green": "#2E8B57",
    "slate": "#475569",
    "muted": "#64748B",
    "paper": "#F8FAFC",
    "grid": "#E2E8F0",
}

st.markdown(
    f"""
    <style>
      .stApp {{ background: {PALETTE['paper']}; }}
      .block-container {{padding-top: 1.15rem; padding-bottom: 2rem;}}
      h1, h2, h3 {{color:{PALETTE['navy']}; letter-spacing:-0.02em;}}
      [data-testid="stSidebar"] {{background:#FFFFFF; border-right:1px solid {PALETTE['grid']};}}
      .hero {{
        padding: 1.25rem 1.4rem; border-radius: 18px;
        background: linear-gradient(120deg, {PALETTE['navy']} 0%, #123A5A 64%, {PALETTE['cyan']} 100%);
        color: white; margin-bottom: .8rem;
        box-shadow: 0 10px 28px rgba(11,31,51,.12);
      }}
      .hero h1 {{color:white; margin:0; font-size:2.05rem;}}
      .hero p {{margin:.35rem 0 0 0; color:#D9E7F2; font-size:1.02rem;}}
      .small-note {{font-size:.83rem; color:{PALETTE['muted']};}}
      .metric-card {{
        background:#FFFFFF; border:1px solid {PALETTE['grid']}; border-radius:16px;
        padding:1rem 1.1rem; min-height:118px; box-shadow:0 4px 14px rgba(15,23,42,.04);
      }}
      .metric-label {{font-size:.8rem; text-transform:uppercase; letter-spacing:.07em; color:{PALETTE['muted']}; font-weight:700;}}
      .metric-value {{font-size:1.85rem; line-height:1.1; color:{PALETTE['navy']}; font-weight:800; margin-top:.4rem;}}
      .metric-sub {{font-size:.82rem; color:{PALETTE['muted']}; margin-top:.35rem;}}
      .info-box {{background:#FFFFFF; border:1px solid {PALETTE['grid']}; border-radius:14px; padding:1rem 1.1rem;}}
      .badge {{display:inline-block; padding:.2rem .55rem; border-radius:999px; font-size:.75rem; font-weight:700; background:#E6F3F7; color:{PALETTE['cyan']};}}
      div[data-testid="stPlotlyChart"] {{background:#FFFFFF; border:1px solid {PALETTE['grid']}; border-radius:16px; padding:.25rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Utilidades de preparación
# -----------------------------
ALIASES: Dict[str, List[str]] = {
    "year": ["Year"],
    "quarter": ["Quarter"],
    "month": ["Month"],
    "day_of_month": ["DayofMonth", "DayOfMonth"],
    "day_of_week": ["DayOfWeek"],
    "flight_date": ["FlightDate"],
    "carrier": ["Reporting_Airline", "Operating_Airline", "Marketing_Airline_Network", "IATA_CODE_Reporting_Airline"],
    "origin": ["Origin"],
    "origin_city": ["OriginCityName"],
    "origin_state": ["OriginState"],
    "dest": ["Dest"],
    "dest_city": ["DestCityName"],
    "dest_state": ["DestState"],
    "dep_delay": ["DepDelay"],
    "dep_delay_min": ["DepDelayMinutes"],
    "dep_del15": ["DepDel15"],
    "dep_time_blk": ["DepTimeBlk"],
    "arr_delay": ["ArrDelay"],
    "arr_delay_min": ["ArrDelayMinutes"],
    "arr_del15": ["ArrDel15"],
    "arr_time_blk": ["ArrTimeBlk"],
    "cancelled": ["Cancelled"],
    "cancel_code": ["CancellationCode"],
    "diverted": ["Diverted"],
    "distance": ["Distance"],
    "flights": ["Flights"],
    "carrier_delay": ["CarrierDelay"],
    "weather_delay": ["WeatherDelay"],
    "nas_delay": ["NASDelay"],
    "security_delay": ["SecurityDelay"],
    "late_aircraft_delay": ["LateAircraftDelay"],
}

STRING_COLS = {
    "carrier", "origin", "origin_city", "origin_state", "dest", "dest_city", "dest_state",
    "dep_time_blk", "arr_time_blk", "cancel_code"
}
INT_COLS = {"year", "quarter", "month", "day_of_month", "day_of_week"}
NUMERIC_COLS = set(ALIASES) - STRING_COLS - INT_COLS - {"flight_date"}


def ensure_kagglehub() -> None:
    try:
        import kagglehub  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])


def download_dataset() -> Path:
    """Descarga el dataset público de Kaggle usando kagglehub."""
    ensure_kagglehub()
    import kagglehub
    path = Path(kagglehub.dataset_download(DATASET_SLUG))
    return path


def source_files(path: Path) -> List[Path]:
    patterns = ["**/*.csv", "**/*.CSV", "**/*.parquet", "**/*.Parquet"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(path.glob(pat))
    # Evitar archivos derivados del mismo proyecto
    files = [p for p in files if "optimized" not in p.name.lower()]
    return sorted(set(files))


def _choose_sources(columns: Iterable[str]) -> Dict[str, str]:
    columns = set(columns)
    mapping: Dict[str, str] = {}
    for canon, aliases in ALIASES.items():
        for alias in aliases:
            if alias in columns:
                mapping[canon] = alias
                break
    return mapping


def _normalize_chunk(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    reverse = {src: canon for canon, src in mapping.items()}
    df = df.rename(columns=reverse)
    out = pd.DataFrame(index=df.index)

    for canon in ALIASES:
        if canon in df.columns:
            out[canon] = df[canon]
        else:
            out[canon] = pd.NA

    # Tipos compactos y consistentes
    for c in STRING_COLS:
        out[c] = out[c].astype("string")
    for c in INT_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Int16")
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    out["flight_date"] = pd.to_datetime(out["flight_date"], errors="coerce")
    if out["year"].isna().all() and out["flight_date"].notna().any():
        out["year"] = out["flight_date"].dt.year.astype("Int16")
    if out["month"].isna().all() and out["flight_date"].notna().any():
        out["month"] = out["flight_date"].dt.month.astype("Int16")
    if out["day_of_week"].isna().all() and out["flight_date"].notna().any():
        out["day_of_week"] = (out["flight_date"].dt.dayofweek + 1).astype("Int16")

    # Flights suele ser 1 por fila; si no existe, se define como 1.
    if out["flights"].isna().all():
        out["flights"] = np.float32(1.0)
    else:
        out["flights"] = out["flights"].fillna(1.0).astype("float32")

    return out


def prepare_optimized_parquet(dataset_path: Path, output_path: Path, chunksize: int = 250_000) -> Path:
    """Crea un parquet compacto con solo las columnas analíticas necesarias."""
    if output_path.exists() and output_path.stat().st_size > 1024:
        return output_path

    files = source_files(dataset_path)
    if not files:
        raise FileNotFoundError(f"No se encontraron CSV/Parquet dentro de {dataset_path}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([
        ("year", pa.int16()), ("quarter", pa.int16()), ("month", pa.int16()),
        ("day_of_month", pa.int16()), ("day_of_week", pa.int16()),
        ("flight_date", pa.timestamp("ns")),
        ("carrier", pa.string()), ("origin", pa.string()), ("origin_city", pa.string()), ("origin_state", pa.string()),
        ("dest", pa.string()), ("dest_city", pa.string()), ("dest_state", pa.string()),
        ("dep_delay", pa.float32()), ("dep_delay_min", pa.float32()), ("dep_del15", pa.float32()), ("dep_time_blk", pa.string()),
        ("arr_delay", pa.float32()), ("arr_delay_min", pa.float32()), ("arr_del15", pa.float32()), ("arr_time_blk", pa.string()),
        ("cancelled", pa.float32()), ("cancel_code", pa.string()), ("diverted", pa.float32()),
        ("distance", pa.float32()), ("flights", pa.float32()),
        ("carrier_delay", pa.float32()), ("weather_delay", pa.float32()), ("nas_delay", pa.float32()),
        ("security_delay", pa.float32()), ("late_aircraft_delay", pa.float32()),
    ])

    writer = pq.ParquetWriter(output_path, schema=schema, compression="zstd", use_dictionary=True)
    rows = 0
    try:
        for file in files:
            suffix = file.suffix.lower()
            if suffix == ".csv":
                header = pd.read_csv(file, nrows=0)
                mapping = _choose_sources(header.columns)
                if "flight_date" not in mapping and "year" not in mapping:
                    continue
                usecols = list(mapping.values())
                for chunk in pd.read_csv(file, usecols=usecols, chunksize=chunksize, low_memory=False):
                    normalized = _normalize_chunk(chunk, mapping)
                    table = pa.Table.from_pandas(normalized, schema=schema, preserve_index=False, safe=False)
                    writer.write_table(table)
                    rows += len(normalized)
            elif suffix == ".parquet":
                cols = pd.read_parquet(file, engine="pyarrow").columns.tolist()
                mapping = _choose_sources(cols)
                usecols = list(mapping.values())
                if not usecols:
                    continue
                pdf = pd.read_parquet(file, columns=usecols)
                for start in range(0, len(pdf), chunksize):
                    normalized = _normalize_chunk(pdf.iloc[start:start + chunksize].copy(), mapping)
                    table = pa.Table.from_pandas(normalized, schema=schema, preserve_index=False, safe=False)
                    writer.write_table(table)
                    rows += len(normalized)
    finally:
        writer.close()

    if rows == 0:
        raise RuntimeError("No fue posible preparar datos. Verifica la estructura descargada desde Kaggle.")
    return output_path


@st.cache_resource(show_spinner=False)
def get_connection(parquet_path: str):
    import duckdb
    con = duckdb.connect(database=":memory:")
    safe = parquet_path.replace("'", "''")
    con.execute(f"CREATE VIEW flights AS SELECT * FROM read_parquet('{safe}')")
    return con


def qdf(con, sql: str, params: Optional[List] = None) -> pd.DataFrame:
    return con.execute(sql, params or []).df()


def quote_list(values: List[str]) -> str:
    if not values:
        return ""
    return ",".join("'" + str(v).replace("'", "''") + "'" for v in values)


def build_where(months: List[int], carriers: List[str], states: List[str]) -> str:
    clauses = ["1=1"]
    if months:
        clauses.append(f"month IN ({','.join(str(int(x)) for x in months)})")
    if carriers:
        clauses.append(f"carrier IN ({quote_list(carriers)})")
    if states:
        clauses.append(f"origin_state IN ({quote_list(states)})")
    return " AND ".join(clauses)


def style_fig(fig, height=390, y_suffix=None):
    fig.update_layout(
        height=height,
        margin=dict(l=25, r=22, t=55, b=35),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, Arial, sans-serif", color=PALETTE["navy"]),
        legend_title_text="",
        hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zeroline=False)
    if y_suffix:
        fig.update_yaxes(ticksuffix=y_suffix)
    return fig


def wilson_interval(successes: np.ndarray, n: np.ndarray, z: float = 1.96):
    n = np.maximum(n.astype(float), 1.0)
    p = successes / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * np.sqrt((p*(1-p)/n) + (z*z/(4*n*n))) / denom
    return np.maximum(0, center-half), np.minimum(1, center+half)


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>",
        unsafe_allow_html=True,
    )

# -----------------------------
# Carga / preparación del dataset
# -----------------------------
st.markdown(
    """
    <div class='hero'>
      <h1>US Flight Delays 2025</h1>
      <p>Dashboard de puntualidad, demoras, cancelaciones y riesgo operacional · BTS On-Time Performance</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Datos y filtros")
    st.caption("Fuente principal: dataset público de Kaggle basado en BTS On-Time Performance.")
    force = st.button("↻ Reconstruir dataset optimizado", use_container_width=True)
    if force and OPT_FILE.exists():
        OPT_FILE.unlink()
        st.cache_resource.clear()
        st.rerun()

try:
    if not OPT_FILE.exists():
        with st.status("Preparando datos por primera vez…", expanded=True) as status:
            st.write("Descargando el dataset desde Kaggle…")
            ds_path = download_dataset()
            st.write(f"Dataset localizado en: `{ds_path}`")
            st.write("Creando parquet optimizado para análisis interactivo…")
            prepare_optimized_parquet(ds_path, OPT_FILE)
            status.update(label="Datos listos", state="complete", expanded=False)
    con = get_connection(str(OPT_FILE))
except Exception as e:
    st.error("No se pudo preparar el dataset automáticamente.")
    st.exception(e)
    st.info("En Google Colab ejecuta el notebook entregado; instalará las dependencias y descargará Kaggle automáticamente.")
    st.stop()

# -----------------------------
# Filtros
# -----------------------------
months_available = qdf(con, "SELECT DISTINCT month FROM flights WHERE month IS NOT NULL ORDER BY month")['month'].dropna().astype(int).tolist()
carriers_available = qdf(con, "SELECT DISTINCT carrier FROM flights WHERE carrier IS NOT NULL ORDER BY carrier")['carrier'].dropna().astype(str).tolist()
states_available = qdf(con, "SELECT DISTINCT origin_state FROM flights WHERE origin_state IS NOT NULL ORDER BY origin_state")['origin_state'].dropna().astype(str).tolist()

with st.sidebar:
    months = st.multiselect("Mes", months_available, default=months_available)
    carriers = st.multiselect("Aerolínea (código)", carriers_available, default=[])
    states = st.multiselect("Estado de origen", states_available, default=[])
    min_airport_flights = st.slider("Mín. vuelos por aeropuerto", 100, 20000, 3000, step=100)
    st.divider()
    st.markdown("<span class='badge'>4 vistas analíticas</span>", unsafe_allow_html=True)
    st.caption("Los filtros afectan KPI y visualizaciones. El mapa resume por estado de origen.")

where = build_where(months, carriers, states)

# -----------------------------
# Métricas
# -----------------------------
metrics = qdf(con, f"""
SELECT
    CAST(SUM(COALESCE(flights,1)) AS BIGINT) AS total_flights,
    AVG(CASE WHEN arr_del15 IS NOT NULL THEN arr_del15 END) AS delayed_rate,
    AVG(CASE WHEN cancelled IS NOT NULL THEN cancelled END) AS cancel_rate,
    AVG(CASE WHEN diverted IS NOT NULL THEN diverted END) AS divert_rate,
    AVG(CASE WHEN arr_del15=1 THEN arr_delay_min END) AS avg_delay_delayed,
    QUANTILE_CONT(CASE WHEN arr_delay_min IS NOT NULL THEN arr_delay_min END, 0.95) AS p95_arr_delay,
    AVG(CASE WHEN dep_del15 IS NOT NULL THEN dep_del15 END) AS dep_delayed_rate
FROM flights WHERE {where}
""").iloc[0]

total_flights = int(metrics.total_flights or 0)
delayed_rate = float(metrics.delayed_rate) if pd.notna(metrics.delayed_rate) else np.nan
on_time_rate = 1 - delayed_rate if pd.notna(delayed_rate) else np.nan
cancel_rate = float(metrics.cancel_rate) if pd.notna(metrics.cancel_rate) else np.nan
divert_rate = float(metrics.divert_rate) if pd.notna(metrics.divert_rate) else np.nan
avg_delay = float(metrics.avg_delay_delayed) if pd.notna(metrics.avg_delay_delayed) else np.nan
p95_delay = float(metrics.p95_arr_delay) if pd.notna(metrics.p95_arr_delay) else np.nan

# -----------------------------
# Tabs: 4 vistas
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "01 · Resumen ejecutivo",
    "02 · Puntualidad y tiempo",
    "03 · Aerolíneas y aeropuertos",
    "04 · Geografía y causas",
])

with tab1:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: metric_card("Vuelos", f"{total_flights:,.0f}", "operaciones filtradas")
    with c2: metric_card("Puntualidad", f"{on_time_rate*100:.1f}%" if pd.notna(on_time_rate) else "—", "llegada < 15 min tarde")
    with c3: metric_card("Con demora", f"{delayed_rate*100:.1f}%" if pd.notna(delayed_rate) else "—", "ArrDel15 = 1")
    with c4: metric_card("Cancelados", f"{cancel_rate*100:.2f}%" if pd.notna(cancel_rate) else "—", "sobre vuelos programados")
    with c5: metric_card("Desviados", f"{divert_rate*100:.2f}%" if pd.notna(divert_rate) else "—", "operaciones desviadas")
    with c6: metric_card("Demora P95", f"{p95_delay:.0f} min" if pd.notna(p95_delay) else "—", "95% de vuelos debajo")

    st.markdown("### Evolución mensual")
    monthly = qdf(con, f"""
    SELECT month,
           SUM(COALESCE(flights,1)) AS flights,
           100*(1-AVG(arr_del15)) AS on_time_pct,
           100*AVG(cancelled) AS cancel_pct,
           AVG(CASE WHEN arr_del15=1 THEN arr_delay_min END) AS avg_delay_delayed
    FROM flights
    WHERE {where} AND month IS NOT NULL
    GROUP BY month ORDER BY month
    """)
    if not monthly.empty:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly.month, y=monthly.flights, name="Vuelos", yaxis="y2", opacity=.25, marker_color=PALETTE['blue']))
        fig.add_trace(go.Scatter(x=monthly.month, y=monthly.on_time_pct, name="Puntualidad %", mode="lines+markers", line=dict(width=3, color=PALETTE['teal'])))
        fig.add_trace(go.Scatter(x=monthly.month, y=monthly.cancel_pct, name="Cancelación %", mode="lines+markers", line=dict(width=2, color=PALETTE['red'])))
        fig.update_layout(
            yaxis=dict(title="Porcentaje", ticksuffix="%", gridcolor=PALETTE['grid']),
            yaxis2=dict(title="Vuelos", overlaying="y", side="right", showgrid=False),
            xaxis=dict(title="Mes", dtick=1),
            title="Volumen, puntualidad y cancelación por mes",
        )
        style_fig(fig, 430)
        st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns([1.1, .9])
    with left:
        st.markdown("### Lectura ejecutiva")
        st.markdown(
            f"""
            <div class='info-box'>
            <b>Objetivo:</b> detectar cuándo y dónde se concentra el riesgo de demora para apoyar decisiones de planificación operacional.<br><br>
            <b>Estado filtrado:</b> {total_flights:,.0f} vuelos; puntualidad de {on_time_rate*100:.1f}% y cancelación de {cancel_rate*100:.2f}%.
            La demora promedio entre vuelos retrasados es de {avg_delay:.1f} min y el percentil 95 alcanza {p95_delay:.0f} min.<br><br>
            <b>Uso recomendado:</b> combinar tendencia mensual, aerolínea, aeropuerto, estado y causa antes de tomar acciones de capacidad, horarios o contingencia.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("### KPI del proyecto")
        kpis = pd.DataFrame({
            "KPI": ["Vuelos", "Puntualidad", "% demora", "% cancelación", "% desviación", "Demora P95"],
            "Fórmula resumida": ["Σ Flights", "1 − AVG(ArrDel15)", "AVG(ArrDel15)", "AVG(Cancelled)", "AVG(Diverted)", "Q95(ArrDelayMinutes)"],
        })
        st.dataframe(kpis, use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Patrón temporal y riesgo por franja horaria")
    colA, colB = st.columns(2)
    with colA:
        daily = qdf(con, f"""
        SELECT day_of_week,
               100*AVG(arr_del15) AS delayed_pct,
               AVG(CASE WHEN arr_del15=1 THEN arr_delay_min END) AS avg_delay
        FROM flights WHERE {where} AND day_of_week IS NOT NULL
        GROUP BY day_of_week ORDER BY day_of_week
        """)
        dow_map = {1:"Lun",2:"Mar",3:"Mié",4:"Jue",5:"Vie",6:"Sáb",7:"Dom"}
        daily["día"] = daily.day_of_week.map(dow_map)
        fig = px.bar(daily, x="día", y="delayed_pct", title="% de llegadas con demora por día", labels={"delayed_pct":"Demora (%)","día":"Día"})
        fig.update_traces(marker_color=PALETTE['blue'])
        style_fig(fig, 390, "%")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        blocks = qdf(con, f"""
        SELECT dep_time_blk,
               100*AVG(arr_del15) AS delayed_pct,
               COUNT(*) AS n
        FROM flights WHERE {where} AND dep_time_blk IS NOT NULL
        GROUP BY dep_time_blk ORDER BY dep_time_blk
        """)
        fig = px.line(blocks, x="dep_time_blk", y="delayed_pct", markers=True, title="Riesgo de demora según hora programada de salida", labels={"dep_time_blk":"Franja salida","delayed_pct":"Demora (%)"})
        fig.update_traces(line_color=PALETTE['cyan'], marker=dict(size=7))
        style_fig(fig, 390, "%")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Relación distancia – demora")
    dist = qdf(con, f"""
    SELECT CAST(FLOOR(distance/250)*250 AS INTEGER) AS dist_bin,
           COUNT(*) AS n,
           100*AVG(arr_del15) AS delayed_pct,
           AVG(CASE WHEN arr_del15=1 THEN arr_delay_min END) AS avg_delay
    FROM flights
    WHERE {where} AND distance IS NOT NULL
    GROUP BY 1 HAVING COUNT(*) >= 500
    ORDER BY 1
    """)
    fig = px.scatter(dist, x="dist_bin", y="delayed_pct", size="n", hover_data=["avg_delay"], title="Demora por bandas de distancia (250 millas)", labels={"dist_bin":"Distancia inicial de banda (millas)","delayed_pct":"Vuelos con demora (%)","n":"Vuelos"})
    fig.update_traces(marker=dict(color=PALETTE['teal'], opacity=.72, line=dict(width=0)))
    style_fig(fig, 420, "%")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown("### Aerolíneas: tasa de demora con incertidumbre")
    airline = qdf(con, f"""
    SELECT carrier,
           COUNT(*) AS n,
           SUM(CASE WHEN arr_del15=1 THEN 1 ELSE 0 END) AS delayed,
           100*AVG(arr_del15) AS delayed_pct,
           AVG(CASE WHEN arr_del15=1 THEN arr_delay_min END) AS avg_delay
    FROM flights
    WHERE {where} AND carrier IS NOT NULL AND arr_del15 IS NOT NULL
    GROUP BY carrier HAVING COUNT(*) >= 500
    ORDER BY delayed_pct DESC
    """)
    if not airline.empty:
        lo, hi = wilson_interval(airline.delayed.to_numpy(), airline.n.to_numpy())
        airline["low"] = lo*100
        airline["high"] = hi*100
        airline["err_plus"] = airline.high-airline.delayed_pct
        airline["err_minus"] = airline.delayed_pct-airline.low
        fig = go.Figure(go.Bar(
            x=airline.delayed_pct, y=airline.carrier, orientation="h",
            error_x=dict(type="data", array=airline.err_plus, arrayminus=airline.err_minus, visible=True),
            marker_color=PALETTE['blue'],
            customdata=np.c_[airline.n, airline.avg_delay],
            hovertemplate="%{y}<br>Demora: %{x:.1f}%<br>Vuelos: %{customdata[0]:,.0f}<br>Demora media: %{customdata[1]:.1f} min<extra></extra>"
        ))
        fig.update_layout(title="% de demora por aerolínea · IC 95% Wilson", xaxis_title="Vuelos con demora (%)", yaxis_title="")
        style_fig(fig, max(360, 42*len(airline)), "%")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("La barra de error representa un intervalo de confianza del 95% para la proporción de vuelos con ArrDel15 = 1.")

    st.markdown("### Aeropuertos de origen: volumen vs. riesgo")
    airports = qdf(con, f"""
    SELECT origin,
           MAX(origin_city) AS city,
           MAX(origin_state) AS state,
           COUNT(*) AS flights,
           100*AVG(arr_del15) AS delayed_pct,
           100*AVG(cancelled) AS cancel_pct,
           AVG(CASE WHEN arr_del15=1 THEN arr_delay_min END) AS avg_delay
    FROM flights
    WHERE {where} AND origin IS NOT NULL
    GROUP BY origin
    HAVING COUNT(*) >= {int(min_airport_flights)}
    ORDER BY delayed_pct DESC
    """)
    fig = px.scatter(
        airports, x="flights", y="delayed_pct", size="avg_delay", color="cancel_pct",
        hover_name="origin", hover_data=["city","state","avg_delay"],
        title=f"Aeropuertos con ≥ {min_airport_flights:,} vuelos",
        labels={"flights":"Vuelos","delayed_pct":"Demora (%)","cancel_pct":"Cancelación (%)","avg_delay":"Demora media"},
        color_continuous_scale="Blues",
    )
    style_fig(fig, 480, "%")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Distribución geográfica")
    geo = qdf(con, f"""
    SELECT origin_state AS state,
           COUNT(*) AS flights,
           100*AVG(arr_del15) AS delayed_pct,
           100*AVG(cancelled) AS cancel_pct
    FROM flights
    WHERE {where} AND origin_state IS NOT NULL
    GROUP BY origin_state
    """)
    fig = px.choropleth(
        geo, locations="state", locationmode="USA-states", scope="usa",
        color="delayed_pct", hover_data=["flights","cancel_pct"],
        color_continuous_scale="Blues",
        title="% de vuelos con demora por estado de origen",
        labels={"delayed_pct":"Demora (%)","flights":"Vuelos","cancel_pct":"Cancelación (%)"},
    )
    fig.update_layout(height=510, margin=dict(l=0,r=0,t=55,b=0), paper_bgcolor="#FFFFFF", geo_bgcolor="#FFFFFF")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Causas acumuladas de minutos de demora")
    causes = qdf(con, f"""
    SELECT
      SUM(COALESCE(carrier_delay,0)) AS carrier,
      SUM(COALESCE(weather_delay,0)) AS weather,
      SUM(COALESCE(nas_delay,0)) AS nas,
      SUM(COALESCE(security_delay,0)) AS security,
      SUM(COALESCE(late_aircraft_delay,0)) AS late_aircraft
    FROM flights WHERE {where}
    """).iloc[0]
    cause_df = pd.DataFrame({
        "Causa": ["Aerolínea","Clima","Sistema NAS","Seguridad","Aeronave previa tardía"],
        "Minutos": [causes.carrier, causes.weather, causes.nas, causes.security, causes.late_aircraft]
    })
    cause_df["Minutos"] = pd.to_numeric(cause_df["Minutos"], errors="coerce").fillna(0)
    cause_df = cause_df.sort_values("Minutos", ascending=True)
    fig = px.bar(cause_df, x="Minutos", y="Causa", orientation="h", title="Minutos de demora reportados por causa")
    fig.update_traces(marker_color=PALETTE['amber'])
    style_fig(fig, 390)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Decisiones apoyadas")
    st.markdown(
        """
        <div class='info-box'>
        <b>1.</b> Priorizar meses y franjas horarias con mayor exposición a retrasos.<br>
        <b>2.</b> Comparar aerolíneas considerando volumen e incertidumbre, no solo porcentajes.<br>
        <b>3.</b> Detectar aeropuertos/estados donde se combinan demora y cancelación.<br>
        <b>4.</b> Distinguir si la presión proviene de aerolínea, clima, NAS, seguridad o llegada tardía del avión.<br>
        <b>5.</b> Orientar planes operativos, buffers de conexión, asignación de recursos y contingencias.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()
st.caption("Definición BTS: se considera retrasado un vuelo con llegada ≥ 15 minutos después del horario programado. Proyecto académico · Visualización de Datos.")
