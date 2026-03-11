from collections import OrderedDict
import json

import matplotlib

# Evita backend interactivo (Tk) que puede fallar en renderizaciones web/redimensionados.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder

from back.config import settings

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
)


def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if not isinstance(df.index, pd.DatetimeIndex):
                if {"anio", "mes"}.issubset(df.columns):
                        if "dia" in df.columns:
                                idx = pd.to_datetime(
                                        dict(year=df["anio"], month=df["mes"], day=df["dia"]),
                                        errors="coerce",
                                )
                        else:
                                idx = pd.to_datetime(
                                        dict(year=df["anio"], month=df["mes"], day=1),
                                        errors="coerce",
                                )
                        df = df.set_index(idx)
                elif "__dt" in df.columns:
                        df = df.set_index(pd.to_datetime(df["__dt"], errors="coerce"))
                else:
                        raise ValueError("df debe tener DatetimeIndex o columnas anio/mes(/dia).")

        return df[~pd.isna(df.index)].sort_index()


def compute_time_axis_bounds(index: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
        if len(index) == 0:
                now = pd.Timestamp.now().normalize()
                return now, now

        dt_index = pd.to_datetime(index, errors="coerce")
        dt_index = dt_index[~pd.isna(dt_index)]
        if len(dt_index) == 0:
                now = pd.Timestamp.now().normalize()
                return now, now

        if len(dt_index) == 1:
                return dt_index[0], dt_index[0]

        dt_min, dt_max = dt_index.min(), dt_index.max()
        step = pd.Series(dt_index).diff().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
                fallback_days = int(settings.get("plots.predictions.fallback_step_days", 30))
                step = pd.Timedelta(days=fallback_days)
        padding_steps = int(settings.get("plots.predictions.x_axis_padding_steps", 2))
        pad = step * padding_steps
        return dt_min - pad, dt_max + pad


def build_interactive_plot_html(
    fig: go.Figure,
    element_id: str,
    click_input_id: str | None = None,
) -> str:
    fig_json = json.dumps(fig.to_plotly_json(), cls=PlotlyJSONEncoder)
    click_value = json.dumps(click_input_id) if click_input_id else "null"

    return f"""
<div id=\"{element_id}\" style=\"width:100%; min-height:420px; background:#fff;\"></div>
<script>
(function() {{
    const spec = {fig_json};
    const domId = {json.dumps(element_id)};
    const clickInputId = {click_value};

    function pointPayload(pt) {{
        const data = Array.isArray(pt.customdata) ? pt.customdata : [];
        return {{
            trace_name: pt.data && pt.data.name ? pt.data.name : null,
            x: pt.x ?? null,
            y: pt.y ?? null,
            date_label: data[0] ?? null,
            real: data[1] ?? null,
            scenario: data[2] ?? null,
            diff: data[3] ?? null,
            diff_pct: data[4] ?? null,
            segment: data[5] ?? null,
        }};
    }}

    function applyCrosshair(gd, x, y) {{
        Plotly.relayout(gd, {{
            shapes: [
                {{
                    type: 'line',
                    x0: x,
                    x1: x,
                    y0: 0,
                    y1: 1,
                    xref: 'x',
                    yref: 'paper',
                    line: {{ color: '#94a3b8', width: 1, dash: 'dot' }}
                }},
                {{
                    type: 'line',
                    x0: 0,
                    x1: 1,
                    y0: y,
                    y1: y,
                    xref: 'paper',
                    yref: 'y',
                    line: {{ color: '#94a3b8', width: 1, dash: 'dot' }}
                }}
            ]
        }});
    }}

        function syncFullscreenStyles(gd) {{
            const isFullscreen = document.fullscreenElement === gd;
            if (isFullscreen) {{
                gd.style.width = '100vw';
                gd.style.height = '100vh';
                gd.style.maxWidth = '100vw';
                gd.style.maxHeight = '100vh';
                gd.style.background = '#ffffff';
                gd.style.padding = '12px';
                gd.style.boxSizing = 'border-box';
            }} else {{
                gd.style.width = '100%';
                gd.style.height = '420px';
                gd.style.maxWidth = '';
                gd.style.maxHeight = '';
                gd.style.background = '#ffffff';
                gd.style.padding = '';
                gd.style.boxSizing = '';
            }}
            if (window.Plotly && window.Plotly.Plots) {{
                window.Plotly.Plots.resize(gd);
            }}
        }}

    function initPlotly() {{
        const gd = document.getElementById(domId);
        if (!gd) return;

        const fullscreenButton = {{
            name: 'Pantalla completa',
            title: 'Pantalla completa',
            icon: {{
                width: 1000,
                height: 1000,
                path: 'M128 128h256v96H224v160H128V128zm488 0h256v256h-96V224H616v-96zM128 616h96v160h160v96H128V616zm648 0h96v256H616v-96h160V616z'
            }},
            click: function() {{
                if (!document.fullscreenElement) {{
                    if (gd.requestFullscreen) {{
                        gd.requestFullscreen();
                    }}
                }} else if (document.exitFullscreen) {{
                    document.exitFullscreen();
                }}
            }}
        }};

        Plotly.newPlot(gd, spec.data, spec.layout, {{
            responsive: true,
            locale: 'es',
            displaylogo: false,
            modeBarButtonsToRemove: ['lasso2d', 'select2d'],
            modeBarButtonsToAdd: [fullscreenButton]
        }});

        gd.style.height = '420px';
        gd.style.background = '#ffffff';
        document.addEventListener('fullscreenchange', function() {{
            syncFullscreenStyles(gd);
        }});
        setTimeout(function() {{
            syncFullscreenStyles(gd);
        }}, 0);

        gd.on('plotly_click', function(evt) {{
            if (!evt || !evt.points || !evt.points.length) return;
            const pt = evt.points[0];
            applyCrosshair(gd, pt.x, pt.y);
            if (window.Shiny && clickInputId) {{
                Shiny.setInputValue(clickInputId, pointPayload(pt), {{ priority: 'event' }});
            }}
        }});

        gd.on('plotly_doubleclick', function() {{
            Plotly.relayout(gd, {{ shapes: [] }});
            if (window.Shiny && clickInputId) {{
                Shiny.setInputValue(clickInputId, null, {{ priority: 'event' }});
            }}
        }});
    }}

    function ensureSpanishLocale(callback) {{
        if (window.Plotly && window.PlotlyLocales && window.PlotlyLocales.es) {{
            callback();
            return;
        }}

        document.addEventListener('forecastingpcd:plotly-locale-es-ready', callback, {{ once: true }});

        if (window.__forecastingpcdPlotlyLocaleEsLoading) {{
            return;
        }}

        window.__forecastingpcdPlotlyLocaleEsLoading = true;
        const localeScript = document.createElement('script');
        localeScript.src = 'https://cdn.plot.ly/plotly-locale-es-latest.js';
        localeScript.onload = function() {{
            window.__forecastingpcdPlotlyLocaleEsLoading = false;
            document.dispatchEvent(new Event('forecastingpcd:plotly-locale-es-ready'));
        }};
        document.head.appendChild(localeScript);
    }}

    if (window.Plotly) {{
        ensureSpanishLocale(initPlotly);
        return;
    }}

    document.addEventListener('forecastingpcd:plotly-ready', function() {{
        ensureSpanishLocale(initPlotly);
    }}, {{ once: true }});

    if (!window.__forecastingpcdPlotlyLoading) {{
        window.__forecastingpcdPlotlyLoading = true;
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-2.35.2.min.js';
        script.onload = function() {{
            window.__forecastingpcdPlotlyLoading = false;
            document.dispatchEvent(new Event('forecastingpcd:plotly-ready'));
        }};
        document.head.appendChild(script);
    }}
}})();
</script>
"""


def plot_predictions(
    df,
    pred,
    title,
    ylabel,
    xlabel,
    column_y,
    periodos_a_predecir=2,
    holidays_col=None,
):
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if {"anio", "mes"}.issubset(df.columns):
            if "dia" in df.columns:
                idx = pd.to_datetime(
                    dict(year=df["anio"], month=df["mes"], day=df["dia"])
                )
            else:
                idx = pd.to_datetime(dict(year=df["anio"], month=df["mes"], day=1))
            df = df.set_index(idx).sort_index()
        else:
            raise ValueError("df debe tener DatetimeIndex o columnas anio/mes(/dia).")

    periodos_a_predecir = int(periodos_a_predecir or 1)
    periodos_a_predecir = max(1, min(periodos_a_predecir, len(df) - 1))
    train = df.iloc[:-periodos_a_predecir]

    if not isinstance(pred.index, pd.DatetimeIndex):
        pred = pd.Series(
            pred.values, index=df.index[-len(pred) :], name=getattr(pred, "name", None)
        )

    figsize = tuple(settings.get("plots.predictions.figsize", [12, 5]))
    scatter_size = int(settings.get("plots.predictions.scatter_size_single_point", 30))
    scatter_color = settings.get("plots.predictions.prediction_scatter_color", "red")
    prediction_marker = settings.get("plots.predictions.prediction_marker", "o")
    prediction_marker_size = int(
        settings.get("plots.predictions.prediction_marker_size", 6)
    )

    fig, ax = plt.subplots(figsize=figsize)
    train[column_y].plot(ax=ax, label="Real")

    if len(pred) == 1:
        ax.scatter(
            pred.index,
            pred.values,
            label="Predicción",
            zorder=5,
            s=scatter_size,
            color=scatter_color,
        )
    else:
        pred.plot(
            ax=ax,
            label="Predicción",
            color=scatter_color,
            marker=prediction_marker,
            markersize=prediction_marker_size,
        )

    ax.relim()
    ax.autoscale_view()

    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 1:
        dt_min, dt_max = df.index.min(), df.index.max()
        step = df.index.to_series().diff().median()
        if pd.isna(step) or step <= pd.Timedelta(0):
            fallback_days = int(
                settings.get("plots.predictions.fallback_step_days", 30)
            )
            step = pd.Timedelta(days=fallback_days)
        padding_steps = int(settings.get("plots.predictions.x_axis_padding_steps", 2))
        pad = step * padding_steps
        ax.set_xlim(dt_min - pad, dt_max + pad)
    else:
        ax.set_xlim(df.index.min(), df.index.max())

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title)

    if holidays_col is not None and holidays_col in df.columns:
        idx_common = df.index.intersection(pred.index)
        holidays_pred = df.loc[idx_common]
        holidays_pred = holidays_pred[holidays_pred[holidays_col] == 1]
        for x in holidays_pred.index:
            ax.axvline(x=x, color="k", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    uniq = OrderedDict()
    for h, l in zip(handles, labels):
        if l and l != "_nolegend_" and l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="best")

    fig.tight_layout(pad=1.2)
    fig.subplots_adjust(left=0.12)
    return fig
