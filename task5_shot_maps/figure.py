"""Interactive Plotly rendering for Task 5 shot maps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from .rates import RateResult, gaussian_smooth_preserve_sum


COLORSCALE = [
    [0.00, "#053061"],
    [0.25, "#4393c3"],
    [0.50, "#f7f7f7"],
    [0.75, "#f46d43"],
    [1.00, "#a50026"],
]

SMOOTH_SIGMA_BINS = 0.6
SMOOTH_RADIUS_BINS = 2


def _rink_shapes() -> list[dict]:
    """Draw the offensive zone with generic Plotly primitives."""

    line = {"color": "#222", "width": 1.3}
    red = {"color": "#d94b64", "width": 1.2}
    blue = {"color": "#1479b8", "width": 2.3}
    shapes: list[dict] = [
        {"type": "rect", "x0": 25, "x1": 100, "y0": -42.5, "y1": 42.5,
         "line": line, "fillcolor": "rgba(255,255,255,0)"},
        {"type": "line", "x0": 25, "x1": 25, "y0": -42.5, "y1": 42.5, "line": blue},
        {"type": "line", "x0": 89, "x1": 89, "y0": -42.5, "y1": 42.5, "line": red},
        {"type": "circle", "x0": 54, "x1": 84, "y0": 7, "y1": 37, "line": red},
        {"type": "circle", "x0": 54, "x1": 84, "y0": -37, "y1": -7, "line": red},
        {"type": "circle", "x0": 68.4, "x1": 69.6, "y0": 21.4, "y1": 22.6,
         "line": red, "fillcolor": "#d94b64"},
        {"type": "circle", "x0": 68.4, "x1": 69.6, "y0": -22.6, "y1": -21.4,
         "line": red, "fillcolor": "#d94b64"},
        {"type": "path", "path": "M 89,-4 L 85,-4 Q 82,0 85,4 L 89,4 Z",
         "line": {"color": "#70b7df", "width": 1.4}, "fillcolor": "rgba(112,183,223,0.22)"},
        {"type": "rect", "x0": 89, "x1": 92, "y0": -3, "y1": 3,
         "line": {"color": "#777", "width": 1.2}, "fillcolor": "rgba(255,255,255,0)"},
    ]
    return shapes


def build_figure(
    result: RateResult,
    season_label: str,
    *,
    team_labels: dict[str, str] | None = None,
    color_limit: float | None = None,
) -> go.Figure:
    """Create a season figure with independent team and raw/smoothed controls."""

    teams = sorted(result.excess_rates)
    if not teams:
        raise ValueError("At least one team is required")
    team_labels = team_labels or {}
    all_values = np.concatenate([result.excess_rates[team].ravel() for team in teams])
    if color_limit is None:
        color_limit = float(np.max(np.abs(all_values)))
    if not np.isfinite(color_limit) or color_limit <= 0:
        color_limit = 1.0

    x_centers = (result.x_edges[:-1] + result.x_edges[1:]) / 2.0
    y_centers = (result.y_edges[:-1] + result.y_edges[1:]) / 2.0
    smoothed_league = gaussian_smooth_preserve_sum(
        result.league_rate,
        sigma_bins=SMOOTH_SIGMA_BINS,
        radius_bins=SMOOTH_RADIUS_BINS,
    )

    def layers(team: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_team = result.team_rates[team]
        smooth_team = gaussian_smooth_preserve_sum(
            raw_team,
            sigma_bins=SMOOTH_SIGMA_BINS,
            radius_bins=SMOOTH_RADIUS_BINS,
        )
        raw_excess = result.excess_rates[team]
        smooth_excess = smooth_team - smoothed_league
        raw_custom = np.stack([raw_team.T, result.league_rate.T], axis=-1)
        smooth_custom = np.stack([smooth_team.T, smoothed_league.T], axis=-1)
        return raw_excess.T, smooth_excess.T, raw_custom, smooth_custom

    first_raw, first_smooth, first_raw_custom, first_smooth_custom = layers(teams[0])
    colorbar = {"title": {"text": "Tirs excédentaires<br>par heure-équipe"}}
    hover = (
        "x=%{x:.1f} pi, y=%{y:.1f} pi"
        "<br>Excédent=%{z:+.3f} tir/h-équipe"
        "<br>Équipe=%{customdata[0]:.3f}"
        "<br>Ligue=%{customdata[1]:.3f}<extra></extra>"
    )
    figure = go.Figure(
        data=[
            go.Heatmap(
                x=x_centers, y=y_centers, z=first_raw, customdata=first_raw_custom,
                visible=True, colorscale=COLORSCALE, zmin=-color_limit,
                zmax=color_limit, zmid=0, colorbar=colorbar,
                hovertemplate=hover, name=teams[0],
            ),
            go.Heatmap(
                x=x_centers, y=y_centers, z=first_smooth,
                customdata=first_smooth_custom, visible=False,
                colorscale=COLORSCALE, zmin=-color_limit, zmax=color_limit,
                zmid=0, colorbar=colorbar, hovertemplate=hover,
                zsmooth="best", connectgaps=True,
                name=teams[0],
            ),
            go.Contour(
                x=x_centers, y=y_centers, z=first_smooth,
                visible=False, showscale=False, hoverinfo="skip",
                contours={
                    "coloring": "none",
                    "start": -color_limit,
                    "end": color_limit,
                    "size": color_limit / 4.0,
                    "showlabels": False,
                },
                line={"color": "rgba(25,25,25,0.30)", "width": 0.8},
                name="Repères de niveau",
            ),
        ]
    )

    team_buttons = []
    for team in teams:
        label = team_labels.get(team, team)
        raw, smooth, raw_custom, smooth_custom = layers(team)
        team_buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {
                        "z": [raw, smooth, smooth],
                        "customdata": [raw_custom, smooth_custom, None],
                        "name": [team, team, "Repères de niveau"],
                    },
                    {"title": {"text": f"{label} — carte de tirs {season_label}"}},
                ],
            }
        )

    view_buttons = [
        {
            "label": "Valeurs brutes 5×5 pi",
            "method": "restyle",
            "args": [{"visible": [True, False, False]}, [0, 1, 2]],
        },
        {
            "label": "Carte lissée",
            "method": "restyle",
            "args": [{"visible": [False, True, True]}, [0, 1, 2]],
        },
    ]

    first_label = team_labels.get(teams[0], teams[0])
    figure.update_layout(
        title={"text": f"{first_label} — carte de tirs {season_label}", "x": 0.5},
        template="plotly_white",
        width=930,
        height=610,
        margin={"l": 65, "r": 120, "t": 120, "b": 65},
        shapes=_rink_shapes(),
        updatemenus=[
            {
                "buttons": team_buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "y": 1.14,
                "xanchor": "left",
                "yanchor": "top",
            },
            {
                "buttons": view_buttons,
                "direction": "right",
                "showactive": True,
                "active": 0,
                "x": 0.42,
                "y": 1.14,
                "xanchor": "left",
                "yanchor": "top",
            },
        ],
        annotations=[
            {
                "text": "Équipe :",
                "x": 0,
                "y": 1.205,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "xanchor": "left",
            },
            {
                "text": "Affichage :",
                "x": 0.42,
                "y": 1.205,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "xanchor": "left",
            },
            {
                "text": "← ligne bleue | l'équipe attaque vers la droite →",
                "x": 0.5,
                "y": -0.13,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
            },
        ],
    )
    figure.update_xaxes(
        title="Coordonnée longitudinale normalisée (pieds)", range=[25, 100],
        constrain="domain", showgrid=False, zeroline=False,
    )
    figure.update_yaxes(
        title="Coordonnée transversale normalisée (pieds)", range=[-42.5, 42.5],
        scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False,
    )
    return figure


def write_figure_html(figure: go.Figure, destination: Path | str) -> Path:
    """Export a centered, responsive, standalone interactive HTML file."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    html = pio.to_html(
        figure,
        include_plotlyjs=True,
        full_html=True,
        div_id=f"task5-{destination.stem}",
        config={"responsive": True},
    )
    centered_style = (
        "<style>"
        "html,body{margin:0;padding:0;width:100%;background:#fff;}"
        ".plotly-graph-div{margin-left:auto!important;margin-right:auto!important;"
        "max-width:100%;}"
        "</style>"
    )
    html = html.replace("</head>", f"{centered_style}</head>", 1)
    destination.write_text(html, encoding="utf-8")
    return destination
