"""
Beam Deflection Calculator
Run with: streamlit run beam_calculator.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Beam Deflection Calculator",
    page_icon="🏗️",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0d0f14; }
    h1 { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -1px; }
    .stMetric label { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #888 !important; }
    .stMetric [data-testid="metric-container"] { 
        background: #13151c; 
        border: 1px solid #2a2d3a; 
        border-radius: 6px; 
        padding: 0.75rem 1rem;
    }
    .sidebar-section {
        background: #13151c;
        border-left: 3px solid #4ecca3;
        padding: 0.5rem 0.75rem;
        margin: 0.5rem 0;
        border-radius: 0 4px 4px 0;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #4ecca3;
    }
    .stSelectbox label, .stNumberInput label { 
        font-family: 'IBM Plex Mono', monospace; 
        font-size: 0.75rem;
        color: #aaa !important;
    }
    [data-testid="stSidebar"] { background: #0d0f14 !important; border-right: 1px solid #2a2d3a; }
    .stButton>button {
        background: #4ecca3 !important;
        color: #0d0f14 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        border: none;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# ── Solver ───────────────────────────────────────────────────────────────────

def solve_beam(L, E, I, support, point_loads_t, dist_loads_t):
    """
    Numerically solves beam using method of integration.

    Sign conventions (internal):
      - Loads:      positive = downward
      - Shear V:    positive = upward force on left face
      - Moment M:   positive = sagging (tension at bottom)
      - Deflection: y_up positive upward; y_down_mm positive downward (for display)

    Returns:
      x         [m]   position array
      V         [kN]  shear force
      M         [kNm] bending moment
      y_mm      [mm]  deflection (positive = downward)
      reactions dict of named reaction values (kN or kNm)
    """
    N = 6000
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    EI = E * I

    # Build distributed load array q(x)  [N/m, positive = downward]
    q = np.zeros(N)
    for (x1, x2, w) in dist_loads_t:
        q[(x >= x1) & (x <= x2)] += w

    # Resultants
    total_q  = np.trapezoid(q, x)
    moment_q = np.trapezoid(q * x, x)  # moment about A
    total_P  = sum(P for (_, P) in point_loads_t)
    moment_P = sum(P * xp for (xp, P) in point_loads_t)
    total_load = total_q + total_P
    total_mom  = moment_q + moment_P

    # ─ Simply Supported ─────────────────────────────────────────────────────
    if support == "Simply Supported":
        if L == 0:
            raise ValueError("Beam length cannot be zero.")
        Rb = total_mom / L
        Ra = total_load - Rb
        reactions = {
            "Ra ↑  (kN)": Ra / 1e3,
            "Rb ↑  (kN)": Rb / 1e3,
        }

        # Shear
        V = np.zeros(N)
        V[0] = Ra
        for i in range(1, N):
            V[i] = V[i - 1] - q[i - 1] * dx
            for (xp, P) in point_loads_t:
                if x[i - 1] < xp <= x[i]:
                    V[i] -= P

        # Moment
        M = np.zeros(N)
        for i in range(1, N):
            M[i] = M[i - 1] + V[i - 1] * dx

        # Double integrate M:  EI·y'' = M,  y(0)=0, y(L)=0
        int1 = np.zeros(N)
        for i in range(1, N):
            int1[i] = int1[i - 1] + M[i - 1] * dx
        int2 = np.zeros(N)
        for i in range(1, N):
            int2[i] = int2[i - 1] + int1[i - 1] * dx

        C1 = -int2[-1] / L   # enforces y(L) = 0
        C2 = 0.0              # enforces y(0) = 0
        y_up = (int2 + C1 * x + C2) / EI  # m, positive upward

    # ─ Cantilever (Fixed at x=0) ─────────────────────────────────────────────
    else:
        Ra = total_load
        Ma = total_mom   # reaction moment at fixed end (counterclockwise)
        reactions = {
            "Ra ↑  (kN)":  Ra / 1e3,
            "Ma  (kNm)":   Ma / 1e3,
        }

        # Shear
        V = np.zeros(N)
        V[0] = Ra
        for i in range(1, N):
            V[i] = V[i - 1] - q[i - 1] * dx
            for (xp, P) in point_loads_t:
                if x[i - 1] < xp <= x[i]:
                    V[i] -= P

        # Moment: M(0) = -Ma  (hogging at root)
        M = np.zeros(N)
        M[0] = -Ma
        for i in range(1, N):
            M[i] = M[i - 1] + V[i - 1] * dx

        # Double integrate M:  EI·y'' = M,  y(0)=0, y'(0)=0
        int1 = np.zeros(N)
        for i in range(1, N):
            int1[i] = int1[i - 1] + M[i - 1] * dx
        int2 = np.zeros(N)
        for i in range(1, N):
            int2[i] = int2[i - 1] + int1[i - 1] * dx
        # C1=0 (from y'(0)=0), C2=0 (from y(0)=0)
        y_up = int2 / EI  # m, positive upward

    # Positive downward for display
    y_mm = -y_up * 1000  # mm

    return x, V / 1e3, M / 1e3, y_mm, reactions


# ── Plotting ─────────────────────────────────────────────────────────────────

BG   = "#0d0f14"
CARD = "#13151c"
GRID = "#1e2130"
ACC  = "#4ecca3"
RED  = "#ff6b6b"
ORG  = "#ffd166"
BLU  = "#74b9ff"

def style_ax(ax, ylabel, color):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors="#888", labelsize=8)
    ax.yaxis.label.set_color(color)
    ax.yaxis.label.set_family("monospace")
    ax.xaxis.label.set_color("#888")
    ax.xaxis.label.set_family("monospace")
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel("x  (m)", fontsize=8)
    ax.grid(axis='y', color=GRID, lw=0.5, alpha=0.8)
    ax.axhline(0, color="#444", lw=0.8)


def make_figure(x, V, M, y_mm, L, point_loads, dist_loads, support):
    fig = plt.figure(figsize=(13, 11), facecolor=BG)
    gs  = GridSpec(4, 1, figure=fig, hspace=0.55,
                   top=0.96, bottom=0.06, left=0.09, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    # ── 0: Beam Diagram ──────────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.set_xlim(-0.08 * L, 1.08 * L)
    ax.set_ylim(-0.3, 2.0)
    ax.set_yticks([])
    ax.set_xlabel("x  (m)", fontsize=8, color="#888")
    ax.set_title("Beam Diagram", color="#ccc", fontsize=10,
                 fontfamily="monospace", pad=6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.tick_params(colors="#888", labelsize=8)

    # Beam body
    beam = patches.FancyBboxPatch(
        (0, 0.42), L, 0.16,
        boxstyle="round,pad=0.01",
        linewidth=1, edgecolor="#555", facecolor="#2a3050"
    )
    ax.add_patch(beam)

    # Supports
    if support == "Simply Supported":
        for xs in [0.0, L]:
            tri = plt.Polygon(
                [[xs - 0.06*L, 0.3], [xs + 0.06*L, 0.3], [xs, 0.42]],
                closed=True, facecolor="#4a5070", edgecolor=ACC, lw=1.2, zorder=4
            )
            ax.add_patch(tri)
            ax.plot([xs - 0.08*L, xs + 0.08*L], [0.28, 0.28],
                    color=ACC, lw=1.5)
        ax.text(0,  0.18, "A", ha='center', va='top', color=ACC,
                fontsize=8, fontfamily="monospace")
        ax.text(L,  0.18, "B", ha='center', va='top', color=ACC,
                fontsize=8, fontfamily="monospace")
    else:  # Cantilever
        wall = patches.Rectangle(
            (-0.12*L, 0.15), 0.10*L, 0.7,
            linewidth=1, edgecolor="#555", facecolor="#2a3050"
        )
        ax.add_patch(wall)
        # hatch lines
        for hy in np.linspace(0.15, 0.85, 8):
            ax.plot([-0.12*L, 0], [hy, hy + 0.07], color="#555", lw=0.8)
        ax.plot([0, 0], [0.15, 0.85], color=ACC, lw=2.5)
        ax.text(L + 0.03*L, 0.50, "Free end",
                ha='left', va='center', color="#777",
                fontsize=7, fontfamily="monospace")

    # Point loads
    for (xp, P) in point_loads:
        ax.annotate(
            "", xy=(xp, 0.60), xytext=(xp, 1.35),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8,
                            mutation_scale=12)
        )
        ax.text(xp, 1.42, f"{P/1e3:.1f} kN",
                ha='center', va='bottom', color=RED,
                fontsize=8, fontfamily="monospace")

    # Distributed loads
    for (x1, x2, w) in dist_loads:
        xs_arr = np.linspace(x1, x2, max(int((x2-x1)/L*20)+2, 5))
        for xi in xs_arr:
            ax.annotate(
                "", xy=(xi, 0.60), xytext=(xi, 1.10),
                arrowprops=dict(arrowstyle="-|>", color=ORG, lw=1.2,
                                mutation_scale=8, alpha=0.85)
            )
        ax.plot([x1, x2], [1.12, 1.12], color=ORG, lw=2)
        ax.text((x1+x2)/2, 1.20, f"{w/1e3:.1f} kN/m",
                ha='center', va='bottom', color=ORG,
                fontsize=8, fontfamily="monospace")

    # Dim line
    ax.annotate("", xy=(L, 0.0), xytext=(0, 0.0),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1))
    ax.text(L/2, -0.08, f"L = {L:.1f} m",
            ha='center', va='top', color="#555",
            fontsize=8, fontfamily="monospace")

    # ── 1: Shear Force ───────────────────────────────────────────────────────
    ax = axes[1]
    style_ax(ax, "V  (kN)", BLU)
    ax.plot(x, V, color=BLU, lw=1.8, zorder=3)
    ax.fill_between(x, V, 0, where=(V >= 0), alpha=0.20, color=BLU)
    ax.fill_between(x, V, 0, where=(V <  0), alpha=0.20, color=RED)
    ax.set_xlim(0, L)
    idx = np.argmax(np.abs(V))
    ax.plot(x[idx], V[idx], 'o', color=BLU, ms=5, zorder=4)
    ax.annotate(f" {V[idx]:.2f}", xy=(x[idx], V[idx]),
                color=BLU, fontsize=8, fontfamily="monospace",
                va='bottom' if V[idx] >= 0 else 'top')

    # ── 2: Bending Moment ────────────────────────────────────────────────────
    ax = axes[2]
    style_ax(ax, "M  (kNm)", ORG)
    ax.plot(x, M, color=ORG, lw=1.8, zorder=3)
    ax.fill_between(x, M, 0, where=(M >= 0), alpha=0.20, color=ORG)
    ax.fill_between(x, M, 0, where=(M <  0), alpha=0.20, color=RED)
    ax.set_xlim(0, L)
    idx = np.argmax(np.abs(M))
    ax.plot(x[idx], M[idx], 'o', color=ORG, ms=5, zorder=4)
    ax.annotate(f" {M[idx]:.2f}", xy=(x[idx], M[idx]),
                color=ORG, fontsize=8, fontfamily="monospace",
                va='bottom' if M[idx] >= 0 else 'top')

    # ── 3: Deflection ────────────────────────────────────────────────────────
    ax = axes[3]
    style_ax(ax, "δ  (mm ↓)", ACC)
    ax.plot(x, y_mm, color=ACC, lw=1.8, zorder=3)
    ax.fill_between(x, y_mm, 0, alpha=0.18, color=ACC)
    ax.set_xlim(0, L)
    ax.invert_yaxis()   # positive downward on screen
    idx = np.argmax(np.abs(y_mm))
    ax.plot(x[idx], y_mm[idx], 'o', color=ACC, ms=5, zorder=4)
    ax.annotate(f" {abs(y_mm[idx]):.3f} mm", xy=(x[idx], y_mm[idx]),
                color=ACC, fontsize=8, fontfamily="monospace",
                va='top' if y_mm[idx] >= 0 else 'bottom')

    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Beam Setup")

    st.markdown('<div class="sidebar-section">BEAM PROPERTIES</div>',
                unsafe_allow_html=True)
    L      = st.number_input("Length  (m)",          0.5,  200.0,  5.0,  0.5)
    E_GPa  = st.number_input("Young's Modulus  (GPa)", 1.0, 500.0, 200.0, 10.0)
    E = E_GPa * 1e9

    cs = st.selectbox("Cross Section", ["Rectangular", "Circular", "I-Beam"])
    if cs == "Rectangular":
        b_mm = st.number_input("Width  b  (mm)",  1.0, 5000.0,  100.0)
        h_mm = st.number_input("Height h  (mm)", 1.0, 5000.0,  200.0)
        b_m, h_m = b_mm*1e-3, h_mm*1e-3
        I      = b_m * h_m**3 / 12
        c_dist = h_m / 2
    elif cs == "Circular":
        d_mm = st.number_input("Diameter  (mm)", 1.0, 5000.0, 100.0)
        d_m  = d_mm * 1e-3
        I      = np.pi * d_m**4 / 64
        c_dist = d_m / 2
    else:
        bf = st.number_input("Flange Width  bf  (mm)", 1.0, 2000.0, 150.0) * 1e-3
        tf = st.number_input("Flange Thick  tf  (mm)", 1.0,  500.0,  10.0) * 1e-3
        hw = st.number_input("Web Height    hw  (mm)", 1.0, 2000.0, 200.0) * 1e-3
        tw = st.number_input("Web Thick     tw  (mm)", 1.0,  500.0,   8.0) * 1e-3
        H      = hw + 2*tf
        I      = (bf * H**3 - (bf - tw) * hw**3) / 12
        c_dist = H / 2

    I_cm4 = I * 1e8  # m^4 → cm^4
    st.info(f"**I = {I_cm4:.3f} cm⁴**")

    st.markdown('<div class="sidebar-section">SUPPORT TYPE</div>',
                unsafe_allow_html=True)
    support = st.selectbox(
        "Boundary Conditions",
        ["Simply Supported", "Cantilever (Fixed at Left)"]
    )

    st.markdown('<div class="sidebar-section">POINT LOADS  ↓ +</div>',
                unsafe_allow_html=True)
    n_pl = st.number_input("Number of point loads", 0, 5, 1, key="npl")
    point_loads = []
    for i in range(int(n_pl)):
        c1, c2 = st.columns(2)
        xp   = c1.number_input("x (m)", 0.0, float(L), float(L/2), key=f"xp{i}")
        P_kN = c2.number_input("P (kN)", value=10.0, key=f"P{i}")
        if i == 0:
            c1.caption("Position")
            c2.caption("Magnitude")
        point_loads.append((float(xp), float(P_kN * 1e3)))

    st.markdown('<div class="sidebar-section">DISTRIBUTED LOADS  ↓ +</div>',
                unsafe_allow_html=True)
    n_dl = st.number_input("Number of UDLs", 0, 3, 0, key="ndl")
    dist_loads = []
    for i in range(int(n_dl)):
        ca, cb = st.columns(2)
        x1   = ca.number_input("x start (m)", 0.0, float(L), 0.0,       key=f"x1{i}")
        x2   = cb.number_input("x end   (m)", 0.0, float(L), float(L),  key=f"x2{i}")
        w_kN = st.number_input("Intensity (kN/m)", value=5.0, key=f"w{i}")
        dist_loads.append((float(x1), float(x2), float(w_kN * 1e3)))


# ── Main Panel ────────────────────────────────────────────────────────────────
st.markdown("# Beam Deflection Calculator")
st.caption("Simply Supported & Cantilever  ·  Point & Distributed Loads  ·  SFD / BMD / Deflection")

if not point_loads and not dist_loads:
    st.info("👈  Add loads in the sidebar to see results.")
    st.stop()

# Validate
error_msg = None
if len(dist_loads) > 0:
    for (x1, x2, w) in dist_loads:
        if x1 >= x2:
            error_msg = f"Distributed load: start ({x1} m) must be less than end ({x2} m)."

if error_msg:
    st.error(error_msg)
    st.stop()

# Solve
try:
    x_arr, V_arr, M_arr, y_arr, reactions = solve_beam(
        float(L), float(E), float(I),
        support,
        tuple(point_loads),
        tuple(dist_loads),
    )
except Exception as e:
    st.error(f"Solver error: {e}")
    st.stop()

# ── Summary Metrics ───────────────────────────────────────────────────────────
max_def    = float(np.max(np.abs(y_arr)))
x_max_def  = float(x_arr[np.argmax(np.abs(y_arr))])
max_M_abs  = float(np.max(np.abs(M_arr)))
max_V_abs  = float(np.max(np.abs(V_arr)))
max_stress = float(max_M_abs * 1e3 * c_dist / I / 1e6)   # MPa

m1, m2, m3, m4 = st.columns(4)
m1.metric("Max Deflection",     f"{max_def:.3f} mm",     f"at x = {x_max_def:.2f} m")
m2.metric("Max Moment",         f"{max_M_abs:.2f} kNm")
m3.metric("Max Shear",          f"{max_V_abs:.2f} kN")
m4.metric("Max Bending Stress", f"{max_stress:.1f} MPa")

# ── Reactions ─────────────────────────────────────────────────────────────────
st.markdown("#### Reactions")
r_cols = st.columns(len(reactions))
for col, (name, val) in zip(r_cols, reactions.items()):
    col.metric(name, f"{val:.3f}")

st.markdown("---")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = make_figure(x_arr, V_arr, M_arr, y_arr,
                  float(L), point_loads, dist_loads, support)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── Data Table ────────────────────────────────────────────────────────────────
with st.expander("📊 Tabular Data  (every 10th node)"):
    import pandas as pd
    step = max(1, len(x_arr) // 50)
    df = pd.DataFrame({
        "x (m)":    np.round(x_arr[::step], 3),
        "V (kN)":   np.round(V_arr[::step], 3),
        "M (kNm)":  np.round(M_arr[::step], 3),
        "δ (mm)":   np.round(y_arr[::step], 4),
    })
    st.dataframe(df, use_container_width=True, height=300)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<br>
<p style='color:#444; font-size:0.75rem; font-family:monospace; text-align:center;'>
Beam Deflection Calculator · Method of Integration · E·I·y'' = M(x)<br>
Simply Supported: y(0)=y(L)=0 · Cantilever: y(0)=y'(0)=0
</p>
""", unsafe_allow_html=True)
