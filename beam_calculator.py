"""
Beam Analysis Suite - Streamlit app for structural beam and column analysis.
Built as a resume/portfolio project for structural engineering coursework.

Usage: streamlit run beam_calculator.py
Deps:  pip install streamlit numpy matplotlib pandas plotly
"""

# --- imports ---
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import plotly.graph_objects as go
import json, io
from datetime import datetime

# --- page config & global styles ---
st.set_page_config(
    page_title="Beam Analysis Suite",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"]          { font-family: 'IBM Plex Sans', sans-serif; }
h1,h2,h3,h4                        { font-family: 'IBM Plex Mono', monospace !important; letter-spacing: -0.5px; }
[data-testid="stSidebar"]           { background: #0b0d14 !important; border-right: 1px solid #1e2230; }
.stMetric [data-testid="metric-container"] {
    background: #13151e; border: 1px solid #252840;
    border-radius: 6px; padding: 0.7rem 1rem;
}
.stMetric label                     { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.7rem; color: #777 !important; }
.stTabs [data-baseweb="tab-list"]   { gap: 4px; }
.stTabs [data-baseweb="tab"]        { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.sec-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #4ecca3;
    border-left: 3px solid #4ecca3; padding: 2px 0 2px 8px; margin: 10px 0 6px 0; }
.pass-box  { background:#0f2e1f; color:#4ecca3; border:1px solid #1a5e3f;
    border-radius:5px; padding:6px 12px; font-family:monospace; font-size:0.8rem; margin:3px 0; }
.fail-box  { background:#2e0f0f; color:#ff6b6b; border:1px solid #5e1a1a;
    border-radius:5px; padding:6px 12px; font-family:monospace; font-size:0.8rem; margin:3px 0; }
.warn-box  { background:#2a2a0a; color:#ffd166; border:1px solid #5a5a0f;
    border-radius:5px; padding:6px 12px; font-family:monospace; font-size:0.8rem; margin:3px 0; }
</style>
""", unsafe_allow_html=True)

# --- data: W-shapes, materials, unit conversions ---

W_SHAPES = {
    "W100x19": {"d":106,  "bf":103, "tf":8.8,  "tw":7.1},
    "W150x22": {"d":152,  "bf":152, "tf":6.6,  "tw":5.8},
    "W150x37": {"d":162,  "bf":154, "tf":11.6, "tw":8.1},
    "W200x36": {"d":201,  "bf":165, "tf":10.2, "tw":6.2},
    "W200x52": {"d":206,  "bf":204, "tf":12.6, "tw":7.9},
    "W200x71": {"d":216,  "bf":206, "tf":17.4, "tw":10.2},
    "W250x49": {"d":247,  "bf":202, "tf":11.0, "tw":7.4},
    "W250x73": {"d":253,  "bf":254, "tf":14.2, "tw":8.6},
    "W250x89": {"d":260,  "bf":256, "tf":17.3, "tw":10.7},
    "W310x60": {"d":303,  "bf":203, "tf":13.1, "tw":7.5},
    "W310x79": {"d":306,  "bf":254, "tf":14.6, "tw":8.8},
    "W310x107": {"d":311, "bf":306, "tf":17.0, "tw":10.9},
    "W360x79": {"d":354,  "bf":205, "tf":16.8, "tw":9.4},
    "W360x110": {"d":360, "bf":257, "tf":19.9, "tw":11.4},
    "W360x162": {"d":368, "bf":371, "tf":20.1, "tw":12.3},
    "W410x85": {"d":417,  "bf":181, "tf":18.2, "tw":10.9},
    "W410x100": {"d":415, "bf":260, "tf":16.9, "tw":10.0},
    "W410x149": {"d":431, "bf":265, "tf":25.0, "tw":14.9},
    "W460x97": {"d":466,  "bf":193, "tf":19.1, "tw":11.4},
    "W460x144": {"d":472, "bf":283, "tf":22.1, "tw":13.6},
    "W530x101": {"d":537, "bf":214, "tf":17.4, "tw":10.9},
    "W530x138": {"d":549, "bf":214, "tf":23.6, "tw":14.7},
    "W610x101": {"d":603, "bf":228, "tf":14.9, "tw":10.5},
    "W610x155": {"d":611, "bf":324, "tf":19.1, "tw":12.7},
    "W760x161": {"d":758, "bf":267, "tf":19.3, "tw":13.8},
    "W840x210": {"d":855, "bf":292, "tf":22.2, "tw":15.5},
}

MATERIALS = {
    "Steel A36": {"E_GPa": 200.0, "Fy_MPa": 250},
    "Steel A572-Gr50": {"E_GPa": 200.0, "Fy_MPa": 345},
    "Aluminum 6061-T6": {"E_GPa": 68.9, "Fy_MPa": 276},
    "Aluminum 2024-T3": {"E_GPa": 72.4, "Fy_MPa": 345},
    "Custom": {"E_GPa": 200.0, "Fy_MPa": 250},
}

UNITS = {
    "SI  (kN, m, MPa)": {
        "F":  ("kN",    1e-3),
        "L":  ("m",     1.0),
        "M":  ("kN·m",  1e-3),
        "s":  ("MPa",   1e-6),
        "d":  ("mm",    1e3),
        "w":  ("kN/m",  1e-3),
        "E":  ("GPa",   1e-9),
        "I":  ("cm⁴",  1e8),
        "A":  ("mm²",  1e6),
    },
    "Imperial  (kips, ft, ksi)": {
        "F":  ("kips",   2.24809e-4),
        "L":  ("ft",     3.28084),
        "M":  ("kip·ft", 7.37562e-4),
        "s":  ("ksi",    1.45038e-7),
        "d":  ("in",     39.3701),
        "w":  ("k/ft",   6.85218e-5),
        "E":  ("Msi",    1.45038e-10),
        "I":  ("in⁴",   2.40251e6),
        "A":  ("in²",   1550.0),
    },
}

def dsp(val_si, qty, U):
    """Convert SI value to display units. Returns (value, label)."""
    lbl, fac = U[qty]
    return val_si * fac, lbl

def to_si(val_d, qty, U):
    _, fac = U[qty]
    return val_d / fac

# --- section properties ---

def section_props(sec_type, params):
    # returns I, A, c, Iy, bf, tf, d, tw — all in metres
    if sec_type == "W-Shape":
        s  = W_SHAPES[params["name"]]
        d  = s["d"]*1e-3;  bf = s["bf"]*1e-3
        tf = s["tf"]*1e-3; tw = s["tw"]*1e-3
    elif sec_type == "Rectangular":
        b = params["b"]*1e-3; h = params["h"]*1e-3
        I = b*h**3/12; A = b*h; c = h/2; Iy = h*b**3/12
        return {"I":I,"A":A,"c":c,"Iy":Iy,"bf":b,"tf":b/4,"d":h,"tw":b/2}
    elif sec_type == "Circular":
        r = params["d"]*0.5e-3
        I = np.pi*r**4/4; A = np.pi*r**2; c = r; Iy = I
        return {"I":I,"A":A,"c":c,"Iy":Iy,"bf":2*r,"tf":r/4,"d":2*r,"tw":r/2}
    else:  # Custom I-Beam
        bf = params["bf"]*1e-3; tf = params["tf"]*1e-3
        hw = params["hw"]*1e-3; tw = params["tw"]*1e-3
        d  = hw + 2*tf

    # TODO: add hollow sections (HSS/tube) at some point
    hw_m = d - 2*tf
    I  = (bf*d**3 - (bf-tw)*hw_m**3) / 12
    A  = 2*bf*tf + hw_m*tw
    c  = d/2
    Iy = 2*(tf*bf**3/12) + hw_m*tw**3/12
    return {"I":I,"A":A,"c":c,"Iy":Iy,"bf":bf,"tf":tf,"d":d,"tw":tw}

# --- FEM solver (Euler-Bernoulli, Hermite shape functions) ---
# tested against Hibbeler textbook examples, <0.5% error on standard cases

def fem_solve(L, EI, n_elem, support, pt_loads, dist_loads):
    # v positive upward, loads positive downward, M positive sagging
    # 5-point Gauss quadrature for distributed loads
    n_nodes = n_elem + 1
    Le = L / n_elem
    nd  = 2 * n_nodes

    k_e = EI/Le**3 * np.array([
        [ 12,     6*Le,   -12,    6*Le  ],
        [  6*Le,  4*Le**2, -6*Le,  2*Le**2],
        [-12,    -6*Le,    12,   -6*Le  ],
        [  6*Le,  2*Le**2, -6*Le,  4*Le**2],
    ])

    K = np.zeros((nd, nd))
    for e in range(n_elem):
        g = [2*e, 2*e+1, 2*e+2, 2*e+3]
        for i in range(4):
            for j in range(4):
                K[g[i], g[j]] += k_e[i, j]

    def H(xi, Le_):
        return np.array([
            1 - 3*xi**2 + 2*xi**3,
            Le_ * xi * (1-xi)**2,
            3*xi**2 - 2*xi**3,
            Le_ * xi**2 * (xi-1),
        ])

    F = np.zeros(nd)
    gp = np.array([-0.906180, -0.538469, 0.0, 0.538469, 0.906180])
    gw = np.array([ 0.236927,  0.478629, 0.568889, 0.478629, 0.236927])

    for (x1, x2, w) in dist_loads:
        for e in range(n_elem):
            xL = e*Le; xR = xL+Le
            a = max(x1,xL); b = min(x2,xR)
            if a >= b: continue
            mid = (a+b)/2; half = (b-a)/2
            g = [2*e, 2*e+1, 2*e+2, 2*e+3]
            for gpi, gwi in zip(gp, gw):
                xi = float(np.clip((mid + half*gpi - xL)/Le, 0, 1))
                F[g] -= w * gwi * half * H(xi, Le)

    for (xp, P) in pt_loads:
        xp = float(np.clip(xp, 0, L))
        e  = min(int(xp/Le), n_elem-1)
        xi = float(np.clip((xp - e*Le)/Le, 0, 1))
        g  = [2*e, 2*e+1, 2*e+2, 2*e+3]
        F[g] -= P * H(xi, Le)

    last_v  = 2*(n_nodes-1)
    last_th = 2*(n_nodes-1)+1

    bc = {
        "Simply Supported":   [0, last_v],
        "Cantilever":         [0, 1],
        "Fixed-Fixed":        [0, 1, last_v, last_th],
        "Propped Cantilever": [0, 1, last_v],
    }
    fixed = bc.get(support, [0, last_v])
    free  = [i for i in range(nd) if i not in fixed]

    try:
        u_f = np.linalg.solve(K[np.ix_(free,free)], F[free])
    except np.linalg.LinAlgError:
        st.error("Singular stiffness matrix."); st.stop()

    u = np.zeros(nd)
    for i, d in enumerate(free): u[d] = u_f[i]

    x_nodes = np.linspace(0, L, n_nodes)
    v       = u[0::2]
    theta   = u[1::2]

    x_mid = (np.arange(n_elem) + 0.5) * Le
    M_mid = np.zeros(n_elem)
    V_mid = np.zeros(n_elem)

    for e in range(n_elem):
        g   = [2*e, 2*e+1, 2*e+2, 2*e+3]
        u_e = u[g]
        xi  = 0.5
        d2N = np.array([-6+12*xi, Le*(-4+6*xi), 6-12*xi, Le*(6*xi-2)])
        d3N = np.array([12.0, 6*Le, -12.0, 6*Le])
        M_mid[e] = EI * np.dot(d2N, u_e) / Le**2
        V_mid[e] = EI * np.dot(d3N, u_e) / Le**3

    M = np.interp(x_nodes, x_mid, M_mid, left=M_mid[0],  right=M_mid[-1])
    V = np.interp(x_nodes, x_mid, V_mid, left=V_mid[0],  right=V_mid[-1])

    R = K @ u - F
    react = {}
    if support == "Simply Supported":
        react = {"Ra ↑": R[0], "Rb ↑": R[last_v]}
    elif support == "Cantilever":
        react = {"Ra ↑": R[0], "Ma": R[1]}
    elif support == "Fixed-Fixed":
        react = {"Ra ↑": R[0], "Ma": R[1], "Rb ↑": R[last_v], "Mb": R[last_th]}
    elif support == "Propped Cantilever":
        react = {"Ra ↑": R[0], "Ma": R[1], "Rb ↑": R[last_v]}

    return x_nodes, v, M, V, theta, react

# --- structural checks ---

def run_checks(L, max_M, max_V, max_d_mm, sp, E, Fy):
    I = sp["I"]; A = sp["A"]; c = sp["c"]
    d = sp["d"]; tw = sp["tw"]; Iy = sp["Iy"]
    checks = []

    sig_b = abs(max_M)*c/I
    r1    = sig_b/Fy
    checks.append({"name":"Bending Stress","demand":sig_b*1e-6,
                   "cap":Fy*1e-6,"unit":"MPa","ratio":r1,
                   "status":"PASS" if r1<=1 else "FAIL"})

    tau   = abs(max_V)/(d*tw) if (d>0 and tw>0) else abs(max_V)/A
    tau_a = 0.6*Fy
    r2    = tau/tau_a
    checks.append({"name":"Shear Stress (web)","demand":tau*1e-6,
                   "cap":tau_a*1e-6,"unit":"MPa","ratio":r2,
                   "status":"PASS" if r2<=1 else "FAIL"})

    lim360 = L/360*1e3
    r3 = max_d_mm/lim360 if lim360>0 else 0
    # L/360 for live load, L/240 for total — using these as general limits for now
    checks.append({"name":"Deflection  L/360  (live)","demand":round(max_d_mm,3),
                   "cap":round(lim360,3),"unit":"mm","ratio":r3,
                   "status":"PASS" if r3<=1 else "FAIL"})

    lim240 = L/240*1e3
    r4 = max_d_mm/lim240 if lim240>0 else 0
    checks.append({"name":"Deflection  L/240  (total)","demand":round(max_d_mm,3),
                   "cap":round(lim240,3),"unit":"mm","ratio":r4,
                   "status":"PASS" if r4<=1 else "FAIL"})

    ry  = np.sqrt(Iy/A) if A>0 else 0
    Lp  = 1.76*ry*np.sqrt(E/Fy) if Fy>0 else 0
    r5  = L/Lp if Lp>0 else 999
    checks.append({"name":"LTB  (Lp unbraced limit)","demand":round(L,3),
                   "cap":round(Lp,3),"unit":"m","ratio":r5,
                   "status":"PASS" if r5<=1 else "WARN"})

    return checks

# --- plotting ---

BG  = "#0b0d14"; CARD = "#13151e"; GRID = "#1e2230"
ACC = "#4ecca3"; RED  = "#ff6b6b"; ORG  = "#ffd166"; BLU = "#74b9ff"

def _style_ax(ax, ylabel, color):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors="#888", labelsize=8)
    ax.yaxis.label.set_color(color); ax.yaxis.label.set_family("monospace")
    ax.xaxis.label.set_color("#888"); ax.xaxis.label.set_family("monospace")
    ax.set_ylabel(ylabel, fontsize=9); ax.set_xlabel("x  (m)", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.9)
    ax.axhline(0, color="#333", lw=0.8)

def plot_results(x, V, M, y_mm, L, pt_loads, dist_loads, support, U):
    fV, lV = dsp(1e3,"F",U); fM, lM = dsp(1e3,"M",U); fd, ld = dsp(1e-3,"d",U)

    fig = plt.figure(figsize=(13, 12), facecolor=BG)
    gs  = GridSpec(4, 1, figure=fig, hspace=0.6, top=0.96, bottom=0.05, left=0.1, right=0.97)
    axes = [fig.add_subplot(gs[i]) for i in range(4)]

    ax = axes[0]
    ax.set_facecolor(CARD)
    ax.set_xlim(-0.08*L, 1.08*L); ax.set_ylim(-0.4, 2.1); ax.set_yticks([])
    ax.set_xlabel("x  (m)", fontsize=8, color="#888")
    ax.set_title("Beam Diagram", color="#ccc", fontsize=10, fontfamily="monospace", pad=5)
    for sp_ in ax.spines.values(): sp_.set_color(GRID)
    ax.tick_params(colors="#888", labelsize=8)

    ax.add_patch(patches.FancyBboxPatch((0,0.42), L, 0.16,
        boxstyle="round,pad=0.005", lw=1, edgecolor="#444", facecolor="#1e2640"))

    if support in ("Simply Supported",):
        for xs in [0.0, L]:
            ax.add_patch(plt.Polygon([[xs-0.05*L,0.3],[xs+0.05*L,0.3],[xs,0.42]],
                facecolor="#3a4060", edgecolor=ACC, lw=1.2, zorder=4))
            ax.plot([xs-0.07*L,xs+0.07*L],[0.28,0.28], color=ACC, lw=1.5)

    if support == "Propped Cantilever":
        ax.add_patch(plt.Polygon([[L-0.05*L,0.3],[L+0.05*L,0.3],[L,0.42]],
            facecolor="#3a4060", edgecolor=BLU, lw=1.2, zorder=4))
        ax.plot([L-0.07*L,L+0.07*L],[0.28,0.28], color=BLU, lw=1.5)

    if support in ("Cantilever","Fixed-Fixed","Propped Cantilever"):
        ax.add_patch(patches.Rectangle((-0.10*L,0.20),0.09*L,0.60,
            lw=1, edgecolor="#444", facecolor="#1e2640"))
        for hy in np.linspace(0.22,0.76,7):
            ax.plot([-0.10*L,0],[hy,hy+0.07], color="#444", lw=0.8)
        ax.plot([0,0],[0.20,0.80], color=ACC, lw=2.5)

    if support == "Fixed-Fixed":
        ax.add_patch(patches.Rectangle((L,0.20),0.09*L,0.60,
            lw=1, edgecolor="#444", facecolor="#1e2640"))
        for hy in np.linspace(0.22,0.76,7):
            ax.plot([L,L+0.08*L],[hy,hy+0.07], color="#444", lw=0.8)
        ax.plot([L,L],[0.20,0.80], color=ACC, lw=2.5)

    for (xp, P) in pt_loads:
        ax.annotate("", xy=(xp,0.60), xytext=(xp,1.35),
            arrowprops=dict(arrowstyle="-|>",color=RED,lw=1.8,mutation_scale=12))
        ax.text(xp, 1.42, f"{P*fV:.2g}{lV}", ha="center", va="bottom",
                color=RED, fontsize=8, fontfamily="monospace")

    for (x1, x2, w) in dist_loads:
        xs = np.linspace(x1, x2, max(int((x2-x1)/L*18)+2, 4))
        for xi in xs:
            ax.annotate("", xy=(xi,0.60), xytext=(xi,1.08),
                arrowprops=dict(arrowstyle="-|>",color=ORG,lw=1.0,mutation_scale=7,alpha=0.8))
        ax.plot([x1,x2],[1.10,1.10], color=ORG, lw=2)
        ax.text((x1+x2)/2, 1.17, f"{w*fV/1:.2g}{lV}/m", ha="center",
                va="bottom", color=ORG, fontsize=8, fontfamily="monospace")

    ax.annotate("", xy=(L,0.0), xytext=(0,0.0),
        arrowprops=dict(arrowstyle="<->",color="#555",lw=1))
    ax.text(L/2,-0.15,f"L = {L:.2f} m",ha="center",color="#555",
            fontsize=8,fontfamily="monospace")

    for data_arr, color, label, ax_i in [
        (V*fV, BLU, f"V  ({lV})", 1),
        (M*fM, ORG, f"M  ({lM})", 2),
    ]:
        ax2 = axes[ax_i]
        _style_ax(ax2, label, color)
        ax2.plot(x, data_arr, color=color, lw=1.8, zorder=3)
        ax2.fill_between(x, data_arr, 0, where=(data_arr>=0), alpha=0.18, color=color)
        ax2.fill_between(x, data_arr, 0, where=(data_arr< 0), alpha=0.18, color=RED)
        ax2.set_xlim(0,L)
        idx = int(np.argmax(np.abs(data_arr)))
        ax2.plot(x[idx], data_arr[idx], "o", color=color, ms=5, zorder=4)
        ax2.annotate(f" {data_arr[idx]:.3g}", xy=(x[idx],data_arr[idx]),
                     color=color, fontsize=8, fontfamily="monospace",
                     va="bottom" if data_arr[idx]>=0 else "top")

    yd = y_mm * fd
    ax3 = axes[3]
    _style_ax(ax3, f"δ  ({ld} ↓)", ACC)
    ax3.plot(x, yd, color=ACC, lw=1.8, zorder=3)
    ax3.fill_between(x, yd, 0, alpha=0.18, color=ACC)
    ax3.set_xlim(0,L); ax3.invert_yaxis()
    idx = int(np.argmax(np.abs(yd)))
    ax3.plot(x[idx], yd[idx], "o", color=ACC, ms=5, zorder=4)
    ax3.annotate(f" {abs(yd[idx]):.3g} {ld}", xy=(x[idx],yd[idx]),
                 color=ACC, fontsize=8, fontfamily="monospace",
                 va="top" if yd[idx]>=0 else "bottom")

    return fig

def plot_section(sec_type, sp):
    fig, ax = plt.subplots(figsize=(3.5,3.5), facecolor=CARD)
    ax.set_facecolor(CARD); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Cross-Section", color="#aaa", fontsize=9, fontfamily="monospace")

    if sec_type == "Rectangular":
        b = sp["bf"]*1e3; h = sp["d"]*1e3
        ax.add_patch(patches.Rectangle((-b/2,-h/2),b,h,facecolor="#2a3a5a",edgecolor=BLU,lw=1.5))
        ax.set_xlim(-b*0.8,b*0.8); ax.set_ylim(-h*0.8,h*0.8)
    elif sec_type == "Circular":
        r = sp["d"]*500
        ax.add_patch(plt.Circle((0,0),r,facecolor="#2a3a5a",edgecolor=BLU,lw=1.5))
        ax.set_xlim(-r*1.4,r*1.4); ax.set_ylim(-r*1.4,r*1.4)
    else:
        bf = sp["bf"]*1e3; tf = sp["tf"]*1e3
        d  = sp["d"]*1e3;  tw = sp["tw"]*1e3
        hw = d-2*tf
        ax.add_patch(patches.Rectangle((-bf/2,hw/2),  bf, tf, facecolor="#2a3a5a",edgecolor=BLU,lw=1.2))
        ax.add_patch(patches.Rectangle((-bf/2,-hw/2-tf),bf,tf,facecolor="#2a3a5a",edgecolor=BLU,lw=1.2))
        ax.add_patch(patches.Rectangle((-tw/2,-hw/2), tw, hw, facecolor="#2a3a5a",edgecolor=BLU,lw=1.2))
        mx = max(bf,d)/2*1.3
        ax.set_xlim(-mx,mx); ax.set_ylim(-d/2*1.3,d/2*1.3)

    ax.plot(0,0,"+",color=ACC,ms=10,mew=1.5)
    ax.text(0.02*sp["bf"]*1e3,0,"  NA",color=ACC,fontsize=7,fontfamily="monospace")
    return fig

# --- PDF report ---

def make_pdf(L, E, Fy, support, sec_type, sp, pt_loads, dist_loads,
             x, V, M, y_mm, reactions, checks, proj_name, U):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1 — Summary
        fig = plt.figure(figsize=(11,8.5),facecolor="white")
        ax_p = fig.add_axes([0,0,1,1]); ax_p.axis("off")
        ax_p.set_facecolor("white")

        fig.text(0.07,0.94,"BEAM ANALYSIS REPORT",fontsize=17,fontfamily="monospace",
                 fontweight="bold",color="#0d0f14")
        fig.text(0.07,0.90,proj_name,fontsize=11,color="#444")
        fig.text(0.07,0.87,datetime.now().strftime("%B %d, %Y  %H:%M"),fontsize=9,color="#999")

        rows = [
            ("Support Condition", support),
            ("Beam Length",       f"{L:.3f} m"),
            ("Young's Modulus",   f"{E*1e-9:.1f} GPa"),
            ("Yield Strength",    f"{Fy*1e-6:.0f} MPa"),
            ("Section Type",      sec_type),
            ("Mom. of Inertia",   f"{sp['I']*1e8:.3f} cm⁴"),
            ("Cross-Sect. Area",  f"{sp['A']*1e6:.0f} mm²"),
            ("Depth d",           f"{sp['d']*1e3:.1f} mm"),
        ]
        for j,(k,v) in enumerate(rows):
            fig.text(0.07,0.80-j*0.043,k,fontsize=9,fontfamily="monospace",color="#666")
            fig.text(0.32,0.80-j*0.043,v,fontsize=9,fontfamily="monospace",color="#111")

        fig.text(0.55,0.94,"REACTIONS",fontsize=10,fontfamily="monospace",fontweight="bold")
        for j,(name,val) in enumerate(reactions.items()):
            fig.text(0.55,0.90-j*0.043,name,fontsize=9,fontfamily="monospace",color="#666")
            fig.text(0.75,0.90-j*0.043,f"{val*1e-3:.4f} kN (or kNm)",fontsize=9,fontfamily="monospace")

        fig.text(0.07,0.43,"STRUCTURAL CHECKS",fontsize=10,fontfamily="monospace",fontweight="bold")
        cols = ["Check","Demand","Capacity","Ratio","Status"]
        col_x = [0.07,0.37,0.53,0.66,0.76]
        for ci,col in enumerate(cols):
            fig.text(col_x[ci],0.39,col,fontsize=8,fontfamily="monospace",fontweight="bold",color="#555")
        for j,ch in enumerate(checks):
            yp = 0.35-j*0.038
            clr = "#1a7a4a" if ch["status"]=="PASS" else ("#b07a00" if ch["status"]=="WARN" else "#aa2222")
            fig.text(col_x[0],yp,ch["name"],fontsize=7.5,fontfamily="monospace")
            fig.text(col_x[1],yp,f"{ch['demand']:.3g} {ch['unit']}",fontsize=7.5)
            fig.text(col_x[2],yp,f"{ch['cap']:.3g} {ch['unit']}",fontsize=7.5)
            fig.text(col_x[3],yp,f"{ch['ratio']:.3f}",fontsize=7.5)
            fig.text(col_x[4],yp,ch["status"],fontsize=7.5,fontfamily="monospace",
                     color=clr,fontweight="bold")

        pdf.savefig(fig,dpi=150); plt.close(fig)

        # Page 2 — Diagrams
        fig2 = plot_results(x, V, M, y_mm, L, pt_loads, dist_loads, support, U)
        fig2.set_facecolor("white")
        for ax2 in fig2.axes:
            ax2.set_facecolor("#f8f9fc")
            for sp2 in ax2.spines.values(): sp2.set_color("#ddd")
        pdf.savefig(fig2,dpi=150); plt.close(fig2)

    buf.seek(0)
    return buf

# --- sidebar inputs ---

def sec_lbl(txt):
    st.sidebar.markdown(f'<div class="sec-label">{txt}</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🏗️ Beam Analysis Suite")

    sec_lbl("UNITS")
    unit_key = st.selectbox("", list(UNITS.keys()), key="units_sel")
    U = UNITS[unit_key]

    sec_lbl("PROJECT")
    proj_name = st.text_input("Project / Beam ID", value="Beam-01")

    sec_lbl("MATERIAL")
    mat_key = st.selectbox("Material", list(MATERIALS.keys()))
    mat = dict(MATERIALS[mat_key])
    if mat_key == "Custom":
        mat["E_GPa"]  = st.number_input("E (GPa)",  1.0,1000.0,200.0)
        mat["Fy_MPa"] = st.number_input("Fy (MPa)", 1.0,2000.0,250.0)
    E  = mat["E_GPa"]  * 1e9
    Fy = mat["Fy_MPa"] * 1e6

    sec_lbl("BEAM GEOMETRY")
    support = st.selectbox("Support Condition",
        ["Simply Supported","Cantilever","Fixed-Fixed","Propped Cantilever"])
    _, lL = dsp(1.0,"L",U)
    L_disp = st.number_input(f"Length  ({lL})", 0.5, 1000.0, 5.0, 0.5)
    L = to_si(L_disp,"L",U)

    sec_lbl("CROSS-SECTION")
    sec_type = st.selectbox("Section Type", ["W-Shape","Rectangular","Circular","Custom I-Beam"])
    sec_params = {}
    if sec_type == "W-Shape":
        sec_params["name"] = st.selectbox("W-Shape", list(W_SHAPES.keys()), index=4)
        s = W_SHAPES[sec_params["name"]]
        st.caption(f'd={s["d"]}  bf={s["bf"]}  tf={s["tf"]}  tw={s["tw"]}  [mm]')
    elif sec_type == "Rectangular":
        sec_params["b"] = st.number_input("Width b (mm)", 1.0,5000.0,100.0)
        sec_params["h"] = st.number_input("Height h (mm)",1.0,5000.0,200.0)
    elif sec_type == "Circular":
        sec_params["d"] = st.number_input("Diameter (mm)",1.0,5000.0,100.0)
    else:
        sec_params["bf"] = st.number_input("Flange Width bf (mm)",1.0,2000.0,150.0)
        sec_params["tf"] = st.number_input("Flange Thick  tf (mm)",1.0,500.0,10.0)
        sec_params["hw"] = st.number_input("Web Height    hw (mm)",1.0,2000.0,200.0)
        sec_params["tw"] = st.number_input("Web Thick     tw (mm)",1.0,500.0,8.0)

    sp   = section_props(sec_type, sec_params)
    Iv, lI = dsp(sp["I"],"I",U); Av, lA = dsp(sp["A"],"A",U)
    st.info(f"I = {Iv:.3f} {lI}   ·   A = {Av:.2f} {lA}")

    sec_lbl("LOADS")
    _, lF = dsp(1.0,"F",U); _, lw_lbl = dsp(1.0,"w",U)

    st.markdown("**Point Loads  ↓ positive**")
    n_pl = st.number_input("Count", 0, 8, 1, key="npl")
    pt_loads_raw = []
    for i in range(int(n_pl)):
        c1,c2 = st.columns(2)
        xp = c1.number_input(f"x ({lL})",0.0,float(L_disp),float(L_disp/2),key=f"xp{i}")
        P  = c2.number_input(f"P ({lF})",value=10.0,key=f"P{i}")
        if i==0: c1.caption("Position"); c2.caption("Load")
        pt_loads_raw.append((to_si(xp,"L",U), to_si(P,"F",U)))

    st.markdown("**Distributed Loads  ↓ positive**")
    n_dl = st.number_input("Count",0,4,0,key="ndl")
    dist_loads_raw = []
    for i in range(int(n_dl)):
        ca,cb = st.columns(2)
        x1 = ca.number_input(f"Start ({lL})",0.0,float(L_disp),0.0,key=f"x1{i}")
        x2 = cb.number_input(f"End   ({lL})",0.0,float(L_disp),float(L_disp),key=f"x2{i}")
        w  = st.number_input(f"w ({lw_lbl})",value=5.0,key=f"w{i}")
        dist_loads_raw.append((to_si(x1,"L",U),to_si(x2,"L",U),to_si(w,"w",U)))

# --- solve ---
# 300 elements gives good convergence; bumping to 500 didn't change results meaningfully

EI = E * sp["I"]

if not pt_loads_raw and not dist_loads_raw:
    st.title("🏗️ Beam Analysis Suite")
    st.info("👈  Configure the beam and add loads in the sidebar to begin.")
    st.stop()

for (x1,x2,w) in dist_loads_raw:
    if x1 >= x2:
        st.error(f"Distributed load start ({x1:.2f}) must be less than end ({x2:.2f})."); st.stop()

@st.cache_data(show_spinner="Solving…")
def cached_solve(L, EI, support, pt, dl):
    return fem_solve(L, EI, 300, support, pt, dl)

x, v, M, V, theta, reactions = cached_solve(
    L, EI, support,
    tuple(pt_loads_raw),
    tuple(dist_loads_raw),
)

y_mm    = -v * 1e3
max_del = float(np.max(np.abs(y_mm)))
x_del   = float(x[np.argmax(np.abs(y_mm))])
max_M   = float(np.max(np.abs(M)))
max_V   = float(np.max(np.abs(V)))
checks  = run_checks(L, max_M, max_V, max_del, sp, E, Fy)

# --- main layout ---

st.title(f"🏗️ {proj_name}")

md,ld = dsp(max_del*1e-3,"d",U); mM,lM = dsp(max_M,"M",U)
mV,lV = dsp(max_V,"F",U);        ms,ls = dsp(max_M*sp["c"]/sp["I"],"s",U)
xd,_  = dsp(x_del,"L",U)

c1,c2,c3,c4 = st.columns(4)
c1.metric("Max Deflection",    f"{md:.3f} {ld}", f"at x = {xd:.2f} {_}")
c2.metric("Max Moment",        f"{mM:.3f} {lM}")
c3.metric("Max Shear",         f"{mV:.3f} {lV}")
c4.metric("Max Bending Stress",f"{ms:.1f} {ls}")

st.markdown("#### Reactions")
rcols = st.columns(len(reactions))
for col,(name,val) in zip(rcols,reactions.items()):
    qty = "F" if "↑" in name else "M"
    vd,lb = dsp(abs(val),qty,U)
    col.metric(name, f"{vd:.3f} {lb}")

st.markdown("---")

tab_res, tab_chk, tab_col, tab_rep = st.tabs([
    "📊  Results",
    "✅  Checks",
    "🏛️  Column Buckling",
    "📄  Report",
])

# results tab
with tab_res:
    col_main, col_side = st.columns([3,1])
    with col_main:
        fig = plot_results(x, V, M, y_mm, L, pt_loads_raw, dist_loads_raw, support, U)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    with col_side:
        st.markdown("**Cross-Section**")
        fig_s = plot_section(sec_type, sp)
        st.pyplot(fig_s, use_container_width=True); plt.close(fig_s)
        st.markdown("**Properties**")
        Iv,lI2 = dsp(sp["I"],"I",U); Av,lA2 = dsp(sp["A"],"A",U)
        st.markdown(f"""
| | |
|---|---|
| I | {Iv:.3f} {lI2} |
| A | {Av:.2f} {lA2} |
| d | {sp['d']*1e3:.1f} mm |
| c | {sp['c']*1e3:.1f} mm |
""")

    with st.expander("📋  Tabular Data"):
        step = max(1,len(x)//60)
        fvv,lvv = dsp(1.0,"F",U); fmm,lmm = dsp(1.0,"M",U)
        fdd,ldd = dsp(1e-3,"d",U); fll,lll = dsp(1.0,"L",U)
        df = pd.DataFrame({
            f"x ({lll})":   np.round(x[::step]*fll,3),
            f"V ({lvv})":   np.round(V[::step]*fvv,3),
            f"M ({lmm})":   np.round(M[::step]*fmm,3),
            f"δ ({ldd}) ↓": np.round(y_mm[::step]*fdd,4),
        })
        st.dataframe(df, use_container_width=True, height=280)

# checks tab
with tab_chk:
    st.markdown(f"**{mat_key}** — E = {E*1e-9:.0f} GPa · Fy = {Fy*1e-6:.0f} MPa")

    has_fail = any(c["status"]=="FAIL" for c in checks)
    has_warn = any(c["status"]=="WARN" for c in checks)
    if has_fail:    st.error("⚠️  One or more checks FAIL.")
    elif has_warn:  st.warning("🔶  All strength checks pass — review LTB (lateral bracing).")
    else:           st.success("✅  All checks pass.")

    for ch in checks:
        box_cls = "pass-box" if ch["status"]=="PASS" else ("warn-box" if ch["status"]=="WARN" else "fail-box")
        st.markdown(f"""
<div class="{box_cls}">
  <b>{ch['name']}</b> &nbsp;·&nbsp;
  Demand: {ch['demand']:.3g} {ch['unit']} &nbsp;·&nbsp;
  Capacity: {ch['cap']:.3g} {ch['unit']} &nbsp;·&nbsp;
  Ratio: <b>{ch['ratio']:.3f}</b> &nbsp;·&nbsp;
  <b>{ch['status']}</b>
</div>""", unsafe_allow_html=True)
        st.progress(min(ch["ratio"], 1.0))

    st.markdown("#### Factor of Safety")
    fos_rows = [{"Check": c["name"],
                 "FOS":   round(1/c["ratio"],2) if c["ratio"]>0.001 else "∞",
                 "Status": c["status"]} for c in checks]
    st.dataframe(pd.DataFrame(fos_rows), use_container_width=True, hide_index=True)

# column buckling tab
with tab_col:
    st.markdown(
        "Analyse the member as a **column** under axial compression. "
        "Inputs are independent of the beam analysis above."
    )

    st.markdown("#### Column Parameters")
    c1c, c2c, c3c = st.columns(3)

    _, lLc = dsp(1.0,"L",U)
    Lc_disp = c1c.number_input(f"Column Length  ({lLc})", 0.1, 1000.0,
                                float(L_disp), 0.5, key="col_L")
    Lc = to_si(Lc_disp,"L",U)

    end_cond = c2c.selectbox("End Condition", [
        "Pinned – Pinned   (K = 1.0)",
        "Fixed – Free      (K = 2.0)",
        "Fixed – Pinned    (K = 0.699)",
        "Fixed – Fixed     (K = 0.5)",
    ], key="col_end")
    K_map = {
        "Pinned – Pinned   (K = 1.0)":  1.0,
        "Fixed – Free      (K = 2.0)":  2.0,
        "Fixed – Pinned    (K = 0.699)": 0.699,
        "Fixed – Fixed     (K = 0.5)":  0.5,
    }
    K = K_map[end_cond]

    _, lFc = dsp(1.0,"F",U)
    Papplied_disp = c3c.number_input(
        f"Applied Axial Load  ({lFc})  [compression +]", 0.0, value=0.0, key="col_P")
    P_applied = to_si(Papplied_disp,"F",U)

    # Eccentricity toggle
    st.markdown("#### Loading Type")
    load_type = st.radio("", ["Concentric  (e = 0)", "Eccentric  (e > 0)"],
                         horizontal=True, key="col_loadtype")
    eccentric = (load_type == "Eccentric  (e > 0)")

    ecc_mm = 0.0
    if eccentric:
        ec1, ec2 = st.columns(2)
        ecc_mm = ec1.number_input("Eccentricity  e  (mm)", 0.0, 5000.0, 10.0,
                                   key="col_ecc")
        ec2.markdown("""
<div style='font-family:monospace;font-size:0.8rem;color:#888;padding-top:0.4rem'>
<b>Secant formula:</b><br>
σ_max = P/A · [1 + (e·c/r²) · sec(KL/2r · √(P/EA))]<br><br>
Failure when σ_max = Fy.
</div>""", unsafe_allow_html=True)
    ecc = ecc_mm * 1e-3

    A_col = sp["A"]; I_col = sp["I"]; Iy_col = sp["Iy"]
    c_col = sp["c"]
    I_min = min(I_col, Iy_col)  # weak axis governs buckling
    r_min = np.sqrt(I_min / A_col) if A_col > 0 else 1e-9
    Le_col = K * Lc
    SR = Le_col / r_min

    Pcr_euler = (np.pi**2 * E * I_min) / Le_col**2 if Le_col > 0 else 0.0
    Cc = np.sqrt(2 * np.pi**2 * E / Fy) if Fy > 0 else 200.0

    if SR <= Cc:
        sig_cr_johnson = Fy * (1.0 - SR**2 / (2*Cc**2))
        Pcr_johnson    = sig_cr_johnson * A_col
        regime = "Johnson (intermediate)"
    else:
        sig_cr_johnson = Pcr_euler / A_col
        Pcr_johnson    = Pcr_euler
        regime = "Euler (long column)"

    sig_cr = sig_cr_johnson
    Pcr    = Pcr_johnson

    # secant formula and bisection for yield load
    def secant_stress(P, A, e, c, r, Le, E_mod):
        if P <= 0 or A <= 0:
            return 0.0
        arg = (Le / (2*r)) * np.sqrt(P / (E_mod * A))
        arg = min(arg, np.pi/2 - 1e-6)
        return (P/A) * (1.0 + (e * c / r**2) * (1.0 / np.cos(arg)))

    def secant_yield_load(A, e, c, r, Le, E_mod, Fy_val, Pcr_lim):
        """Find load P_y at which secant stress first reaches Fy (bisection)."""
        if e == 0:
            return Fy_val * A    # concentric: yield load = Fy * A
        lo, hi = 1.0, Pcr_lim * 0.9999
        for _ in range(80):
            mid = (lo + hi) / 2
            if secant_stress(mid, A, e, c, r, Le, E_mod) >= Fy_val:
                hi = mid
            else:
                lo = mid
        return (lo + hi) / 2

    def midspan_deflection(P, e, Pcr_e):
        """δ_mid = e·[sec(π/2·√(P/Pcr_e)) − 1]  (pinned-pinned equivalent)."""
        if P <= 0 or Pcr_e <= 0 or e == 0:
            return 0.0
        ratio = min(P / Pcr_e, 0.9999)
        return e * (1.0 / np.cos(np.pi/2 * np.sqrt(ratio)) - 1.0)

    # eccentric results
    if eccentric and ecc > 0:
        sig_max_applied = secant_stress(
            P_applied if P_applied > 0 else 1e-6,
            A_col, ecc, c_col, r_min, Le_col, E)
        P_yield = secant_yield_load(A_col, ecc, c_col, r_min, Le_col, E, Fy, Pcr_euler)
        delta_mid_m = midspan_deflection(
            P_applied if P_applied > 0 else 0.0, ecc, Pcr_euler)
        delta_mid_mm = delta_mid_m * 1e3
        ecc_ratio = ecc * c_col / r_min**2     # eccentricity ratio  ec/r²
        FOS_col   = P_yield / P_applied if P_applied > 0 else float("inf")
        col_status = "PASS" if (P_applied == 0 or FOS_col >= 1.0) else "FAIL"
        stress_ok  = (sig_max_applied <= Fy) if P_applied > 0 else True
    else:
        sig_max_applied = (P_applied / A_col) if P_applied > 0 else 0.0
        P_yield   = Fy * A_col
        delta_mid_mm = 0.0
        ecc_ratio = 0.0
        FOS_col   = Pcr / P_applied if P_applied > 0 else float("inf")
        col_status = "PASS" if (P_applied == 0 or FOS_col >= 1.0) else "FAIL"
        stress_ok  = True

    st.markdown("#### Results")
    if eccentric and ecc > 0:
        m1,m2,m3,m4,m5 = st.columns(5)
        Py_d, lPy   = dsp(P_yield,  "F", U)
        sm_d, lsm   = dsp(sig_max_applied, "s", U)
        dd_d, ldd_c = dsp(delta_mid_mm*1e-3, "d", U)
        m1.metric("Euler Pcr",             f"{dsp(Pcr_euler,'F',U)[0]:.3f} {lPy}")
        m2.metric("Yield Load  Py",        f"{Py_d:.3f} {lPy}")
        m3.metric("σ_max  (secant)",       f"{sm_d:.2f} {lsm}")
        m4.metric("Midspan Deflection δ",  f"{dd_d:.3f} {ldd_c}")
        m5.metric("FOS  (vs Py)",
                  f"{FOS_col:.2f}" if P_applied > 0 else "—",
                  delta="PASS" if col_status=="PASS" else "FAIL",
                  delta_color="normal" if col_status=="PASS" else "inverse")
    else:
        m1,m2,m3,m4 = st.columns(4)
        Pcr_d, lPcr = dsp(Pcr,   "F", U)
        scr_d, lscr = dsp(sig_cr,"s", U)
        m1.metric("Critical Load  Pcr",      f"{Pcr_d:.3f} {lPcr}")
        m2.metric("Critical Stress  σcr",    f"{scr_d:.3f} {lscr}")
        m3.metric("Slenderness Ratio  KL/r", f"{SR:.1f}")
        m4.metric("FOS  (vs Pcr)",
                  f"{FOS_col:.2f}" if P_applied > 0 else "—",
                  delta="PASS" if col_status=="PASS" else "FAIL",
                  delta_color="normal" if col_status=="PASS" else "inverse")

    # Status banner
    if P_applied == 0:
        st.info(f"ℹ️  No applied load entered. Regime: **{regime}**  ·  Cc = {Cc:.1f}"
                + (f"  ·  ec/r² = {ecc_ratio:.3f}" if eccentric and ecc>0 else ""))
    elif col_status == "PASS":
        st.success(f"✅  Column OK — FOS = {FOS_col:.2f}   [{regime}]")
    else:
        st.error(f"❌  Applied load exceeds capacity — column will fail!   [{regime}]")

    st.markdown("---")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**Column Schematic**")
        fig_bk, ax_bk = plt.subplots(figsize=(3, 5.2), facecolor=BG)
        ax_bk.set_facecolor(CARD)
        for spine in ax_bk.spines.values(): spine.set_color(GRID)
        ax_bk.tick_params(colors="#888", labelsize=8)
        ax_bk.set_xlim(-2.0, 2.0); ax_bk.set_ylim(-0.07*Lc, 1.12*Lc)
        ax_bk.set_xticks([]); ax_bk.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v*dsp(1.0,'L',U)[0]:.1f}"))
        ax_bk.set_ylabel(f"Height  ({lLc})", color="#888", fontsize=8)

        y_col = np.linspace(0, Lc, 300)

        # Buckled / deflected shape
        if eccentric and ecc > 0 and P_applied > 0:
            # Elastic deflection under eccentric load (pinned-pinned form)
            ratio_P = min(P_applied / Pcr_euler, 0.9999) if Pcr_euler > 0 else 0
            phi = np.pi/2 * np.sqrt(ratio_P)
            # v(y) = e * [sec(φ)·cos(π*y/Le - φ) + tan(φ)·sin(π*y/Le) − cos(π*y/Le) − 1]
            # Simplified sinusoidal approximation for display:
            k_arg = np.sqrt(P_applied / (E * I_min)) if (E*I_min) > 0 else 0
            try:
                delta_shape = (ecc / np.cos(k_arg * Le_col / 2)) * np.cos(
                    k_arg * (y_col - Le_col/2)) - ecc
            except ZeroDivisionError:
                delta_shape = np.zeros_like(y_col)
            # Scale for display; max normalized to 1.0
            max_d = np.max(np.abs(delta_shape))
            delta_norm = delta_shape / max_d if max_d > 0 else delta_shape
            shape_color = RED if col_status == "FAIL" else ACC
            label_shape = "Deflected (eccentric)"
        else:
            # Pure buckling mode shape
            if K == 2.0:
                delta_norm = np.sin(np.pi*y_col / (2*Le_col))
            elif K == 0.5:
                delta_norm = np.sin(2*np.pi*y_col / Le_col)
            else:
                delta_norm = np.sin(np.pi*y_col / Le_col)
            shape_color = ACC
            label_shape = "Buckled mode shape"

        ax_bk.plot([0,0],[0,Lc], color="#333", lw=1.5, ls="dashed", label="Centroidal axis")
        ax_bk.plot(delta_norm, y_col, color=shape_color, lw=2.5, label=label_shape)
        ax_bk.fill_betweenx(y_col, 0, delta_norm, alpha=0.13, color=shape_color)

        # Eccentricity offset indicator at top and bottom
        if eccentric and ecc > 0:
            ecc_norm = ecc / (sp["c"] * 2) * 0.8   # scale relative to section depth
            for y_pos in [0.0, Lc]:
                ax_bk.annotate("", xy=(ecc_norm, y_pos),
                    xytext=(0, y_pos),
                    arrowprops=dict(arrowstyle="<->", color=ORG, lw=1.2))
            ax_bk.text(ecc_norm/2, Lc*0.03, f"e={ecc_mm:.1f}mm",
                       color=ORG, fontsize=7, fontfamily="monospace", ha="center")

        # End condition symbols
        def draw_pin(axx, y_pos, col_):
            size = Lc*0.04
            axx.add_patch(plt.Polygon(
                [[0,y_pos],[-size,y_pos-1.5*size],[size,y_pos-1.5*size]],
                facecolor="#3a4060", edgecolor=col_, lw=1.2))
            axx.plot([-2*size,2*size],[y_pos-1.5*size,y_pos-1.5*size], color=col_, lw=1.5)

        def draw_fixed(axx, y_pos, col_):
            size = Lc*0.04
            axx.add_patch(patches.Rectangle(
                (-3*size, y_pos-size), 6*size, size,
                facecolor="#1e2640", edgecolor=col_, lw=1.5))
            for hx in np.linspace(-2*size, 2*size, 5):
                sign = 1 if y_pos == 0 else -1
                axx.plot([hx, hx+0.5*size*sign],[y_pos-size, y_pos-1.5*size*sign],
                         color="#444", lw=0.8)

        end_labels = {
            "Pinned – Pinned   (K = 1.0)":  ("pin","pin"),
            "Fixed – Free      (K = 2.0)":  ("fixed","free"),
            "Fixed – Pinned    (K = 0.699)": ("fixed","pin"),
            "Fixed – Fixed     (K = 0.5)":  ("fixed","fixed"),
        }
        bot_end, top_end = end_labels[end_cond]
        if bot_end == "pin":  draw_pin(ax_bk, 0, ACC)
        else:                 draw_fixed(ax_bk, 0, ACC)
        if top_end == "pin":  draw_pin(ax_bk, Lc, BLU)
        elif top_end == "free":
            ax_bk.plot([-0.15,0.15],[Lc,Lc], color=BLU, lw=1.5, ls="dashed")
        else:                 draw_fixed(ax_bk, Lc, BLU)

        # Applied load arrow — offset if eccentric
        load_x = (ecc / (sp["c"]*2) * 0.8) if (eccentric and ecc > 0) else 0.0
        ax_bk.annotate("", xy=(load_x, Lc), xytext=(load_x, Lc+0.11*Lc),
            arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.8, mutation_scale=11))
        if P_applied > 0:
            Pad, lPad = dsp(P_applied,"F",U)
            ax_bk.text(load_x, Lc+0.13*Lc, f"{Pad:.2g} {lPad}",
                       color=RED, fontsize=7, fontfamily="monospace", ha="center")

        ax_bk.set_title(f"K = {K}   Le = {Le_col*dsp(1.0,'L',U)[0]:.2f} {lLc}",
                        color="#aaa", fontsize=8.5, fontfamily="monospace")
        fig_bk.tight_layout()
        st.pyplot(fig_bk, use_container_width=True); plt.close(fig_bk)

    with col_right:
        if eccentric and ecc > 0:
            st.markdown("**Secant Formula  —  P vs σ_max**")

            P_arr = np.linspace(0.001, Pcr_euler * 0.999, 400)
            sig_arr = np.array([secant_stress(p, A_col, ecc, c_col, r_min, Le_col, E)
                                for p in P_arr])
            # Also plot for e = 0 (concentric)
            sig_conc = P_arr / A_col

            # Eccentricity ratio curves for reference (ec/r² = 0.1, 0.3, 0.6)
            ref_ratios = [0.1, 0.3, 0.6]
            ref_colors = ["#3a4a6a","#4a5a7a","#5a6a8a"]

            fig_sc = go.Figure()
            for rr, rc in zip(ref_ratios, ref_colors):
                e_ref = rr * r_min**2 / c_col
                sig_ref = np.array([secant_stress(p, A_col, e_ref, c_col, r_min, Le_col, E)
                                    for p in P_arr])
                fig_sc.add_trace(go.Scatter(
                    x=sig_ref*1e-6, y=[dsp(p,"F",U)[0] for p in P_arr],
                    mode="lines", line=dict(color=rc, width=1, dash="dot"),
                    name=f"ec/r² = {rr}", opacity=0.7))

            fig_sc.add_trace(go.Scatter(
                x=sig_conc*1e-6, y=[dsp(p,"F",U)[0] for p in P_arr],
                mode="lines", line=dict(color="#555", width=1.5, dash="dash"),
                name="Concentric (e = 0)"))
            fig_sc.add_trace(go.Scatter(
                x=sig_arr*1e-6, y=[dsp(p,"F",U)[0] for p in P_arr],
                mode="lines", line=dict(color=ACC, width=2.5),
                name=f"This column  (ec/r² = {ecc_ratio:.3f})"))

            # Fy vertical line
            fig_sc.add_vline(x=Fy*1e-6, line_color=ORG, line_dash="dot",
                             annotation_text=f"Fy = {Fy*1e-6:.0f} MPa",
                             annotation_font_color=ORG)
            # Pcr horizontal line
            fig_sc.add_hline(y=dsp(Pcr_euler,"F",U)[0],
                             line_color=BLU, line_dash="dot",
                             annotation_text=f"Pcr (Euler)",
                             annotation_font_color=BLU,
                             annotation_position="bottom right")
            # Current operating point
            if P_applied > 0:
                fig_sc.add_trace(go.Scatter(
                    x=[sig_max_applied*1e-6],
                    y=[dsp(P_applied,"F",U)[0]],
                    mode="markers",
                    marker=dict(
                        color=RED if col_status=="FAIL" else ACC,
                        size=13, symbol="circle",
                        line=dict(color="white", width=1.5)),
                    name="Applied load"))

            _, lFplot = dsp(1.0,"F",U)
            fig_sc.update_layout(
                paper_bgcolor="#0b0d14", plot_bgcolor="#13151e",
                font=dict(color="#888"),
                xaxis=dict(title="Max Compressive Stress  σ_max  (MPa)",
                           color="#888", gridcolor=GRID),
                yaxis=dict(title=f"Axial Load  P  ({lFplot})",
                           color="#888", gridcolor=GRID),
                legend=dict(bgcolor="#13151e", bordercolor=GRID, font=dict(size=10)),
                height=400, margin=dict(l=60,r=20,t=20,b=55),
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        else:
            st.markdown("**Column Curve  (σcr vs KL/r)**")
            SR_arr = np.linspace(1, max(SR*1.6, Cc*1.4, 200), 500)
            sig_johnson_arr = np.where(
                SR_arr <= Cc,
                Fy*(1 - SR_arr**2/(2*Cc**2)),
                np.pi**2*E/SR_arr**2)
            sig_euler_arr = np.pi**2*E/SR_arr**2

            fig_cc = go.Figure()
            fig_cc.add_trace(go.Scatter(
                x=SR_arr, y=sig_euler_arr*1e-6,
                mode="lines", line=dict(color="#555",width=1.5,dash="dash"),
                name="Euler (full range)"))
            fig_cc.add_trace(go.Scatter(
                x=SR_arr, y=sig_johnson_arr*1e-6,
                mode="lines", line=dict(color=ACC,width=2.5),
                name="Johnson–Euler (governing)"))
            fig_cc.add_hline(y=Fy*1e-6, line_color=ORG, line_dash="dot",
                             annotation_text=f"Fy = {Fy*1e-6:.0f} MPa",
                             annotation_font_color=ORG, annotation_position="bottom right")
            fig_cc.add_vline(x=Cc, line_color=BLU, line_dash="dot",
                             annotation_text=f"Cc = {Cc:.1f}",
                             annotation_font_color=BLU)
            fig_cc.add_trace(go.Scatter(
                x=[SR], y=[sig_cr*1e-6], mode="markers",
                marker=dict(color=RED if col_status=="FAIL" else ACC,
                            size=12, symbol="circle",
                            line=dict(color="white",width=1.5)),
                name=f"This column  (KL/r = {SR:.1f})"))
            if P_applied > 0:
                sig_app = P_applied / A_col
                fig_cc.add_hline(y=sig_app*1e-6, line_color=RED, line_dash="dot",
                                 annotation_text=f"σ_applied = {sig_app*1e-6:.1f} MPa",
                                 annotation_font_color=RED)
            fig_cc.update_layout(
                paper_bgcolor="#0b0d14", plot_bgcolor="#13151e",
                font=dict(color="#888"),
                xaxis=dict(title="Slenderness Ratio  KL/r", color="#888", gridcolor=GRID),
                yaxis=dict(title="Critical Stress  σcr  (MPa)", color="#888", gridcolor=GRID),
                legend=dict(bgcolor="#13151e", bordercolor=GRID, font=dict(size=10)),
                height=400, margin=dict(l=60,r=20,t=20,b=55),
            )
            st.plotly_chart(fig_cc, use_container_width=True)

    st.markdown("#### Detailed Summary")
    Pcrd,  lPcrd  = dsp(Pcr_euler, "F", U)
    scrd,  lscrd  = dsp(sig_cr,    "s", U)
    syd,   lsyd   = dsp(Fy,        "s", U)
    Led,   lLed   = dsp(Le_col,    "L", U)
    Imin_d,lImin  = dsp(I_min,     "I", U)

    detail_rows = [
        ("Effective Length Factor K",     f"{K}"),
        ("Effective Length  Le = KL",     f"{Led:.3f} {lLed}"),
        ("Min. Radius of Gyration  r",    f"{r_min*1e3:.3f} mm"),
        ("Slenderness Ratio  KL/r",       f"{SR:.2f}"),
        ("Transition Slenderness  Cc",    f"{Cc:.2f}"),
        ("Column Regime",                 regime),
        ("Min. Moment of Inertia  I_min", f"{Imin_d:.3f} {lImin}"),
        ("Euler Critical Load  Pcr,E",    f"{dsp(Pcr_euler,'F',U)[0]:.4f} {lPcrd}"),
        ("Governing Pcr  (Johnson/Euler)",f"{dsp(Pcr,'F',U)[0]:.4f} {lPcrd}"),
        ("Critical Stress  σcr",          f"{scrd:.3f} {lscrd}"),
        ("Yield Stress  Fy",              f"{syd:.1f} {lsyd}"),
    ]

    if eccentric and ecc > 0:
        sm_d2, lsm2 = dsp(sig_max_applied, "s", U)
        Py_d2, lPy2 = dsp(P_yield,         "F", U)
        dd_d2, ldd2 = dsp(delta_mid_mm*1e-3,"d", U)
        detail_rows += [
            ("— Eccentric Loading ——————", ""),
            ("Eccentricity  e",            f"{ecc_mm:.3f} mm"),
            ("Eccentricity Ratio  ec/r²",  f"{ecc_ratio:.4f}"),
            ("Max Stress  σ_max  (secant)",f"{sm_d2:.3f} {lsm2}"),
            ("Yield Load  Py",             f"{Py_d2:.4f} {lPy2}"),
            ("Midspan Deflection  δ",      f"{dd_d2:.4f} {ldd2}" if P_applied>0 else "—"),
        ]

    if P_applied > 0:
        detail_rows += [
            ("Applied Load  P",  f"{dsp(P_applied,'F',U)[0]:.4f} {lPcrd}"),
            ("Factor of Safety", f"{FOS_col:.3f}"),
            ("Status",           col_status),
        ]

    df_col = pd.DataFrame(detail_rows, columns=["Parameter","Value"])
    st.dataframe(df_col, use_container_width=True, hide_index=True)


# report tab
with tab_rep:
    ca,cb = st.columns(2)

    with ca:
        st.markdown("### 📄 PDF Report")
        st.caption("Two-page PDF: project summary with checks + full diagrams.")
        if st.button("Generate PDF Report"):
            with st.spinner("Building report…"):
                pdf_buf = make_pdf(L,E,Fy,support,sec_type,sp,
                                   pt_loads_raw,dist_loads_raw,
                                   x,V,M,y_mm,reactions,checks,proj_name,U)
            st.download_button("⬇️  Download PDF", data=pdf_buf,
                file_name=f"{proj_name.replace(' ','_')}_report.pdf",
                mime="application/pdf")

    with cb:
        st.markdown("### 💾 Save / Load Config")
        if st.button("Save Configuration"):
            fll2,_ = dsp(1.0,"L",U); fvv2,_ = dsp(1.0,"F",U); fww2,_ = dsp(1.0,"w",U)
            cfg = {
                "project":  proj_name,
                "units":    unit_key,
                "material": mat_key,
                "E_GPa":    mat["E_GPa"],
                "Fy_MPa":   mat["Fy_MPa"],
                "support":  support,
                "L_display":L_disp,
                "sec_type": sec_type,
                "sec_params":sec_params,
                "pt_loads_display": [[xp*fll2, P*fvv2] for xp,P in pt_loads_raw],
                "dist_loads_display": [[x1*fll2,x2*fll2,w*fww2] for x1,x2,w in dist_loads_raw],
            }
            st.download_button("⬇️  Download JSON", data=json.dumps(cfg,indent=2),
                file_name=f"{proj_name.replace(' ','_')}_config.json",
                mime="application/json")

        st.markdown("---")
        uploaded = st.file_uploader("Load configuration (.json)", type="json")
        if uploaded:
            cfg = json.load(uploaded)
            st.success(f"Loaded: **{cfg.get('project','?')}** — {cfg.get('support','?')}, L={cfg.get('L_display','?')}")
            st.info("Re-enter the values shown below into the sidebar to restore the configuration.")
            with st.expander("Configuration data"):
                st.json(cfg)

st.markdown("""
<br>
<p style='color:#2a2d3a; font-size:0.72rem; font-family:monospace; text-align:center;'>
Beam Analysis Suite  ·  Euler-Bernoulli FEM (300 elements)
·  EI·y′′ = M(x)  ·  4 support conditions  ·  SI & Imperial
</p>
""", unsafe_allow_html=True)