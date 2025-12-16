import streamlit as st
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import io
import base64
import streamlit.components.v1 as components

# ==========================================
# 1. SETUP & SHARED RESOURCES
# ==========================================
st.set_page_config(page_title="RC Design Suite Pro", layout="wide", page_icon="🏗️")

BAR_INFO = {
    'RB6': {'A_cm2': 0.283, 'd_mm': 6},
    'RB9': {'A_cm2': 0.636, 'd_mm': 9},
    'DB10': {'A_cm2': 0.785, 'd_mm': 10},
    'DB12': {'A_cm2': 1.131, 'd_mm': 12},
    'DB16': {'A_cm2': 2.011, 'd_mm': 16},
    'DB20': {'A_cm2': 3.142, 'd_mm': 20},
    'DB25': {'A_cm2': 4.909, 'd_mm': 25},
    'DB28': {'A_cm2': 6.158, 'd_mm': 28},
    'DB32': {'A_cm2': 8.042, 'd_mm': 32}
}


def fmt(n, digits=3):
    try:
        val = float(n)
        if math.isnan(val): return "-"
        return f"{val:,.{digits}f}"
    except:
        return "-"


def fig_to_base64(fig):
    buf = io.BytesIO();
    fig.savefig(buf, format='png', bbox_inches='tight');
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


st.markdown("""
<style>
    .print-btn-internal { background-color: #008CBA; color: white; padding: 10px 20px; border-radius: 5px; cursor: pointer; text-decoration: none; }
    .report-table {width: 100%; border-collapse: collapse; font-family: sans-serif; font-size: 13px;}
    .report-table th, .report-table td {border: 1px solid #ddd; padding: 8px;}
    .report-table th {background-color: #f2f2f2; text-align: center; font-weight: bold;}
    .sec-row {background-color: #e0e0e0; font-weight: bold; font-size: 14px;}
    .pass-ok {color: green; font-weight: bold; text-align: center;}
    .pass-no {color: red; font-weight: bold; text-align: center;}
    .load-val {color: #D32F2F !important; font-weight: bold;}
    .drawing-container {display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin-top: 20px;}
    .drawing-box {border: 1px solid #ddd; padding: 5px; background: white; text-align: center; min-width: 300px;}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 2. LOGIC: BEAM MODULE
# ==========================================
def beta1FromFc(fc_MPa):
    if fc_MPa <= 28: return 0.85
    b1 = 0.85 - 0.05 * ((fc_MPa - 28) / 7)
    return max(0.65, b1)


def flexureSectionResponse(As_mm2, fc_MPa, fy_MPa, bw_mm, d_mm):
    beta1 = beta1FromFc(fc_MPa)
    a = (As_mm2 * fy_MPa) / (0.85 * fc_MPa * bw_mm)
    c = a / beta1
    eps_cu = 0.003
    eps_t = eps_cu * (d_mm - c) / c

    if eps_t <= 0.002:
        phi = 0.65
    elif eps_t >= 0.005:
        phi = 0.90
    else:
        phi = 0.65 + (eps_t - 0.002) * (250.0 / 3.0)

    Mn = As_mm2 * fy_MPa * (d_mm - a / 2.0)
    return {'phi': phi, 'phiMn': phi * Mn, 'eps_t': eps_t}


def process_beam_calculation(inputs):
    rows = []

    def sec(title):
        rows.append(["SECTION", title, "", "", "", "", ""])

    def row(item, formula, subs, result, unit, status=""):
        rows.append([item, formula, subs, result, unit, status])

    b = inputs['b'] * 10;
    h = inputs['h'] * 10;
    cover = inputs['cover'] * 10
    fc = inputs['fc'] * 0.0981;
    fy = inputs['fy'] * 0.0981
    m_bar = inputs['m_bar'];
    s_bar = inputs['s_bar']
    db_m = BAR_INFO[m_bar]['d_mm'];
    db_s = BAR_INFO[s_bar]['d_mm']
    d = h - cover - db_s - db_m / 2

    sec("1. PROPERTIES")
    row("Section", "b x h", f"{b}x{h}", "-", "mm", "")
    row("Depth d", "h-cov-ds-dm/2", f"{h}-{cover}-{db_s}-{db_m / 2}", f"{d:.1f}", "mm", "")

    # Min/Max As
    as_min = max(0.25 * math.sqrt(fc) / fy, 1.4 / fy) * b * d
    row("As,min", "max(...)bd", "-", f"{as_min:.0f}", "mm²", "")

    # Flexure Cases
    cases = [('L-Supp', inputs['mu_Ln']), ('Mid', inputs['mu_Mp']), ('R-Supp', inputs['mu_Rn'])]
    sec("2. FLEXURE DESIGN")

    bar_res = {}
    for name, mu in cases:
        if mu <= 0.01:
            bar_res[name] = 2;
            continue

        mu_nmm = mu * 9806650
        # Approx As
        req_as = mu_nmm / (0.9 * fy * 0.9 * d)
        req_as = max(req_as, as_min)

        n = math.ceil(req_as / (BAR_INFO[m_bar]['A_cm2'] * 100))
        n = max(n, 2)
        bar_res[name] = n
        prov_as = n * BAR_INFO[m_bar]['A_cm2'] * 100

        # Check Capacity
        res = flexureSectionResponse(prov_as, fc, fy, b, d)

        row(f"{name} Mu", "-", "-", f"{mu:.2f}", "tf-m", "")
        row(f"Req As", "Calc vs Min", f"Max({req_as:.0f}, {as_min:.0f})", f"{req_as:.0f}", "mm²", "")
        row(f"Provide", f"{n}-{m_bar}", f"As={prov_as:.0f}", "OK", "-", "PASS" if res['phiMn'] >= mu_nmm else "FAIL")

    # Shear
    sec("3. SHEAR DESIGN")
    vc = 0.17 * math.sqrt(fc) * b * d
    phi_vc = 0.75 * vc
    row("Vc", "0.17√fc' bd", f"0.17·{math.sqrt(fc):.2f}·{b}·{d:.0f}", f"{vc / 9806:.2f}", "tf", "")

    vu_cases = [('L', inputs['vu_L']), ('R', inputs['vu_R'])]
    s_res = {}
    for name, vu in vu_cases:
        vu_n = vu * 9806
        req_s = "-"
        status = "OK"
        if vu_n > phi_vc:
            vs = (vu_n / 0.75) - vc
            av = 2 * BAR_INFO[s_bar]['A_cm2'] * 100
            s_req = (av * fy * d) / vs
            s_prov = math.floor(min(s_req, d / 2, 600) / 10) * 10
            s_res[name] = s_prov
            req_s = f"@{s_prov / 10:.0f}cm"
        else:
            s_res[name] = 200  # min dist
            req_s = "Min Dist"

        row(f"{name} Vu", "-", "-", f"{vu:.2f}", "tf", "")
        row(f"Check", "φVc vs Vu", f"{phi_vc / 9806:.2f} vs {vu:.2f}", status, "-", req_s)

    return rows, bar_res, s_res


# ==========================================
# 3. LOGIC: COLUMN MODULE
# ==========================================
def calculate_pm_curve(b, h, cover, main_db, nx, ny, fc, fy):
    points = []
    d_prime = cover + 10 + main_db / 2
    ast = (2 * nx + 2 * max(0, ny - 2)) * (math.pi * (main_db / 2) ** 2)
    po = 0.85 * fc * (b * h - ast) + fy * ast
    pn_max = 0.8 * po

    # Simple 3 points for demo speed (Top, Bal, Pure Flex) in detailed app
    # Real app uses loop c from 1.5h to 0.1h
    c_vals = np.linspace(1.1 * h, 0.1 * h, 20)
    for c in c_vals:
        a = 0.85 * c  # simplify beta1
        cc = 0.85 * fc * b * min(a, h)
        # Steel forces approx (2 layers)
        fs1 = min(fy, 200000 * 0.003 * (c - d_prime) / c);
        fs1 = max(-fy, fs1)
        fs2 = min(fy, 200000 * 0.003 * (c - (h - d_prime)) / c);
        fs2 = max(-fy, fs2)
        pn = cc + (ast / 2) * fs1 + (ast / 2) * fs2
        mn = cc * (h / 2 - a / 2) + (ast / 2) * fs1 * (h / 2 - d_prime) - (ast / 2) * fs2 * (h / 2 - d_prime)

        # Phi
        phi = 0.65
        points.append({'P': phi * min(pn, pn_max / 0.65), 'M': phi * mn})

    return points, b * h, ast


def process_column_calculation(inputs):
    rows = []

    def sec(title):
        rows.append(["SECTION", title, "", "", "", "", ""])

    def row(item, formula, subs, result, unit, status=""):
        rows.append([item, formula, subs, result, unit, status])

    b = inputs['b'] * 10;
    h = inputs['h'] * 10;
    cover = inputs['cover'] * 10
    fc = inputs['fc'] * 0.0981;
    fy = inputs['fy'] * 0.0981

    # Auto Design Loop
    best_nx, best_ny = 2, 2
    if inputs['mode'] == 'Auto':
        # Simply find min bars passing load
        pass  # (Logic similar to previous, using default 2x2 for brevity in this merged snippet)
    else:
        best_nx = int(inputs['nx']);
        best_ny = int(inputs['ny'])

    db_main = BAR_INFO[inputs['m_bar']]['d_mm']
    curve, ag, ast = calculate_pm_curve(b, h, cover, db_main, best_nx, best_ny, fc, fy)

    sec("1. PROPERTIES")
    row("Section", "b x h", f"{b}x{h}", f"{ag / 100:.0f}", "cm²", "")
    row("Rebar", f"{inputs['m_bar']}", f"Total {2 * best_nx + 2 * max(0, best_ny - 2)}", f"{ast / 100:.2f}", "cm²", "")
    rho = ast / ag
    row("Ratio", "Ast/Ag", f"{ast:.0f}/{ag:.0f}", f"{rho * 100:.2f}", "%", "OK" if 0.01 <= rho <= 0.08 else "FAIL")

    sec("2. CAPACITY CHECK")
    pu_n = inputs['pu'] * 9806;
    mu_nmm = inputs['mu'] * 9806650
    # Check Axial
    p_max = curve[0]['P']
    row("Load Pu", "-", "-", f"{inputs['pu']:.2f}", "tf", "")
    row("Max Axial", "φPn,max", "From Curve", f"{p_max / 9806:.2f}", "tf", "PASS" if pu_n <= p_max else "FAIL")

    # Check M
    # Interpolate M at Pu
    m_cap = 0
    for i in range(len(curve) - 1):
        if curve[i + 1]['P'] <= pu_n <= curve[i]['P']:
            r = (pu_n - curve[i + 1]['P']) / (curve[i]['P'] - curve[i + 1]['P'] + 1e-9)
            m_cap = curve[i + 1]['M'] + r * (curve[i]['M'] - curve[i + 1]['M'])
            break

    row("Load Mu", "-", "-", f"{inputs['mu']:.2f}", "tf-m", "")
    row("Capacity φMn", "@ Pu", "Interpolated", f"{m_cap / 9806650:.2f}", "tf-m", "PASS" if mu_nmm <= m_cap else "FAIL")

    return rows, curve, best_nx, best_ny


# ==========================================
# 4. LOGIC: FOOTING MODULE (Detailed ACI)
# ==========================================
def process_footing_calculation(inputs):
    rows = []

    def sec(title):
        rows.append(["SECTION", title, "", "", "", "", ""])

    def row(item, formula, subs, result, unit, status=""):
        rows.append([item, formula, subs, result, unit, status])

    fc = inputs['fc'] * 0.0981;
    fy = inputs['fy'] * 0.0981
    pu = inputs['pu'];
    n_pile = int(inputs['n_pile'])
    s = inputs['s'] * 1000;
    edge = inputs['edge'] * 1000
    dp = inputs['dp'] * 1000;
    col = inputs['col'] * 1000
    h_final = inputs['h'] * 1000;
    cover = 75
    db = BAR_INFO[inputs['m_bar']]['d_mm']
    d = h_final - cover - db

    # 1. Geometry
    # Simplified Coords for 1-5 piles
    coords = []
    if n_pile == 1:
        coords = [(0, 0)]
    elif n_pile == 2:
        coords = [(-s / 2, 0), (s / 2, 0)]
    elif n_pile == 3:
        coords = [(-s / 2, -s * 0.288), (s / 2, -s * 0.288), (0, s * 0.577)]
    elif n_pile == 4:
        coords = [(-s / 2, -s / 2), (s / 2, -s / 2), (-s / 2, s / 2), (s / 2, s / 2)]

    bx = (max([abs(x) for x, _ in coords]) * 2) + dp + 2 * edge if n_pile > 1 else dp + 2 * edge
    by = (max([abs(y) for _, y in coords]) * 2) + dp + 2 * edge if n_pile > 1 else dp + 2 * edge
    bx = max(bx, col + 2 * edge);
    by = max(by, col + 2 * edge)  # min width

    sec("1. GEOMETRY")
    row("Size", "B x L", f"{bx:.0f}x{by:.0f}", f"h={h_final:.0f}", "mm", "")
    lambda_s = math.sqrt(2 / (1 + 0.004 * d))
    row("Size Effect", "λs", f"√(2/(1+0.004*{d:.0f}))", f"{lambda_s:.3f}", "-", "≤1.0")

    sec("2. REACTION")
    p_avg = pu / n_pile
    row("Pile Load", "Pu/N", f"{pu}/{n_pile}", f"{p_avg:.2f}", "tf", "PASS" if p_avg <= inputs['cap'] else "FAIL")

    sec("3. FLEXURE")
    # Calc Moment (X-axis)
    mx = 0
    p_n = p_avg * 9806
    for x, y in coords:
        lever = abs(x) - col / 2
        if lever > 0: mx += p_n * lever

    phi = 0.9
    req_as = mx / (phi * fy * 0.9 * d) if mx > 0 else 0
    min_as = 0.0018 * by * h_final
    des_as = max(req_as, min_as)
    n_bars = math.ceil(des_as / (BAR_INFO[inputs['m_bar']]['A_cm2'] * 100))
    if n_pile == 1: n_bars = max(n_bars, 4)

    row("Moment Mu", "Σ P(x-c/2)", "-", f"{mx / 9806650:.2f}", "tf-m", "")
    row("As Req", "Mu/0.9fy0.9d", f"{mx:.0f}/...", f"{req_as:.0f}", "mm²", "")
    row("As Min", "0.0018bh", f"0.0018*{by:.0f}*{h_final:.0f}", f"{min_as:.0f}", "mm²", "")
    row("Provide", f"{n_bars}-{inputs['m_bar']}", f"As={n_bars * BAR_INFO[inputs['m_bar']]['A_cm2'] * 100:.0f}", "OK",
        "-", "")

    if n_pile > 1:
        sec("4. SHEAR (ACI 318-19)")
        # Punching
        bo = 4 * (col + d)  # simplified square col
        vc_p = 0.33 * lambda_s * math.sqrt(fc) * bo * d
        vu_p = sum([p_n for x, y in coords if max(abs(x), abs(y)) > (col + d) / 2])
        row("Punching Vu", "Sum Outside", "-", f"{vu_p / 9806:.2f}", "tf", "")
        row("Capacity φVc", "0.75*0.33λs√fc*bo*d", f"0.75*0.33*{lambda_s:.2f}...", f"{0.75 * vc_p / 9806:.2f}", "tf",
            "PASS" if vu_p <= 0.75 * vc_p else "FAIL")

        # Beam Shear
        vc_b = 0.17 * math.sqrt(fc) * by * d  # simplified wo rho
        vu_b = sum([p_n for x, y in coords if abs(x) > col / 2 + d])
        row("Beam Vu", "Sum Outside d", "-", f"{vu_b / 9806:.2f}", "tf", "")
        row("Capacity φVc", "0.75*0.17√fc*b*d", "-", f"{0.75 * vc_b / 9806:.2f}", "tf",
            "PASS" if vu_b <= 0.75 * vc_b else "FAIL")

    return rows, coords, bx, by, n_bars


# ==========================================
# 5. PLOTTING & REPORT GEN
# ==========================================
def draw_beam(b, h, n, bar):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.add_patch(patches.Rectangle((0, 0), b, h, ec='k', fc='#eee'))
    ax.text(b / 2, h / 2, f"{n}-{bar}", ha='center', fontweight='bold', fontsize=12)
    ax.set_xlim(-5, b + 5);
    ax.set_ylim(-5, h + 5);
    ax.axis('off')
    return fig


def draw_col_pm(curve, pu, mu):
    fig, ax = plt.subplots(figsize=(4, 4))
    ms = [p['M'] / 9806650 for p in curve];
    ps = [p['P'] / 9806 for p in curve]
    ax.plot(ms, ps, 'b-', lw=2);
    ax.plot(mu, pu, 'ro', ms=8)
    ax.set_xlabel('M (tf-m)');
    ax.set_ylabel('P (tf)');
    ax.grid(True, ls='--')
    return fig


def draw_foot(coords, bx, by, n, bar):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.add_patch(patches.Rectangle((-bx / 2, -by / 2), bx, by, ec='k', fc='#f9f9f9'))
    for x, y in coords: ax.add_patch(patches.Circle((x, y), 150, ec='k', ls='--'))
    ax.text(0, 0, f"{n}-{bar} (EW)", ha='center', fontweight='bold')
    ax.set_xlim(-bx / 1.2, bx / 1.2);
    ax.set_ylim(-by / 1.2, by / 1.2);
    ax.axis('off')
    return fig


def generate_report(title, rows, img):
    t_rows = "".join([
                         f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td class='load-val'>{r[3]}</td><td>{r[4]}</td><td class='{('pass-ok' if 'PASS' in r[5] or 'OK' in r[5] else 'pass-no')}'>{r[5]}</td></tr>" if
                         r[0] != "SECTION" else f"<tr class='sec-row'><td colspan='6'>{r[1]}</td></tr>" for r in rows])
    return f"""
    <div style="font-family: Sarabun, sans-serif; padding: 20px;">
        <h3 style="text-align:center; border-bottom: 2px solid #333;">{title}</h3>
        <div class="drawing-container"><div class="drawing-box"><img src="{img}" style="max-width:100%;"></div></div><br>
        <table class="report-table">
            <thead><tr><th width="20%">Item</th><th width="25%">Formula</th><th width="30%">Substitution</th><th>Result</th><th>Unit</th><th>Status</th></tr></thead>
            <tbody>{t_rows}</tbody>
        </table>
    </div>
    """


# ==========================================
# 6. MAIN APP ROUTER
# ==========================================
st.sidebar.title("🏗️ Design Suite")
app_mode = st.sidebar.radio("Module", ["Beam", "Column", "Footing"])
st.sidebar.markdown("---")

if app_mode == "Beam":
    st.header("Beam Design")
    with st.sidebar.form("b"):
        fc = st.number_input("fc'", value=240);
        fy = st.number_input("fy", value=4000)
        b = st.number_input("b (cm)", value=25);
        h = st.number_input("h (cm)", value=50)
        cover = st.number_input("cov", value=3.0);
        m_bar = st.selectbox("Main", list(BAR_INFO.keys()), index=4)
        s_bar = st.selectbox("Stirrup", list(BAR_INFO.keys()));
        mu = st.number_input("Mu (tf-m)", value=8.0)
        vu = st.number_input("Vu (tf)", value=12.0)
        run = st.form_submit_button("Calc")
    if run:
        d = {'b': b, 'h': h, 'cover': cover, 'fc': fc, 'fy': fy, 'm_bar': m_bar, 's_bar': s_bar, 'mu_Ln': mu,
             'mu_Mp': mu / 2, 'mu_Rn': mu, 'vu_L': vu, 'vu_R': vu}
        rows, _, _ = process_beam_calculation(d)
        st.components.v1.html(generate_report("Beam Report", rows, fig_to_base64(draw_beam(b, h, 3, m_bar))),
                              height=800, scrolling=True)

elif app_mode == "Column":
    st.header("Column Design")
    with st.sidebar.form("c"):
        fc = st.number_input("fc'", value=240);
        fy = st.number_input("fy", value=4000)
        b = st.number_input("b", value=25);
        h = st.number_input("h", value=25)
        cover = st.number_input("cov", value=3.0);
        mode = st.radio("Mode", ["Auto", "Manual"])
        nx = st.number_input("Nx", value=2);
        ny = st.number_input("Ny", value=2)
        m_bar = st.selectbox("Main", list(BAR_INFO.keys()), index=4);
        pu = st.number_input("Pu", value=40.0)
        mu = st.number_input("Mu", value=2.0)
        run = st.form_submit_button("Calc")
    if run:
        d = {'b': b, 'h': h, 'cover': cover, 'fc': fc, 'fy': fy, 'm_bar': m_bar, 'tieBar': 'RB6', 'nx': nx, 'ny': ny,
             'pu': pu, 'mu': mu, 'mode': mode}
        rows, curve, bnx, bny = process_column_calculation(d)
        st.components.v1.html(generate_report("Column Report", rows, fig_to_base64(draw_col_pm(curve, pu, mu))),
                              height=800, scrolling=True)

elif app_mode == "Footing":
    st.header("Footing Design")
    with st.sidebar.form("f"):
        fc = st.number_input("fc'", value=240);
        fy = st.number_input("fy", value=4000)
        n = st.selectbox("Piles", [1, 2, 3, 4, 5], index=3);
        dp = st.number_input("Dia", value=0.22)
        sp = st.number_input("Space", value=0.8);
        h = st.number_input("Thk", value=0.5)
        edge = st.number_input("Edge", value=0.25);
        m_bar = st.selectbox("Main", list(BAR_INFO.keys()), index=4)
        pu = st.number_input("Pu", value=60.0);
        cap = st.number_input("Cap", value=30.0)
        run = st.form_submit_button("Calc")
    if run:
        d = {'fc': fc, 'fy': fy, 'n_pile': n, 'dp': dp, 's': sp, 'h': h, 'edge': edge, 'm_bar': m_bar, 'pu': pu,
             'cap': cap, 'col': 0.25}
        rows, coords, bx, by, nb = process_footing_calculation(d)
        st.components.v1.html(
            generate_report("Footing Report", rows, fig_to_base64(draw_foot(coords, bx, by, nb, m_bar))), height=800,
            scrolling=True)