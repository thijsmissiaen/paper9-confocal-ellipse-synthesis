#!/usr/bin/env python3
"""
Verification Suite for Paper IX:
Shifted Mellin Bridge Decomposition, Diagonal Stieltjes Tail Bounds,
and Off-Diagonal Obstructions for Euler-Sieved Approximations to Riemann's Xi-Function
========================================================================================
Author: Thijs Missiaen (2026)

High-precision (50-decimal mpmath) verification of:
  0. Canonical normalization:
     Xi(z) = 8 * int_0^inf Phi(u) cos(2 z u) du,
     W(d, t) = (1/2) d/dd |Xi(t - i d)|^2 = 16 * int_0^inf y sinh(2 d y) I(y, t) dy.
  1. Riemann-type shifted u-domain three-part bridge decomposition for Xi * E_X,
     exact incomplete-Gamma shell J_n(s), Gaussian bridge-tail envelope, and
     quantitative demonstration of the forward/reflected branch asymmetry
     |E_X(s) F_+(1 - s) - F_+^{E_X}(1 - s)| > 0 caused by E_X(s) != E_X(1 - s).
  2. Exact hyperbolic mode-ratio identity
       sinh^2(d omega_n / 2) / sinh^2(d omega_1 / 2)
         = n^{-d} * ((1 - (n / N_*)^d) / (1 - N_*^{-d}))^2 <= n^{-d},
     unconditional Rosser-Schoenfeld Stieltjes bound for the L^2 diagonal prime sum
     sum_{X < p <= N(t)} p^{-1 - 2d} on X >= sqrt(N(t)), Mertens' (1874)
     harmonic benchmark theta_* = (1/2) exp(-1/c_theta), and explicit demonstration
     that the true RS linear-amplitude L^1 sum (~ sum p^{-(1/2 + d)}) diverges as
     t -> infinity for every 0 < d < 1/2.
  3. Elementary hyperbolic-trigonometric identity
       cosh^2(a) cos^2(b) + sinh^2(a) sin^2(b) = cos^2(b) + sinh^2(a),
     decomposition |T(d, t)|^2 = D_E(d, t) + C_off(d, t) into positive L^2 diagonal
     plus oscillatory off-diagonal cross-terms, and numerical demonstration of strong
     destructive off-diagonal interference on d > 1/4 (consistent with Bohr density
     and Voronin universality).
"""

import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

mp.mp.dps = 50

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PUB_ROOT = os.path.dirname(SCRIPT_DIR)
FIG_DIR_ROOT = os.path.join(PUB_ROOT, "figures")
FIG_DIR_PAPER = os.path.join(PUB_ROOT, "paper", "figures")
os.makedirs(FIG_DIR_ROOT, exist_ok=True)
os.makedirs(FIG_DIR_PAPER, exist_ok=True)

C_THETA_RS = mp.mpf("1.000081")


def sieve_primes(limit_int: int):
    if limit_int < 2:
        return []
    sieve = bytearray(b"\x01") * (limit_int + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(math.isqrt(limit_int)) + 1):
        if sieve[p]:
            sieve[p * p : limit_int + 1 : p] = b"\x00" * (((limit_int - p * p) // p) + 1)
    return [i for i, v in enumerate(sieve) if v]


def compute_b_euler_sieve(max_n: int, x_cut: int):
    """Exact Dirichlet coefficients b_N^E(X) of zeta(s) * prod_{p <= X} (1 - p^{-s})."""
    primes_le_x = set(sieve_primes(min(x_cut, max_n)))
    smallest_p = [0] * (max_n + 1)
    for p in range(2, max_n + 1):
        if smallest_p[p] == 0:
            for mult in range(p, max_n + 1, p):
                if smallest_p[mult] == 0:
                    smallest_p[mult] = p
    b = [0] * (max_n + 1)
    if max_n >= 1:
        b[1] = 1
    for n in range(2, max_n + 1):
        m = n
        valid = True
        while m > 1:
            p = smallest_p[m]
            if p in primes_le_x:
                valid = False
                break
            while m % p == 0:
                m //= p
        b[n] = 1 if valid else 0
    return b


def phi_single_mode(u, n: int):
    n_mp = mp.mpf(n)
    e4u = mp.exp(4 * u)
    arg = mp.pi * (n_mp ** 2) * e4u
    if mp.re(arg) > 220:
        return mp.mpf(0)
    return (2 * (mp.pi ** 2) * (n_mp ** 4) * mp.exp(9 * u)
            - 3 * mp.pi * (n_mp ** 2) * mp.exp(5 * u)) * mp.exp(-arg)


def phi_kernel_mp(u, n_max: int = 45):
    total = mp.mpf(0)
    for n in range(1, n_max + 1):
        term = phi_single_mode(u, n)
        total += term
        if n > 3 and abs(term) < mp.mpf("1e-56"):
            break
    return total


def xi_completed_mp(s):
    return mp.mpf("0.5") * s * (s - 1) * mp.power(mp.pi, -s / 2) * mp.gamma(s / 2) * mp.zeta(s)


def euler_sieve_poly_mp(s, x_cut: int):
    prod = mp.mpc(1)
    for p in sieve_primes(x_cut):
        prod *= (1 - mp.power(p, -s))
    return prod


def j_shell_closed_form(s, n: int):
    """Exact incomplete-Gamma shell J_n(s) = 2 * int_0^inf phi_n(u) e^{2(s-1/2)u} du."""
    n_mp = mp.mpf(n)
    a = mp.pi * (n_mp ** 2)
    return (mp.mpf("0.5") * mp.power(mp.pi, -s / 2) * mp.power(n_mp, -s)
            * ((s - 1) * mp.gammainc(s / 2 + 1, a) + 2 * mp.power(a, s / 2 + 1) * mp.exp(-a)))


def verify_section0_canonical_wronskian():
    print("\n" + "=" * 80)
    print("SECTION 0: Canonical Xi(z) and Transverse Wronskian W(d,t) Normalization")
    print("=" * 80)

    xi_half = xi_completed_mp(mp.mpf("0.5"))
    int_phi = mp.quad(lambda u: phi_kernel_mp(u), [0, mp.mpf("2.8")])
    res_xi0 = abs(xi_half - 8 * int_phi)
    print(f"  Xi(0) = xi(1/2) = {mp.nstr(xi_half, 24)}, |Xi(0) - 8 int_0^inf Phi(u) du| = {mp.nstr(res_xi0, 4)}")
    assert res_xi0 < mp.mpf("1e-45")

    # Spectral trapezoidal grid in half-step u_k = k * h_half for exact 2D autocorrelation
    h_half = mp.mpf("2.6") / 160
    phi_cache = {k: phi_kernel_mp(k * h_half) for k in range(-324, 325)}
    step = 2 * h_half

    max_res_w = mp.mpf(0)
    for d_val, t_val in [(mp.mpf("0.15"), mp.mpf("14.5")), (mp.mpf("0.30"), mp.mpf("21.0"))]:
        eps = mp.mpf("1e-18")
        m_p = abs(xi_completed_mp(mp.mpf("0.5") + (d_val + eps) + 1j * t_val)) ** 2
        m_m = abs(xi_completed_mp(mp.mpf("0.5") + (d_val - eps) + 1j * t_val)) ** 2
        w_exact = (m_p - m_m) / (4 * eps)

        cos_cache = {m: mp.cos(2 * t_val * (m * step)) for m in range(-160, 161)}
        w_int = mp.mpf(0)
        for j in range(0, 161):
            wj = mp.mpf("0.5") if (j == 0 or j == 160) else mp.mpf("1.0")
            yj = j * step
            iy = mp.mpf(0)
            for m in range(-160, 161):
                wm = mp.mpf("0.5") if abs(m) == 160 else mp.mpf("1.0")
                iy += wm * phi_cache[j + m] * phi_cache[j - m] * cos_cache[m]
            iy *= step
            w_int += wj * yj * mp.sinh(2 * d_val * yj) * iy
        w_int *= 16 * step
        res = abs(w_exact - w_int)
        max_res_w = max(max_res_w, res)
        print(f"  (d={mp.nstr(d_val,3)}, t={mp.nstr(t_val,4)}): W_exact={mp.nstr(w_exact,14)}, "
              f"16*int={mp.nstr(w_int,14)}, |diff|={mp.nstr(res,4)}")
        assert res < mp.mpf("1e-35")

    return float(max_res_w)


def verify_section1_bridge_and_asymmetry():
    print("\n" + "=" * 80)
    print("SECTION 1: Shifted Mellin u-Domain Bridge Identity & Branch Asymmetry")
    print("=" * 80)

    test_s = mp.mpf("0.72") + 1j * mp.mpf("18.4")
    max_j_err = mp.mpf(0)
    for n in [1, 2, 3, 5, 7]:
        j_quad = 2 * mp.quad(
            lambda u: phi_single_mode(u, n) * mp.exp(2 * (test_s - mp.mpf("0.5")) * u),
            [0, mp.mpf("2.8")],
        )
        j_cf = j_shell_closed_form(test_s, n)
        max_j_err = max(max_j_err, abs(j_quad - j_cf))
    print(f"  [Check 1A] Max |J_n^quad(s) - J_n^closed(s)| = {mp.nstr(max_j_err, 5)}")
    assert max_j_err < mp.mpf("1e-42")

    primes_list = sieve_primes(11)
    test_cases = [
        (3, mp.mpf("0.15"), mp.mpf("14.134725")),
        (5, mp.mpf("0.18"), mp.mpf("14.134725")),
        (7, mp.mpf("0.30"), mp.mpf("25.010858")),
        (11, mp.mpf("0.22"), mp.mpf("30.424876")),
    ]

    bridge_rows = []
    for x_cut, d_val, t_val in test_cases:
        s = mp.mpf("0.5") + d_val + 1j * t_val
        lhs_exact = mp.mpf("0.5") * xi_completed_mp(s) * euler_sieve_poly_mp(s, x_cut)

        b_coeffs = compute_b_euler_sieve(60, x_cut)
        i_plus_main = mp.mpc(0)
        for n_idx in range(1, 35):
            if b_coeffs[n_idx] != 0:
                j_term = j_shell_closed_form(s, n_idx)
                i_plus_main += b_coeffs[n_idx] * j_term
                if n_idx > 5 and abs(j_term) < mp.mpf("1e-55"):
                    break

        f_plus_refl = mp.mpc(0)
        for m_idx in range(1, 35):
            j_term = j_shell_closed_form(1 - s, m_idx)
            f_plus_refl += j_term
            if m_idx > 5 and abs(j_term) < mp.mpf("1e-55"):
                break
        refl_term = euler_sieve_poly_mp(s, x_cut) * f_plus_refl

        # Exact bridge integral over non-trivial divisor D > 1 of prod_{p<=X}(1 - p^{-s})
        p_sub = [p for p in primes_list if p <= x_cut]
        divs = [(1, 1)]
        for p in p_sub:
            divs.extend([(d_prev * p, -mu_prev) for d_prev, mu_prev in list(divs)])

        b_bridge = mp.mpc(0)
        b_env = mp.mpf(0)
        for div_d, mu_d in divs:
            if div_d == 1:
                continue
            u_d = mp.mpf("0.5") * mp.log(div_d)
            pref = mu_d * mp.power(div_d, -mp.mpf("0.5"))

            def ig_s(w):
                return phi_kernel_mp(w) * mp.exp(2 * (s - mp.mpf("0.5")) * w)

            b_bridge += 2 * mu_d * mp.power(div_d, -s) * mp.quad(ig_s, [0, u_d])

            def ig_env(w):
                return phi_kernel_mp(w) * mp.exp(2 * d_val * w)

            b_env += 2 * mp.power(div_d, -(mp.mpf("0.5") + d_val)) * mp.quad(ig_env, [0, u_d])

        res_bridge = abs(lhs_exact - (i_plus_main + refl_term + b_bridge))
        ratio_env = abs(b_bridge) / b_env
        print(f"  [Check 1B] X={x_cut:2d}, d={mp.nstr(d_val,3)}, t={mp.nstr(t_val,6)}: "
              f"residual={mp.nstr(res_bridge,4)}, |B|/B_env={mp.nstr(ratio_env,5)}")
        assert res_bridge < mp.mpf("1e-42")
        assert abs(b_bridge) <= b_env * (1 + mp.mpf("1e-40"))
        bridge_rows.append((x_cut, float(d_val), float(t_val), float(abs(b_bridge)), float(b_env), float(res_bridge)))

    # Check 1C: Forward/Reflected Branch Asymmetry Obstruction
    # Compare true reflected term E_X(s) F_+(1-s) vs symmetric sieved F_+^{E_X}(1-s)
    print("  [Check 1C] Forward/Reflected Branch Asymmetry (E_X(s) != E_X(1-s)):")
    asym_rows = []
    for x_cut, d_val, t_val in [(5, mp.mpf("0.20"), mp.mpf("25.0")), (11, mp.mpf("0.25"), mp.mpf("40.0"))]:
        s = mp.mpf("0.5") + d_val + 1j * t_val
        b_coeffs = compute_b_euler_sieve(60, x_cut)
        f_plus_refl = sum(j_shell_closed_form(1 - s, m) for m in range(1, 25))
        true_refl = euler_sieve_poly_mp(s, x_cut) * f_plus_refl
        sym_sieved_refl = sum(b_coeffs[m] * j_shell_closed_form(1 - s, m) for m in range(1, 25) if b_coeffs[m] != 0)
        rel_mismatch = abs(true_refl - sym_sieved_refl) / abs(true_refl)
        print(f"    X={x_cut:2d}, (d,t)=({mp.nstr(d_val,3)},{mp.nstr(t_val,4)}): "
              f"|E_X(s)F_+(1-s) - F_+^E(1-s)| / |E_X(s)F_+(1-s)| = {mp.nstr(rel_mismatch, 6)}")
        assert rel_mismatch > mp.mpf("0.05")
        asym_rows.append((x_cut, float(d_val), float(t_val), float(rel_mismatch)))

    return bridge_rows, asym_rows


def stieltjes_l2_diag_bound(d_val, n_rs: int, x_cut: int):
    """Exact unconditional IBP Stieltjes upper bound B_primes for sum_{X < p <= N} p^{-1 - 2d} (X >= sqrt(N))."""
    assert x_cut >= int(math.isqrt(n_rs)), f"Requires X >= sqrt(N): got X={x_cut}, N={n_rs}"
    if n_rs <= x_cut:
        return mp.mpf(0)
    x_mp = mp.mpf(max(x_cut, 2))
    n_mp = mp.mpf(n_rs)
    d_mp = mp.mpf(d_val)

    return C_THETA_RS * (
        mp.power(x_mp, -2 * d_mp) / mp.log(x_mp)
        + mp.quad(lambda u: mp.power(u, -1 - 2 * d_mp) / mp.log(u), [x_mp, n_mp])
    )


def verify_section2_stieltjes_mertens_and_l1_divergence():
    print("\n" + "=" * 80)
    print("SECTION 2: Corrected Hyperbolic Ratio, L^2 Stieltjes Bound, Mertens & L^1 Divergence")
    print("=" * 80)

    # Check 2A: Exact hyperbolic mode ratio with N_*(t) = sqrt(t / (2*pi)), omega_n = ln(N_* / n)
    t_val = mp.mpf("50000")
    d_val = mp.mpf("0.18")
    n_star = mp.sqrt(t_val / (2 * mp.pi))
    omega_1 = mp.log(n_star)
    max_id_err = mp.mpf(0)
    for n in [1, 2, 3, 5, 11, 29, 71]:
        omega_n = mp.log(n_star / n)
        lhs = (mp.sinh(mp.mpf("0.5") * d_val * omega_n) / mp.sinh(mp.mpf("0.5") * d_val * omega_1)) ** 2
        rhs = mp.power(n, -d_val) * ((1 - mp.power(n / n_star, d_val)) / (1 - mp.power(n_star, -d_val))) ** 2
        max_id_err = max(max_id_err, abs(lhs - rhs))
        assert lhs <= mp.power(n, -d_val) * (1 + mp.mpf("1e-45"))
        lhs2 = (mp.sinh(d_val * omega_n) / mp.sinh(d_val * omega_1)) ** 2
        rhs2 = mp.power(n, -2 * d_val) * ((1 - mp.power(n / n_star, 2 * d_val)) / (1 - mp.power(n_star, -2 * d_val))) ** 2
        max_id_err = max(max_id_err, abs(lhs2 - rhs2))
    print(f"  [Check 2A] Max residual of exact hyperbolic mode ratios = {mp.nstr(max_id_err, 5)}")
    assert max_id_err < mp.mpf("1e-45")

    # Check 2B: Unconditional Stieltjes bound for forward L^2 diagonal sum
    diag_rows = []
    for t_num, d_num, x_cut in [(1e4, "0.10", 17), (1e5, "0.15", 43), (1e6, "0.20", 113), (1e7, "0.25", 311)]:
        t_mp = mp.mpf(str(t_num))
        d_mp = mp.mpf(d_num)
        n_rs = int(mp.floor(mp.sqrt(t_mp / (2 * mp.pi))))
        b_arr = compute_b_euler_sieve(n_rs, x_cut)
        exact_l2 = mp.mpf(0)
        for n in range(x_cut + 1, n_rs + 1):
            if b_arr[n] != 0:
                exact_l2 += mp.power(n, -1 - 2 * d_mp)
        bound_l2 = stieltjes_l2_diag_bound(d_mp, n_rs, x_cut)
        print(f"  [Check 2B] t={t_num:.0e}, N={n_rs:4d}, X={x_cut:3d}, d={d_num}: "
              f"E_diag^L2={mp.nstr(exact_l2,6)} <= B_St={mp.nstr(bound_l2,6)}")
        assert exact_l2 <= bound_l2
        diag_rows.append((t_num, n_rs, x_cut, float(d_mp), float(exact_l2), float(bound_l2)))

    # Check 2C: Mertens' theorem (1874) benchmark vs Chebyshev-Stieltjes threshold
    theta_mertens = mp.mpf("0.5") * mp.exp(-1)
    theta_stieltjes = mp.mpf("0.5") * mp.exp(-1 / C_THETA_RS)
    print(f"  [Check 2C] Mertens benchmark (1/2)e^{{-1}} = {mp.nstr(theta_mertens, 10)}, "
          f"Chebyshev-Stieltjes (1/2)e^{{-1/c_theta}} = {mp.nstr(theta_stieltjes, 10)}")

    # Check 2D: OBSTRUCTION — True RS weights n^{-1/2} * sinh^2(d*omega_n/2)/sinh^2(d*omega_1/2) ~ n^{-(1/2+d)} diverge!
    print("  [Check 2D] True RS-Weighted Prime-Power Sum Divergence (d = 0.20 < 1/2, X = floor(N_*^{1/2})):")
    l1_rows = []
    d_l1 = mp.mpf("0.20")
    for t_num in [1e4, 1e6, 1e8, 1e9]:
        t_mp = mp.mpf(str(t_num))
        n_star_t = mp.sqrt(t_mp / (2 * mp.pi))
        n_rs = int(mp.floor(n_star_t))
        x_cut = max(2, int(mp.floor(mp.sqrt(n_star_t))))
        b_arr = compute_b_euler_sieve(n_rs, x_cut)
        sh1_sq = mp.sinh(mp.mpf("0.5") * d_l1 * mp.log(n_star_t)) ** 2
        l1_true = mp.mpf(0)
        for n in range(x_cut + 1, n_rs + 1):
            if b_arr[n] != 0:
                om_n = mp.log(n_star_t / n)
                w_true = mp.power(n, -mp.mpf("0.5")) * (mp.sinh(mp.mpf("0.5") * d_l1 * om_n) ** 2) / sh1_sq
                l1_true += w_true
        print(f"    t={t_num:.0e}, N(t)={n_rs:5d}, X={x_cut:3d}: Y_W-pure^true = {mp.nstr(l1_true, 7)}")
        l1_rows.append((t_num, n_rs, x_cut, float(l1_true)))
    assert l1_rows[-1][3] > 2.5 * l1_rows[0][3] and l1_rows[-1][3] > 1.0

    return diag_rows, l1_rows, float(theta_mertens), float(theta_stieltjes)


def verify_section3_trig_identity_and_offdiag():
    print("\n" + "=" * 80)
    print("SECTION 3: Elementary Hyperbolic-Trig Identity & Off-Diagonal Obstruction")
    print("=" * 80)

    max_trig_err = mp.mpf(0)
    for a_str, b_str in [("0.37", "1.29"), ("1.85", "-2.41"), ("0.02", "14.77")]:
        a = mp.mpf(a_str)
        b = mp.mpf(b_str)
        lhs = (mp.cosh(a) ** 2) * (mp.cos(b) ** 2) + (mp.sinh(a) ** 2) * (mp.sin(b) ** 2)
        rhs = (mp.cos(b) ** 2) + (mp.sinh(a) ** 2)
        max_trig_err = max(max_trig_err, abs(lhs - rhs))
    print(f"  [Check 3A] Max residual of elementary cosh^2 cos^2 + sinh^2 sin^2 identity = {mp.nstr(max_trig_err, 5)}")
    assert max_trig_err < mp.mpf("1e-48")

    # Scan symmetric mode model |T(d,t)|^2 = D_E + C_off on t in [320, 1200] where N(t) in [7, 13] > X = 5
    d_scan = 0.30
    x_cut = 5
    t_grid = np.linspace(320.0, 1200.0, 1800)
    b_arr = compute_b_euler_sieve(20, x_cut)

    diag_vals = np.zeros_like(t_grid)
    total_vals = np.zeros_like(t_grid)
    for idx, tv in enumerate(t_grid):
        n_star_v = math.sqrt(tv / (2.0 * math.pi))
        n_rs = max(1, int(math.floor(n_star_v)))
        theta_rs = 0.5 * tv * math.log(tv / (2.0 * math.pi)) - 0.5 * tv - math.pi / 8.0
        s_tot = 0.0 + 0.0j
        d_sum = 0.0
        for n in range(1, n_rs + 1):
            if b_arr[n] == 0:
                continue
            phi_n = theta_rs - tv * math.log(n)
            om_n = math.log(n_star_v / n)
            tn = (2.0 / math.sqrt(n)) * (
                math.cosh(d_scan * om_n) * math.cos(phi_n)
                + 1j * math.sinh(d_scan * om_n) * math.sin(phi_n)
            )
            s_tot += b_arr[n] * tn
            d_sum += (b_arr[n] ** 2) * (4.0 / n) * ((math.cos(phi_n) ** 2) + (math.sinh(d_scan * om_n) ** 2))
        diag_vals[idx] = d_sum
        total_vals[idx] = abs(s_tot) ** 2

    offdiag_vals = total_vals - diag_vals
    min_ratio_idx = int(np.argmin(total_vals / diag_vals))
    min_ratio = float(total_vals[min_ratio_idx] / diag_vals[min_ratio_idx])
    print(f"  [Check 3B] At d={d_scan} (> 1/4), X={x_cut}: min(|T|^2 / D_E) on t in [320,1200] = {min_ratio:.4f} "
          f"at t={t_grid[min_ratio_idx]:.2f} (strong negative off-diagonal interference C_off < 0)")
    assert min_ratio < 0.70 and np.min(offdiag_vals) < -0.5

    # Also evaluate exact |zeta(0.80 + i t)| minimum on [14, 250] to illustrate Bohr/Voronin excursions on d = 0.30 > 1/4
    zeta_mods = []
    t_zeta_scan = np.linspace(14.0, 250.0, 400)
    mp.mp.dps = 20
    for tv in t_zeta_scan:
        z_v = abs(mp.zeta(mp.mpf("0.80") + 1j * mp.mpf(float(tv))))
        zeta_mods.append(float(z_v))
    mp.mp.dps = 50
    min_z = min(zeta_mods)
    t_min_z = float(t_zeta_scan[int(np.argmin(zeta_mods))])
    print(f"  [Check 3C] On sigma = 0.80 (d = 0.30 > 1/4), min |zeta(0.80 + i t)| on [14,250] = {min_z:.4f} at t={t_min_z:.2f}")

    return t_grid, diag_vals, total_vals, offdiag_vals, t_zeta_scan, np.array(zeta_mods)


def generate_figures(diag_rows, l1_rows, t_grid, diag_vals, total_vals, offdiag_vals, t_zeta_scan, zeta_mods):
    print("\n" + "=" * 80)
    print("GENERATING PUBLICATION FIGURES (FIGURES 1 & 2)")
    print("=" * 80)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9.5,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })

    # FIGURE 1: Bridge Envelope, L^2 Stieltjes Bound, Mertens Benchmark, and L^1 Divergence
    fig1, axes1 = plt.subplots(2, 2, figsize=(11.5, 8.2))

    # Panel 1A: Exact Bridge Remainder vs Gaussian Envelope (fast NumPy quadrature)
    ax = axes1[0, 0]
    d_arr = np.linspace(0.02, 0.48, 25)
    t_fixed_f = 25.010858
    primes_sub = sieve_primes(7)
    for x_cut, col, mk in [(3, "#1b4f72", "o"), (7, "#922b21", "s")]:
        p_sub = [p for p in primes_sub if p <= x_cut]
        divs = [(1, 1)]
        for p in p_sub:
            divs.extend([(dp * p, -mp_v) for dp, mp_v in list(divs)])
        ex_v, env_v = [], []
        # Precompute w grids per divisor
        div_grids = []
        for dd, md in divs:
            if dd == 1:
                continue
            ud = 0.5 * math.log(dd)
            w_g = np.linspace(0.0, ud, 60)
            dw = w_g[1] - w_g[0]
            wt = np.ones_like(w_g) * dw
            wt[0] *= 0.5
            wt[-1] *= 0.5
            phi_g = np.zeros_like(w_g)
            for n_m in range(1, 12):
                a_m = np.pi * (n_m ** 2) * np.exp(4.0 * w_g)
                phi_g += (2.0 * (np.pi ** 2) * (n_m ** 4) * np.exp(9.0 * w_g)
                          - 3.0 * np.pi * (n_m ** 2) * np.exp(5.0 * w_g)) * np.exp(-a_m)
            div_grids.append((dd, md, w_g, phi_g * wt))
        for dv in d_arr:
            s_c = (0.5 + dv) + 1j * t_fixed_f
            b_c = 0.0 + 0.0j
            b_e = 0.0
            for dd, md, w_g, p_wt in div_grids:
                b_c += 2.0 * md * (dd ** (-s_c)) * np.sum(p_wt * np.exp(2.0 * (dv + 1j * t_fixed_f) * w_g))
                b_e += 2.0 * (dd ** (-(0.5 + dv))) * np.sum(p_wt * np.exp(2.0 * dv * w_g))
            ex_v.append(abs(b_c))
            env_v.append(b_e)
        ax.semilogy(d_arr, env_v, "--", color=col, lw=1.4, label=rf"Envelope $\mathcal{{B}}_{{\mathrm{{env}}}}$ ($X={x_cut}$)")
        ax.semilogy(d_arr[::2], ex_v[::2], mk, color=col, ms=4.5, label=rf"Exact $|\mathcal{{B}}_{{\mathrm{{bridge}}}}|$ ($X={x_cut}$)")
    ax.set_xlabel(r"Transverse offset $d = \sigma - \frac{1}{2}$")
    ax.set_ylabel(r"Bridge magnitude ($t = 25.01$)")
    ax.set_title(r"A. Shifted Mellin Bridge Remainder vs. Envelope")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 1B: Hyperbolic Mode Ratio n^{-2d} ((1 - (n/N_*)^{2d})/(1 - N_*^{-2d}))^2
    ax = axes1[0, 1]
    t_demo = 1e6
    n_star_demo = math.sqrt(t_demo / (2.0 * math.pi))
    n_demo = np.arange(1, int(n_star_demo) + 1, dtype=float)
    for d_v, col in [(0.10, "#1f618d"), (0.25, "#117864"), (0.40, "#b9770e")]:
        om1 = math.log(n_star_demo)
        omn = np.log(n_star_demo / n_demo)
        ratio = (np.sinh(d_v * omn) / math.sinh(d_v * om1)) ** 2
        env_n = n_demo ** (-2.0 * d_v)
        ax.loglog(n_demo, ratio, "-", color=col, lw=1.6, label=rf"Exact $\sinh^2(d\omega_n)/\sinh^2(d\omega_1)$ ($d={d_v:.2f}$)")
        ax.loglog(n_demo, env_n, ":", color=col, lw=1.2, label=rf"Envelope $n^{{-2d}}$ ($d={d_v:.2f}$)")
    ax.set_ylim(1e-5, 1.5)
    ax.set_xlabel(r"Mode index $n \leq N(t)$ ($t = 10^6$)")
    ax.set_ylabel(r"Quadratic mode ratio")
    ax.set_title(r"B. Quadratic Hyperbolic Mode Ratio $\leq n^{-2d}$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", ncol=2, fontsize=7.2)

    # Panel 1C: Mertens Harmonic Benchmark vs L^2 Diagonal Sum
    ax = axes1[1, 0]
    thetas = np.linspace(0.08, 0.45, 80)
    mertens_curve = np.log(1.0 / (2.0 * thetas))
    stieltjes_curve = float(C_THETA_RS) * np.log(1.0 / (2.0 * thetas))
    ax.plot(thetas, mertens_curve, "-", color="#1b4f72", lw=1.8, label=r"Mertens asymptotic $\ln(1/(2\theta))$")
    ax.plot(thetas, stieltjes_curve, "--", color="#922b21", lw=1.4, label=r"Chebyshev bound $c_\vartheta \ln(1/(2\theta))$")
    ax.axhline(1.0, color="black", ls=":", lw=1.0, label="Unit benchmark ($= 1$)")
    ax.axvline(0.5 * math.exp(-1.0), color="#117864", ls="-.", lw=1.2, label=r"$\theta = \frac{1}{2}\mathrm{e}^{-1} \approx 0.18394$")
    ax.set_xlabel(r"Logarithmic sieve exponent $\theta$ ($X = N_\star(t)^{2\theta}$ vs. $N(t) = \lfloor N_\star(t)\rfloor$)")
    ax.set_ylabel(r"Harmonic prime tail $\sum_{X < p \leq N(t)} p^{-1}$")
    ax.set_title(r"C. Mertens' Theorem (1874) Harmonic Sieve Benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Panel 1D: Contrast between Convergent L^2 Diagonal and Divergent True L^1 Amplitude Sum
    ax = axes1[1, 1]
    t_vals_l1 = [r[0] for r in l1_rows]
    l1_vals = [r[3] for r in l1_rows]
    l2_cmp = []
    for t_num, n_rs, x_cut, _ in l1_rows:
        b_arr = compute_b_euler_sieve(n_rs, x_cut)
        l2_cmp.append(sum(n ** (-1.0 - 2.0 * 0.20) for n in range(x_cut + 1, n_rs + 1) if b_arr[n] != 0))
    ax.loglog(t_vals_l1, l1_vals, "o-", color="#922b21", lw=1.8, ms=5.5,
              label=r"True RS $L^1$ amplitude sum ($\sim N(t)^{1/2-d}/\ln N \to \infty$)")
    ax.loglog(t_vals_l1, l2_cmp, "s--", color="#117864", lw=1.8, ms=5.5,
              label=r"$L^2$ diagonal sum $\sum_{X<n\leq N} (b_n^{\mathcal{E}})^2 n^{-1-2d} \to 0$")
    ax.axhline(1.0, color="black", ls=":", lw=1.0)
    ax.set_xlabel(r"Spectral height $t$ ($d = 0.20$, $X = \lfloor\sqrt{N_\star(t)}\rfloor$)")
    ax.set_ylabel(r"Prime-power tail sum")
    ax.set_title(r"D. $L^1$ Amplitude Divergence vs. $L^2$ Diagonal Decay")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", fontsize=7.6)

    fig1.tight_layout()
    for out_dir in (FIG_DIR_ROOT, FIG_DIR_PAPER):
        fig1.savefig(os.path.join(out_dir, "fig1_bridge_complementarity_and_stieltjes_bounds.pdf"))
        fig1.savefig(os.path.join(out_dir, "fig1_bridge_complementarity_and_stieltjes_bounds.png"), dpi=220)
    plt.close(fig1)

    # FIGURE 2: Confocal Ellipse Mode Geometry, Off-Diagonal Cancellation, Branch Asymmetry, and Bohr/Voronin Excursions
    fig2, axes2 = plt.subplots(2, 2, figsize=(11.5, 8.2))

    # Panel 2A: Single-Mode Confocal Ellipse & Elementary Identity
    ax = axes2[0, 0]
    phi_plot = np.linspace(0, 2 * np.pi, 400)
    for a_v, col, lbl in [(0.20, "#1f618d", r"$a = 0.20$"), (0.55, "#117864", r"$a = 0.55$"), (0.95, "#922b21", r"$a = 0.95$")]:
        x_el = np.cosh(a_v) * np.cos(phi_plot)
        y_el = np.sinh(a_v) * np.sin(phi_plot)
        ax.plot(x_el, y_el, lw=1.6, color=col, label=lbl)
    ax.plot([-1, 1], [0, 0], "ko", ms=4.5, label=r"Foci $(\pm 1, 0)$")
    ax.set_aspect("equal", "box")
    ax.set_xlabel(r"$\cosh(a)\cos\phi$")
    ax.set_ylabel(r"$\sinh(a)\sin\phi$")
    ax.set_title(r"A. Single-Mode Ellipse $\cos^2\phi + \sinh^2 a \geq \sinh^2 a$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7.8)

    # Panel 2B: Multi-Mode Sum |T(d,t)|^2 = D_E + C_off Showing Destructive Off-Diagonal Interference
    ax = axes2[0, 1]
    mask_z = (t_grid >= 320.0) & (t_grid <= 460.0)
    ax.plot(t_grid[mask_z], diag_vals[mask_z], "--", color="#117864", lw=1.5, label=r"Diagonal $D_{\mathcal{E}}(d,t) > 0$")
    ax.plot(t_grid[mask_z], total_vals[mask_z], "-", color="#1b4f72", lw=1.5, label=r"Total $|T(d,t)|^2 = D_{\mathcal{E}} + \mathcal{C}_{\mathrm{off}}$")
    ax.plot(t_grid[mask_z], offdiag_vals[mask_z], ":", color="#922b21", lw=1.3, label=r"Off-diagonal $\mathcal{C}_{\mathrm{off}}(d,t)$")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel(r"Spectral height $t \in [320, 460]$ ($d = 0.30 > 1/4$, $X = 5$)")
    ax.set_ylabel(r"Quadratic mode sum")
    ax.set_title(r"B. Off-Diagonal Cross-Terms Cancel Diagonal Lower Bounds")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.6)

    # Panel 2C: Forward vs Reflected Branch Asymmetry |E_X(s) F_+(1-s) - F_+^{E_X}(1-s)|
    ax = axes2[1, 0]
    t_asym_grid = np.linspace(15.0, 60.0, 45)
    mp.mp.dps = 25
    for x_cut, col in [(5, "#1b4f72"), (11, "#922b21")]:
        b_c = compute_b_euler_sieve(30, x_cut)
        rel_diffs = []
        for tv in t_asym_grid:
            s_m = mp.mpf("0.75") + 1j * mp.mpf(float(tv))
            fp = sum(j_shell_closed_form(1 - s_m, m) for m in range(1, 16))
            tr = euler_sieve_poly_mp(s_m, x_cut) * fp
            sr = sum(b_c[m] * j_shell_closed_form(1 - s_m, m) for m in range(1, 16) if b_c[m] != 0)
            rel_diffs.append(float(abs(tr - sr) / abs(tr)))
        ax.plot(t_asym_grid, rel_diffs, "-", color=col, lw=1.6, label=rf"$X = {x_cut}$ ($d = 0.25$)")
    mp.mp.dps = 50
    ax.set_xlabel(r"Spectral height $t$")
    ax.set_ylabel(r"$|E_X(s)F_+(1-s) - F_+^{\mathcal{E}_X}(1-s)| / |E_X(s)F_+(1-s)|$")
    ax.set_title(r"C. Reflected-Branch Asymmetry ($E_X(s) \neq E_X(1-s)$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Panel 2D: Exact |zeta(0.80 + i t)| on sigma = 0.80 (d = 0.30 > 1/4) illustrating Bohr-Voronin Excursions
    ax = axes2[1, 1]
    ax.plot(t_zeta_scan, zeta_mods, "-", color="#1f618d", lw=1.4, label=r"$|\zeta(0.80 + \mathrm{i}t)|$ ($d = 0.30 > 1/4$)")
    ax.axhline(1.0, color="gray", ls=":", lw=1.0, label="Mean baseline $1$")
    ax.set_xlabel(r"Spectral height $t \in [14, 250]$")
    ax.set_ylabel(r"$|\zeta(\frac{1}{2} + d + \mathrm{i}t)|$")
    ax.set_title(r"D. Off-Line Excursions on $d = 0.30 > 1/4$ (Bohr–Voronin Regime)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    fig2.tight_layout()
    for out_dir in (FIG_DIR_ROOT, FIG_DIR_PAPER):
        fig2.savefig(os.path.join(out_dir, "fig2_confocal_ellipse_and_three_domain_synthesis.pdf"))
        fig2.savefig(os.path.join(out_dir, "fig2_confocal_ellipse_and_three_domain_synthesis.png"), dpi=220)
    plt.close(fig2)

    for fn in ("fig1_bridge_complementarity_and_stieltjes_bounds.pdf", "fig2_confocal_ellipse_and_three_domain_synthesis.pdf"):
        sz = os.path.getsize(os.path.join(FIG_DIR_PAPER, fn)) / 1024.0
        print(f"  [Saved] {fn}: {sz:.1f} KB (< 500 KB)")
        assert sz < 500.0


def main():
    t0 = time.time()
    print("=" * 80)
    print("PAPER IX VERIFICATION SUITE: BRIDGE IDENTITIES, DIAGONAL BOUNDS & OBSTRUCTIONS")
    print("=" * 80)
    verify_section0_canonical_wronskian()
    bridge_rows, asym_rows = verify_section1_bridge_and_asymmetry()
    diag_rows, l1_rows, th_m, th_s = verify_section2_stieltjes_mertens_and_l1_divergence()
    t_grid, diag_vals, total_vals, offdiag_vals, t_z, z_m = verify_section3_trig_identity_and_offdiag()
    generate_figures(diag_rows, l1_rows, t_grid, diag_vals, total_vals, offdiag_vals, t_z, z_m)
    print("\n" + "=" * 80)
    print(f"ALL PAPER IX VERIFICATIONS COMPLETED IN {time.time() - t0:.2f}s (EXIT CODE 0)")
    print("=" * 80)


if __name__ == "__main__":
    main()
