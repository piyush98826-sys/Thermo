"""
Real-gas PVT behaviour and thermodynamic properties (H and S) from cubic
equations of state.

Project  : Software for predicting PVT behaviour of real gases using a cubic EOS
HW part  : Determination of enthalpy (H) and entropy (S) using the cubic EOS

Equations of state (generalised form, Smith-Van Ness-Abbott):
    P = R T/(V - b) - a(T) / ((V + eps*b)(V + sigma*b))
        vdW : Van der Waals       (sigma = eps = 0)
        RK  : Redlich-Kwong       (sigma = 1, eps = 0)
        SRK : Soave-Redlich-Kwong (sigma = 1, eps = 0)
        PR  : Peng-Robinson       (sigma = 1+sqrt2, eps = 1-sqrt2)

Usage:
    python real_gas_eos.py            -> interactive calculator
    python real_gas_eos.py --demo     -> reproduces all tables/figures of report

Units: T [K], P [bar], V [cm3/mol], H [J/mol], S [J/mol/K]
"""
import sys
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

R = 8.314462          # J/(mol K)
R_V = 83.14462        # cm3 bar/(mol K)
T0, P0 = 298.15, 1.0  # reference state: ideal gas, H = S = 0


# ----------------------------------------------------------------------------
# 1. Data base  (Smith, Van Ness & Abbott, Tables B.1 and C.1)
#    Cp_ig/R = A + B*T + D/T^2       (T in K, valid roughly 300-1500 K)
# ----------------------------------------------------------------------------
@dataclass
class Gas:
    name: str
    Tc: float      # K
    Pc: float      # bar
    w: float       # acentric factor
    A: float
    B: float
    D: float = 0.0


GASES = {
    "methane":  Gas("Methane",        190.6, 45.99, 0.012, 1.702, 9.081e-3, 0.0),
    "ethane":   Gas("Ethane",         305.3, 48.72, 0.100, 1.131, 19.225e-3, 0.0),
    "propane":  Gas("Propane",        369.8, 42.48, 0.152, 1.213, 28.785e-3, 0.0),
    "butane":   Gas("n-Butane",       425.1, 37.96, 0.200, 1.935, 36.915e-3, 0.0),
    "co2":      Gas("Carbon dioxide", 304.2, 73.83, 0.224, 5.457, 1.045e-3, -1.157e5),
    "nitrogen": Gas("Nitrogen",       126.2, 34.00, 0.038, 3.280, 0.593e-3, 0.040e5),
}


# ----------------------------------------------------------------------------
# 2. Equation-of-state definitions
# ----------------------------------------------------------------------------
@dataclass
class EOS:
    name: str
    sigma: float
    eps: float
    omega: float   # b = omega * R Tc / Pc
    psi: float     # a = psi * alpha * R^2 Tc^2 / Pc

    def kappa(self, w):
        if self.name == "SRK":
            return 0.480 + 1.574 * w - 0.176 * w**2
        if self.name == "PR":
            return 0.37464 + 1.54226 * w - 0.26992 * w**2
        return 0.0

    def alpha(self, Tr, w):
        if self.name == "vdW":
            return 1.0
        if self.name == "RK":
            return Tr ** -0.5
        return (1.0 + self.kappa(w) * (1.0 - np.sqrt(Tr))) ** 2

    def dlnalpha(self, Tr, w):
        """d ln(alpha) / d ln(Tr)"""
        if self.name == "vdW":
            return 0.0
        if self.name == "RK":
            return -0.5
        k = self.kappa(w)
        return -k * np.sqrt(Tr) / (1.0 + k * (1.0 - np.sqrt(Tr)))


s2 = np.sqrt(2.0)
EOS_LIST = {
    "vdW": EOS("vdW", 0.0, 0.0, 1 / 8, 27 / 64),
    "RK":  EOS("RK", 1.0, 0.0, 0.08664, 0.42748),
    "SRK": EOS("SRK", 1.0, 0.0, 0.08664, 0.42748),
    "PR":  EOS("PR", 1 + s2, 1 - s2, 0.07780, 0.45724),
}


# ----------------------------------------------------------------------------
# 3. Compressibility factor: roots of the cubic
# ----------------------------------------------------------------------------
def beta_q(eos, gas, T, P):
    Tr, Pr = T / gas.Tc, P / gas.Pc
    beta = eos.omega * Pr / Tr
    q = eos.psi * eos.alpha(Tr, gas.w) / (eos.omega * Tr)
    return beta, q


def z_roots(eos, gas, T, P):
    """Physical (real, Z > beta) roots of the cubic in Z, ascending order.
    Z = 1 + beta - q*beta*(Z - beta)/((Z + eps*beta)(Z + sigma*beta))"""
    beta, q = beta_q(eos, gas, T, P)
    s, e = eos.sigma, eos.eps
    coeff = [1.0,
             (e + s) * beta - (1 + beta),
             e * s * beta**2 - (1 + beta) * (e + s) * beta + q * beta,
             -(1 + beta) * e * s * beta**2 - q * beta**2]
    r = np.roots(coeff)
    r = np.sort(r[np.abs(r.imag) < 1e-9].real)
    return r[r > beta * (1 + 1e-12)]


def I_term(eos, Z, beta):
    if abs(eos.sigma - eos.eps) < 1e-12:       # van der Waals
        return beta / Z
    return np.log((Z + eos.sigma * beta) / (Z + eos.eps * beta)) / (eos.sigma - eos.eps)


def ln_phi(eos, gas, T, P, Z):
    beta, q = beta_q(eos, gas, T, P)
    return Z - 1 - np.log(Z - beta) - q * I_term(eos, Z, beta)


# ----------------------------------------------------------------------------
# 4. Vapour pressure (equal fugacity) - needed to choose the right root
# ----------------------------------------------------------------------------
def psat(eos, gas, T):
    """Saturation pressure [bar] and (Z_liq, Z_vap); None if T >= Tc or no loop."""
    if T >= gas.Tc:
        return None
    grid = gas.Pc * np.logspace(-6, 0, 600)
    three = [P for P in grid if len(z_roots(eos, gas, T, P)) == 3]
    if len(three) < 2:
        return None

    def f(P):
        z = z_roots(eos, gas, T, P)
        return ln_phi(eos, gas, T, P, z[0]) - ln_phi(eos, gas, T, P, z[-1])

    lo, hi = min(three), max(three)
    if f(lo) * f(hi) > 0:
        return None
    P = brentq(f, lo, hi, xtol=1e-12, rtol=1e-12)
    z = z_roots(eos, gas, T, P)
    return P, z[0], z[-1]


# ----------------------------------------------------------------------------
# 5. Ideal-gas contributions and residual (departure) properties
# ----------------------------------------------------------------------------
def int_cp(g, T):          # integral of Cp_ig/R dT   (antiderivative)
    return g.A * T + g.B * T**2 / 2 - g.D / T


def int_cp_T(g, T):        # integral of Cp_ig/(R T) dT (antiderivative)
    return g.A * np.log(T) + g.B * T - g.D / (2 * T**2)


def departure(eos, gas, T, P, Z):
    """H^R/RT and S^R/R at (T, P) for root Z."""
    Tr = T / gas.Tc
    beta, q = beta_q(eos, gas, T, P)
    I = I_term(eos, Z, beta)
    dla = eos.dlnalpha(Tr, gas.w)
    HR_RT = Z - 1 + (dla - 1.0) * q * I
    SR_R = np.log(Z - beta) + dla * q * I
    return HR_RT, SR_R


def properties(eos, gas, T, P, phase="auto"):
    """All properties at (T, P). phase: 'auto', 'vapor' or 'liquid'."""
    z = z_roots(eos, gas, T, P)
    if len(z) == 0:
        raise ValueError("No physical root found for Z.")
    if len(z) == 1:
        Z, ph = z[0], ("vapor" if z[0] > 0.3 else "liquid")
        if T < gas.Tc:   # single root below Tc: decide from phase boundary
            ps = psat(eos, gas, T)
            ph = "vapor" if (ps is None or P < ps[0]) else "liquid"
        elif z[0] < 0.3:
            ph = "dense fluid"
        else:
            ph = "gas"
    else:
        ps = psat(eos, gas, T)
        if phase == "auto":
            phase = "vapor" if (ps is None or P < ps[0]) else "liquid"
        Z, ph = (z[-1], "vapor") if phase == "vapor" else (z[0], "liquid")
    HR_RT, SR_R = departure(eos, gas, T, P, Z)
    H_ig = R * (int_cp(gas, T) - int_cp(gas, T0))
    S_ig = R * (int_cp_T(gas, T) - int_cp_T(gas, T0)) - R * np.log(P / P0)
    return dict(phase=ph, Z=Z, V=Z * R_V * T / P,
                HR=HR_RT * R * T, SR=SR_R * R,
                H=H_ig + HR_RT * R * T, S=S_ig + SR_R * R)


def enthalpy_of_vaporization(eos, gas, T):
    ps = psat(eos, gas, T)
    if ps is None:
        return None
    P, Zl, Zv = ps
    hl, sl = departure(eos, gas, T, P, Zl)
    hv, sv = departure(eos, gas, T, P, Zv)
    return P, (hv - hl) * R * T


# ----------------------------------------------------------------------------
# 6. Interactive front end
# ----------------------------------------------------------------------------
def ask_choice(prompt, options):
    print(prompt)
    keys = list(options)
    for i, k in enumerate(keys, 1):
        print(f"  {i}. {k}")
    while True:
        s = input("Choice (number or name): ").strip()
        if s.isdigit() and 1 <= int(s) <= len(keys):
            return keys[int(s) - 1]
        if s in options:
            return s
        print("  invalid, try again.")


def interactive():
    print("=== Real-gas PVT, H and S from cubic equations of state ===\n")
    while True:
        gk = ask_choice("Select gas:", GASES)
        ek = ask_choice("Select equation of state:", EOS_LIST)
        gas, eos = GASES[gk], EOS_LIST[ek]
        T = float(input("Temperature T [K]: "))
        P = float(input("Pressure P [bar]: "))
        try:
            p = properties(eos, gas, T, P)
            print(f"\n{gas.name}, {ek}, T = {T} K, P = {P} bar  ({p['phase']})")
            print(f"  Z        = {p['Z']:.4f}")
            print(f"  V        = {p['V']:.2f} cm3/mol")
            print(f"  H^R      = {p['HR']:.1f} J/mol     S^R = {p['SR']:.3f} J/mol/K")
            print(f"  H        = {p['H']:.1f} J/mol     S   = {p['S']:.3f} J/mol/K")
            print(f"  (reference: ideal gas at {T0} K and {P0} bar, H = S = 0)\n")
        except ValueError as err:
            print("Error:", err)
        if input("Another calculation? (y/n): ").strip().lower() != "y":
            break


# ----------------------------------------------------------------------------
# 7. Demo / report generator
# ----------------------------------------------------------------------------
def demo(outdir="."):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- Table 1: four EOS at one state -------------------------------------
    print("TABLE 1: propane, T = 400 K, P = 20 bar")
    g = GASES["propane"]
    for k, e in EOS_LIST.items():
        p = properties(e, g, 400.0, 20.0)
        print(f"{k:4s} Z={p['Z']:.4f} V={p['V']:8.1f} HR={p['HR']:9.1f} "
              f"SR={p['SR']:7.3f} H={p['H']:9.1f} S={p['S']:7.3f}")

    # --- Table 2: enthalpy of vaporisation at the normal boiling point -------
    print("\nTABLE 2: dHvap at normal boiling point (PR and SRK)")
    for name, Tb, lit in [("methane", 111.67, 8.19), ("nitrogen", 77.36, 5.56),
                          ("propane", 231.04, 18.77)]:
        gas = GASES[name]
        for k in ("SRK", "PR"):
            r = enthalpy_of_vaporization(EOS_LIST[k], gas, Tb)
            print(f"{name:9s} {k:3s} Psat={r[0]:.3f} bar  dHvap={r[1]/1000:.2f} kJ/mol "
                  f"(lit {lit}, err {100*(r[1]/1000-lit)/lit:+.1f}%)")

    # --- Figure 1: PV isotherms (propane, PR) and Z vs P (methane) ----------
    pr = EOS_LIST["PR"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    gas = GASES["propane"]
    for T in (320, 350, 369.8, 400, 450):
        b = pr.omega * R_V * gas.Tc / gas.Pc
        Vs = np.logspace(np.log10(b * 1.03), np.log10(2500), 3000)
        a = pr.psi * pr.alpha(T / gas.Tc, gas.w) * (R_V * gas.Tc) ** 2 / gas.Pc
        Pp = R_V * T / (Vs - b) - a / ((Vs + pr.eps * b) * (Vs + pr.sigma * b))
        ax[0].plot(Vs, Pp, label=f"{T:g} K")
    Tsat = np.linspace(260, gas.Tc - 1.0, 60)
    dome = [psat(pr, gas, T) for T in Tsat]
    Tsat = [T for T, d in zip(Tsat, dome) if d is not None]
    dome = [d for d in dome if d is not None]
    Vl = [R_V * T * d[1] / d[0] for T, d in zip(Tsat, dome)]
    Vv = [R_V * T * d[2] / d[0] for T, d in zip(Tsat, dome)]
    ax[0].plot(Vl + Vv[::-1], [d[0] for d in dome] + [d[0] for d in dome][::-1],
               "k--", lw=1, label="saturation")
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 70); ax[0].set_xlim(60, 2500)
    ax[0].set_xlabel("V [cm$^3$/mol]"); ax[0].set_ylabel("P [bar]")
    ax[0].set_title("(a) Propane, PR isotherms", fontsize=10)
    ax[0].legend(fontsize=7)

    gm = GASES["methane"]
    Ps = np.linspace(1, 300, 300)
    for k, st in zip(EOS_LIST, ("-", "--", "-.", ":")):
        Z = [properties(EOS_LIST[k], gm, 250.0, P)["Z"] for P in Ps]
        ax[1].plot(Ps, Z, st, label=k)
    ax[1].axhline(1, color="grey", lw=0.6)
    ax[1].set_xlabel("P [bar]"); ax[1].set_ylabel("Z")
    ax[1].set_title("(b) Methane, Z at 250 K", fontsize=10)
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig1.png", dpi=200)

    # --- Figure 2: H^R and S^R (methane, PR) ----------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.4))
    Ps = np.linspace(1, 200, 200)
    for T in (200, 250, 300, 400, 500):
        res = [properties(pr, gm, T, P) for P in Ps]
        ax[0].plot(Ps, [r["HR"] / 1000 for r in res], label=f"{T} K")
        ax[1].plot(Ps, [r["SR"] for r in res], label=f"{T} K")
    ax[0].set_xlabel("P [bar]"); ax[0].set_ylabel("$H^R$ [kJ/mol]")
    ax[0].set_title("(a) Residual enthalpy, methane (PR)", fontsize=10)
    ax[1].set_xlabel("P [bar]"); ax[1].set_ylabel("$S^R$ [J/(mol K)]")
    ax[1].set_title("(b) Residual entropy, methane (PR)", fontsize=10)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{outdir}/fig2.png", dpi=200)

    # --- Thermodynamic consistency check ---------------------------------------
    # (dH/dP)_T = V - T (dV/dT)_P   checked numerically for PR methane
    T, P, h = 300.0, 50.0, 1e-3
    Hm = lambda p: properties(pr, gm, T, p)["H"]
    Vf = lambda t: properties(pr, gm, t, P)["V"]
    lhs = (Hm(P + h) - Hm(P - h)) / (2 * h) * 10.0   # J/(mol bar) -> cm3/mol (1 J = 10 cm3 bar)
    dVdT = (Vf(T + h) - Vf(T - h)) / (2 * h)
    rhs = (Vf(T) - T * dVdT)
    print(f"\nConsistency: (dH/dP)_T = {lhs:.4f} cm3/mol, V - T(dV/dT)_P = {rhs:.4f} cm3/mol")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        interactive()
