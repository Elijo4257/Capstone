# ----------------------------
# Resource-limited tumor-immune model (Tier 3)
# ----------------------------
import torch
torch.set_default_dtype(torch.float64)
DEVICE = "cpu"

# State indices (extend as you like)
IDX = {
    "T": 0,     # tumor
    "E": 1,     # effector CD8+
    "H": 2,     # helper CD4+
    "PDL1": 3,  # checkpoint axis activity
    "TGFb": 4,  # immunosuppressive cytokine
    "ctDNA": 5, # circulating tumor DNA
    "Adr": 6, "Cyc": 7, "Tax": 8, "Tam": 9, "IO": 10, "TIL": 11,  # drug states
    "N": 12,    # nutrient/oxygen resource (chemostat)
    "BM": 13    # bone marrow / immune setpoint resource
}

def sat(x, K, n=1.0):
    # Michaelis–Menten / Hill saturation: x^n / (K^n + x^n), safe for x>=0
    x_pos = torch.clamp(x, min=0)
    if n == 1.0:
        return x_pos / (K + x_pos + 1e-12)
    xn = torch.pow(x_pos + 1e-12, n)
    Kn = torch.pow(K + 1e-12, n)
    return xn / (Kn + xn)

def emax(C, Emax, EC50, n=1.0):
    # Emax/MM pharmacodynamics
    return Emax * sat(C, EC50, n=n)

def positive(x):  # tiny softplus to keep parameters > 0 if you optimize in unconstrained space
    return torch.nn.functional.softplus(x) + 1e-8

class ResourceLimitedRHS(torch.nn.Module):
    """
    Replaces K-style capacities with resources:
      - Tumor growth ~ rT_max * Sat_N * T
      - Immune growth ~ rE_max * Sat_BM * E, rH_max * Sat_BM * H
    """
    def __init__(self, p):
        super().__init__()
        self.p = p  # dictionary of scalars (torch tensors or floats)

    def forward(self, t, x):
        p = self.p
        # unpack states (vectorized batch OK if x is (..., state_dim))
        T    = x[..., IDX["T"]]
        E    = x[..., IDX["E"]]
        H    = x[..., IDX["H"]]
        PDL1 = x[..., IDX["PDL1"]]
        TGFb = x[..., IDX["TGFb"]]
        ct   = x[..., IDX["ctDNA"]]

        Adr  = x[..., IDX["Adr"]]
        Cyc  = x[..., IDX["Cyc"]]
        Tax  = x[..., IDX["Tax"]]
        Tam  = x[..., IDX["Tam"]]
        IO   = x[..., IDX["IO"]]
        TIL  = x[..., IDX["TIL"]]

        N    = x[..., IDX["N"]]     # nutrient/oxygen resource
        BM   = x[..., IDX["BM"]]    # marrow/immune setpoint resource

        # ----------------------------
        # Saturating signals & PD terms
        # ----------------------------
        SN     = sat(N,   p["K_N"],   n=p["n_N"])          # nutrient-limited tumor proliferation
        SBM    = sat(BM,  p["K_BM"],  n=p["n_BM"])         # bone-marrow-limited immune proliferation
        SIO    = sat(IO,  p["K_IO"],  n=p["n_IO"])
        SPD    = sat(PDL1,p["K_PDL1"],n=p["n_PDL1"])       # PD-L1 suppression signal
        STGFb  = sat(TGFb,p["K_TGFb"],n=p["n_TGFb"])

        # PD-L1/TGFβ reduce effective immune growth (multiplicative dampers)
        damp_PD   = 1.0 - torch.clamp(p["phi_PD"]*SPD,   0.0, 1.0)
        damp_TGFb = 1.0 - torch.clamp(p["phi_TGFb"]*STGFb, 0.0, 1.0)

        # IO stimulates immune and blocks PD-L1 axis
        stim_IO   = 1.0 + p["gamma_IO"]*SIO
        block_PD  = 1.0 - p["gamma_IO_PD"]*SIO            # reduces net PD effect

        # Chemo Emax/MM cytotoxic effects (apply to proliferating fraction of T)
        EAdr = emax(Adr, p["Emax_Adr"], p["EC50_Adr"], n=p["n_Adr"])
        ECyc = emax(Cyc, p["Emax_Cyc"], p["EC50_Cyc"], n=p["n_Cyc"])
        ETax = emax(Tax, p["Emax_Tax"], p["EC50_Tax"], n=p["n_Tax"])
        chemo_kill_rate = (EAdr + ECyc + ETax) * sat(N, p["K_N_prolif"])  # more kill when nutrient supports cycling

        # Immune kill (Holling II): k_ET * E * T/(K_ET + T)
        kill_immune = p["k_ET"] * E * (T / (p["K_ET"] + T + 1e-12))

        # Tam endocrine modulation (down-weights proliferation drive)
        STam = sat(Tam, p["K_Tam"])
        g_endo = 1.0 - torch.clamp(p["k_Tam_ERPR"] * STam, 0.0, 1.0)

        # ----------------------------
        # Dynamics
        # ----------------------------

        # (1) Tumor: resource-limited growth + losses
        #   dT = rT_max * g_endo * SN * T  - (immune kill + chemo kill) * T
        dT = p["rT_max"] * g_endo * SN * T  - (kill_immune + chemo_kill_rate) * T
        # Optional: add PD-L1 "boost" to net tumor growth (immune escape)
        if "k_PDboost_T" in p:
            dT = dT + p["k_PDboost_T"] * SPD * T

        # (2) Effector (E): resource-limited growth with antigen & helper boosts; PD-L1/TGFβ dampers; exhaustion/turnover
        stim_T  = p["k_T_to_E"] * sat(T, p["K_Tstim_E"])
        boost_H = p["k_H_to_E"] * sat(H, p["K_Hboost_E"])
        growth_E = p["rE_max"] * SBM * (1.0 + stim_T + boost_H) * damp_PD * block_PD * damp_TGFb * stim_IO
        dE = growth_E * E  - p["delta_E"] * E

        # (3) Helper (H): resource-limited + antigen stimulus; similar dampers
        stimT_H = p["k_T_to_H"] * sat(T, p["K_Tstim_H"])
        growth_H = p["rH_max"] * SBM * (1.0 + stimT_H) * damp_PD * block_PD * damp_TGFb
        dH = growth_H * H  - p["delta_H"] * H

        # (4) PD-L1: upregulated by T and E (IFNγ axis), decays; blocked by IO (modeled as extra decay)
        dPDL1 = p["PDL1_basal"] + p["k_PDL1_up_T"]*sat(T, p["K_PDL1_up_T"]) + p["k_PDL1_up_E"]*sat(E, p["K_PDL1_up_E"]) \
                - (p["k_PDL1_decay"] + p["k_IO_block_PDL1"]*SIO) * PDL1

        # (5) TGFβ: secreted by T, decays
        dTGFb = p["k_T_secrete_TGFb"] * T - p["k_TGFb_decay"] * TGFb

        # (6) ctDNA: shed from T, cleared
        dct = p["p_ct"] * T - p["k_ctclr"] * ct

        # (7) PK: simple one-compartment first-order decay (controls can add to these states externally)
        dAdr = -p["lambda_Adr"] * Adr
        dCyc = -p["lambda_Cyc"] * Cyc
        dTax = -p["lambda_Tax"] * Tax
        dTam = -p["lambda_Tam"] * Tam
        dIO  = -p["lambda_IO"]  * IO
        dTIL = -p["lambda_TIL"] * TIL

        # (8) Resource N (chemostat): supply - washout - consumption by T and E
        #     dN = sN - dN*N - qT*T*Sat_Nuse - qE*E*Sat_NuseE
        #     You can let high-dose chemo transiently drop N via an extra term if desired.
        SNuse_T = sat(N, p["K_N_use_T"])
        SNuse_E = sat(N, p["K_N_use_E"])
        chemo_N_drop = p["chi_N_chemo"] * (EAdr + ECyc + ETax)  # optional: therapy reduces effective nutrient temporarily
        dN = p["sN"] - p["dN"]*N - p["qT"]*T*SNuse_T - p["qE"]*E*SNuse_E - chemo_N_drop * N

        # (9) Resource BM: supply - decay - myelosuppression by chemo; recovery boosted by IO/TIL if desired
        tox_signal = (EAdr * p["tox_w_Adr"] + ECyc * p["tox_w_Cyc"] + ETax * p["tox_w_Tax"])
        recov_boost = 1.0 + p["gamma_BM_IO"]*SIO + p["gamma_BM_TIL"]*sat(TIL, p["K_TIL"])
        dBM = p["sBM"]*recov_boost - p["dBM"]*BM - p["qBM_chemo"]*tox_signal*BM

        # stack
        dx = torch.stack([dT, dE, dH, dPDL1, dTGFb, dct, dAdr, dCyc, dTax, dTam, dIO, dTIL, dN, dBM], dim=-1)
        return dx

# ----------------------------------------
# Minimal RK4 (same shape as your existing helper)
# ----------------------------------------
def rk4_step(f, t, x, h):
    k1 = f(t, x)
    k2 = f(t + 0.5*h, x + 0.5*h*k1)
    k3 = f(t + 0.5*h, x + 0.5*h*k2)
    k4 = f(t + h,     x + h*k3)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def solve_ode(x0, rhs_module, t0, t1, saveat):
    x = x0
    t_prev = t0
    xs = [x0]
    ts = [t0]
    for t_next in saveat[1:]:
        h = t_next - t_prev
        x = rk4_step(rhs_module.forward, t_prev, x, h)
        xs.append(x)
        ts.append(t_next)
        t_prev = t_next
    return torch.stack(ts), torch.stack(xs)
