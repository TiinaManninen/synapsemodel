# Simulation of tripartite synapse model
# Tiina Manninen and Ausra Saudargiene
# Reference: Tiina Manninen, Ausra Saudargiene, and Marja-Leena Linne. Astrocyte-mediated spike-timing-dependent
# long-term depression modulates synaptic properties in the developing cortex. PLoS Comput Biol, 2020.

# -----------------------------------------------------------------------

from math import exp


# Presynaptic neuron model includes the model components in the presynaptic neuron only.
class Pre:

    __DEFAULT_INITIAL_VALUES = {
        "Ca_CaNHVA_pre": 0.082523,  # uM; Presynaptic Ca_NHVA-mediated Ca concentration
        "Ca_NMDAR_pre": 0.05,       # uM; Presynaptic NMDAR-mediated Ca concentration
        "CaN_pre": 0.00012499,      # uM; Presynaptic CaN concentration
        "h_CaNHVA_pre": 0.91675,    # 1; Gating variable for presynaptic Ca_NHVA inactivation
        "h_Na_pre": 0.99976,        # 1; Gating variable for presynaptic Na inactivation
        "m_CaNHVA_pre": 0.006381,   # 1; Gating variable for presynaptic Ca_NHVA activation
        "m_Na_pre": 4.5445e-5,      # 1; Gating variable for presynaptic Na activation
        "n_K_pre": 4.5445e-5,       # 1; Gating variable for presynaptic K activation
        "n_K2_pre": 4.5445e-5,      # 1; Gating variable for presynaptic K activation
        "Prel_pre": 0,              # 1; Release probability of presynaptic Glu vesicles
        "R_pre": 0,                 # 1; Fraction of presynaptic NMDARs in closed state
        "RA_pre": 0,                # 1; Fraction of presynaptic NMDARs in single liganded state
        "RA2_pre": 0,               # 1; Fraction of presynaptic NMDARs in fully liganded state
        "RA2d1_pre": 0,             # 1; Fraction of presynaptic NMDARs in desensitized state
        "RA2d2_pre": 0,             # 1; Fraction of presynaptic NMDARs in desensitized state
        "RA2f_pre": 0,              # 1; Fraction of presynaptic NMDARs in faster conformational change of GluN1 subunit
        "RA2s_pre": 0,              # 1; Fraction of presynaptic NMDARs in slower conformational change of GluN2 subunit
        "RA2_O_pre": 0,             # 1; Fraction of presynaptic NMDARs in open state
        "RAMg_pre": 0,              # 1; Fraction of presynaptic NMDARs in single liganded Mg-blocked state
        "RA2Mg_pre": 0,             # 1; Fraction of presynaptic NMDARs in fully liganded Mg-blocked state
        "RA2d1Mg_pre": 0,           # 1; Fraction of presynaptic NMDARs in desensitized Mg-blocked state
        "RA2d2Mg_pre": 0,           # 1; Fraction of presynaptic NMDARs in desensitized Mg-blocked state
        "RA2fMg_pre": 0,            # 1; Fraction of presynaptic NMDARs in faster conformational change of Mg-blocked \
                                    # GluN1 subunit
        "RA2sMg_pre": 0,            # 1; Fraction of presynaptic NMDARs in slower conformational change of Mg-blocked \
                                    # GluN2 subunit
        "RA2_OMg_pre": 0,           # 1; Fraction of presynaptic NMDARs in Mg-blocked state
        "RMg_pre": 1,               # 1; Fraction of presynaptic NMDARs in closed Mg-blocked state
        "Rrel_pre": 1,              # 1; Fraction of releasable presynaptic vesicles
        "s_Na_pre": 0.99983,        # 1; Gating variable for presynaptic Na inactivation
        "V_pre": -59.9969,          # mV; Presynaptic membrane potential
        "X_ac_pre": 0               # uM; Concentration of presynaptic active protein affecting vesicular release
    }

    __DEFAULT_PARAMS = {
        # Basic parameters
        "F": 96485,                 # C/mol; Faraday constant
        "N_A": 6.0221e23,           # 1/mol; Avogadro's constant
        "R": 8.3145,                # J/(K mol); Molar gas constant
        "T_celsius": 36,            # C; Temperature; Sarid et al., 2007
        "z": 2,	                    # 1; Valence of Ca ion

        # Membrane capacitance
        "Cm_pre": 1.5,              # uF/cm^2; Presynaptic membrane capacitance per unit area; Lavzin et al., 2012

        # Conductances
        "gCaNHVA_pre": 0.3,         # mS/cm^2; Maximum conductance of presynaptic Ca_NHVA current per unit area; \
                                    # Safiulina et al., 2010: 0.0003 mho/cm^2 = 0.0003 S/cm^2 = 0.3 mS/cm^2
        "gK_pre": 20,               # mS/cm^2; Maximum conductance of presynaptic K current per unit area; \
                                    # Modified from Lavzin et al., 2012: 120 mS/cm^2
        "gK2_pre": 20,              # mS/cm^2; Maximum conductance of presynaptic K current per unit area; \
                                    # Modified from Lavzin et al., 2012: 120 mS/cm^2
        "gL_pre": 0.2,              # mS/cm^2; Leak conductance of presynaptic neuron per unit area; \
                                    # Modified from Lavzin et al., 2012: 0.1 mS/cm^2
        "gNa_pre": 30,              # mS/cm^2; Maximum conductance of presynaptic Na current per unit area; \
                                    # Modified from Lavzin et al., 2012: 200 mS/cm^2
        "gNMDAR_pre": 0.1,          # mS/cm^2; Maximum conductance of presynaptic NMDAR per unit area; \
                                    # Modified from Gabbiani and Cox, 2010: 0.014  mS/cm^2

        # Particle parameters
        "tau_h_CaNHVA_pre": 80,     # ms; Time constant for presynaptic Ca_NHVA inactivation; Safiulina et al., 2010
        "tau_h_Na_pre": 1,          # ms; Time constant for presynaptic Na inactivation; \
                                    # Modified from Lavzin et al., 2012: 0.5 ms
        "tau_m_Na_pre": 0.05,       # ms; Time constant for presynaptic Na activation; Lavzin et al., 2012
        "tau_n_K_pre": 1,           # ms; Time constant for presynaptic K activation; Lavzin et al., 2012
        "tau_n_K2_pre": 10,         # ms; Time constant for presynaptic K activation; Lavzin et al., 2012
        "tau_s_Na_pre": 30,         # ms; Time constant for presynaptic Na inactivation; Lavzin et al., 2012
        "tau_sb_Na_pre": 0.1,       # ms; Time constant for presynaptic Na inactivation; \
                                    # Modified from Lavzin et al., 2012: 0.5 ms
        "V_sd_Na_pre": 1,           # mV; Shift in presynaptic membrane potential used in calculation of Na current; \
                                    # Lavzin et al., 2012
        "Vshift_pre": -10,          # mV; Shift in presynaptic membrane potential; \
                                    # Modified from Lavzin et al., 2012: 0 mV
        "V_sv_Na_pre": 10,          # mV; Shift in presynaptic membrane potential used in calculation of Na current; \
                                    # Modified from Lavzin et al., 2012: 30 mV

        # Reversal potentials
        "VK_pre": -77,              # mV; Reversal potential of presynaptic K current; Lavzin et al., 2012
        "VL_pre": -60,              # mV; Leak reversal potential of presynaptic neuron; \
                                    # Modified from Lavzin et al., 2012: -70mV
        "VNa_pre": 50,              # mV; Reversal potential of presynaptic Na current; \
                                    # Modified from Lavzin et al., 2012: 40 mv and 70mV
        "V_NMDAR_pre": 0,           # mV; Reversal potential of presynaptic NMDAR current; Destexhe et al., 1998

        # Other calcium related parameters
        # Safiulina et al., 2010
        "a0m_pre": 0.03,            # 1/ms; Parameter used in calculation of presynaptic Ca_NHVA activation time \
                                    # constant
        "c_Ca_pre": 1.9297,         # (uA ms)/(cm^2 uM); Presynaptic scaling factor to convert from units uA/cm^2 to \
                                    # uM/ms
        "c_V_pre": 13.3203,         # mV; Presynaptic scaling factor to convert from units mV to 1 used in calculation \
                                    # of GHK current equation
        "Ca_ext_pre": 2000,         # uM; Ca concentration outside presynaptic neuron
        "Ca_rest_pre": 0.05,        # uM; Presynaptic resting Ca concentration
        "d_pre": 0.1,               # um; Depth of presynaptic axonal shell
        "gmm_pre": 0.1,             # 1; Parameter used in calculation of presynaptic Ca_NHVA activation time constant
        "k_Ca_pre": 10000,          # 1; Presynaptic scaling factor used in calculation of c_Ca_pre
        "k_V_pre": 1000,            # 1; Presynaptic scaling factor used in calculation of c_V_pre
        "K_inh_pre": 1,             # uM; Presynaptic Ca_NHVA channel dissociation constant (inhibition)
        "tau_Ca_pre": 100,          # ms; Time constant of presynaptic pump
        "tau_mmin_pre": 0.2,        # ms; Minimun value for presynaptic Ca_NHVA activation time constant
        "Vhalfm_pre": -14,          # mV; Half-activation potential of presynaptic Ca_NHVA channel
        "zm_pre": 2, 	            # 1; Parameter used in calculation of presynaptic Ca_NHVA activation time constant

        # NMDAR parameters
        "kd1_f_pre": 0.055,         # 1/ms; Rate constant into desensitized state of presynaptic NMDAR; \
                                    # Modified from Erreger et al., 2005: 0.55 1/ms
        "kd1_b_pre": 0.0814,        # 1/ms; Rate constant out of desensitized state of presynaptic NMDAR; \
                                    # Erreger et al., 2005
        "kd2_f_pre": 0.0112,        # 1/ms; Rate constant into desensitized state of presynaptic NMDAR; \
                                    # Modified from Erreger et al., 2005: 0.112 1/ms
        "kd2_b_pre": 9.1e-4,        # 1/ms; Rate constant out of desensitized state of presynaptic NMDAR; \
                                    # Erreger et al., 2005
        "kf_f_pre": 2.836,          # 1/ms; Rate constant for gating-associated conformational changes of \
                                    # presynaptic GluN1 subunit of NMDAR; Erreger et al., 2005; Clarke and Johnson, 2008
        "kf_b_pre": 0.175,          # 1/ms; Rate constant for gating-associated conformational changes of \
                                    # presynaptic GluN1 subunit of NMDAR; Erreger et al., 2005; Clarke and Johnson, 2008
        "kon_pre": 2.83e-3,         # 1/(uM ms); Rate constant for agonist (Glu) binding to presynaptic NMDAR; \
                                    # Erreger et al., 2005; Clarke and Johnson, 2008
        "koff_pre": 0.0381,         # 1/ms; Rate constant for agonist (Glu) unbinding from presynaptic NMDAR; \
                                    # Erreger et al., 2005; Clarke and Johnson, 2008
        "ks_f_0_pre": 0.048,        # 1/ms; Rate constant for gating-associated conformational changes of \
                                    # presynaptic GluN2 subunit of NMDAR; Erreger et al., 2005; Clarke and Johnson, 2008
        "ks_b_pre": 0.23,           # 1/ms; Rate constant for gating-associated conformational changes of \
                                    # presynaptic GluN2 subunit of NMDAR; Erreger et al., 2005; Clarke and Johnson, 2008

        # Glu release related parameters
        "C_thr_pre": 3,             # uM; Ca threshold concentration of Glu release in presynaptic neuron
        "CaN_max_pre": 2,           # uM; Total concentration of presynaptic CaN; Steuber and Willshaw, 2004
        "f_Glu_pre": 0.1,           # 1; Factor representing spillover of Glu from synaptic cleft that presynaptic \
                                    # NMDAR receives
        "G_pre": 1092,              # 1; Number of Glu per presynaptic vesicle; Riveros et al., 1986
        "k1_pre": 0.001,            # 1/(uM^3 ms); Rate constant for Ca activation of presynaptic CaN; \
                                    # Steuber and Willshaw, 2004
        "k2_pre": 0.002,            # 1/ms; Rate constant for inactivation of presynaptic CaN; \
                                    # Modified from Steuber and Willshaw, 2004: 0.012 1/ms
        "k_f_pre": 0.0075,          # 1/ms; Presynaptic facilitation rate constant
        "k_Glu_pre": 1e-6,          # 1; Presynaptic scaling factor to convert from units M to uM
        "k_recov_pre": 0.0075,      # 1/ms; Presynaptic recovery rate constant from empty to releasable state; \
                                    # Markram et al., 1998: 7.7e-3; Lee et al., 2009: 7.5e-3
        "KA_pre": 2,                # uM; Presynaptic CaN concentration producing half occupation
        "K_rel_pre": 5,             # uM; Presynaptic Ca concentration producing half occupation used in calculation \
                                    # of Glu release; Modified from Lee et al., 2009: 20 uM
        "n1_pre": 2,                # 1; Presynaptic Hill coefficient; Modified from Lee et al., 2009: 4
        "n2_pre": 2,                # 1; Presynaptic Hill coefficient
        "N_pre": 2,                 # 1; Number of readily releasable presynaptic vesicles; \
                                    # Schikorski and Stevens, 1997: 4.6 +/- 3.0
        "p1_pre": 3e-5,             # 1/ms; Rate constant for presynaptic protein activation affecting vesicular release
        "V_syncleft": 2e-18,        # l; Volume of synaptic cleft; \
                                    # Harris and Sultan, 1995: 0.7e-18 l - 8e-18 l; \
                                    # Schikorski and Stevens, 1997: 0.76e-3 um^3 = 0.76e-18 l
        "X_total_pre": 0.1          # uM; Total concentration of presynaptic protein affecting vesicular release
    }

    @staticmethod
    def get_parameters():
        p = dict(Pre.__DEFAULT_PARAMS)
        intermed = Pre.parameter_equations(p)
        p["c_Ca_pre"] = intermed["c_Ca_pre"]
        p["c_V_pre"] = intermed["c_V_pre"]

        return p

    @staticmethod
    def get_initial_values(p):
        x = dict(Pre.__DEFAULT_INITIAL_VALUES)
        intermed = Pre.intermediate_equations(p, x)
        x["h_CaNHVA_pre"] = intermed["h_inf_CaNHVA_pre"]
        x["h_Na_pre"] = intermed["h_inf_Na_pre"]
        x["m_CaNHVA_pre"] = intermed["m_inf_CaNHVA_pre"]
        x["m_Na_pre"] = intermed["m_inf_Na_pre"]
        x["n_K_pre"] = intermed["n_inf_K_pre"]
        x["n_K2_pre"] = intermed["n_inf_K2_pre"]
        x["s_Na_pre"] = intermed["s_inf_Na_pre"]

        return x

    @staticmethod
    def parameter_equations(p):

        # ----------------------
        # Presynaptic parameters
        # ----------------------

        # Safiulina et al., 2010
        c_Ca_pre = p["z"] * p["F"] * p["d_pre"] / p["k_Ca_pre"]
        c_V_pre = p["k_V_pre"] * p["R"] * (p["T_celsius"] + 273.15) / (p["z"] * p["F"])

        return {"c_Ca_pre": c_Ca_pre,
                "c_V_pre": c_V_pre
                }

    @staticmethod
    def intermediate_equations(p, x):

        # -------------------------------
        # Presynaptic algebraic equations
        # -------------------------------

        # K and Na channels in axon
        # Lavzin et al., 2012
        sigmas_Na_pre = 1 / (1 + exp((x["V_pre"] + p["V_sv_Na_pre"] + p["Vshift_pre"]) / p["V_sd_Na_pre"]))  # 1
        h_inf_Na_pre = 1 / (1 + exp((x["V_pre"] + 45 + p["Vshift_pre"]) / 3))   # 1
        m_inf_Na_pre = 1 / (1 + exp(-(x["V_pre"] + 40 + p["Vshift_pre"]) / 3))  # 1
        s_inf_Na_pre = 1 / (1 + exp((x["V_pre"] + 44 + p["Vshift_pre"]) / 3))   # 1
        n_inf_K_pre = 1 / (1 + exp(-(x["V_pre"] + 40 + p["Vshift_pre"]) / 3))   # 1
        n_inf_K2_pre = 1 / (1 + exp(-(x["V_pre"] + 40 + p["Vshift_pre"]) / 3))  # 1

        # N-type HVA Ca channel in axon
        # Safiulina et al., 2010
        alph_CaNHVA_pre = 1.6e-4 * exp(-x["V_pre"] / 48.4)          # 1/ms
        alpm_CaNHVA_pre = 0.1967 * (-x["V_pre"] + 19.88) / (exp((-x["V_pre"] + 19.88) / 10) - 1)            # 1/ms
        alpmt_CaNHVA_pre = exp(0.0378 * p["zm_pre"] * (x["V_pre"] - p["Vhalfm_pre"]))   # 1
        beth_CaNHVA_pre = 1 / (1 + exp((-x["V_pre"] + 39) / 10))    # 1/ms
        betm_CaNHVA_pre = 0.046 * exp(-x["V_pre"] / 20.73)          # 1/ms
        betmt_CaNHVA_pre = exp(0.0378 * p["zm_pre"] * p["gmm_pre"] * (x["V_pre"] - p["Vhalfm_pre"]))        # 1
        h_inf_CaNHVA_pre = alph_CaNHVA_pre / (alph_CaNHVA_pre + beth_CaNHVA_pre)        # 1
        m_inf_CaNHVA_pre = alpm_CaNHVA_pre / (alpm_CaNHVA_pre + betm_CaNHVA_pre)        # 1
        h_inf_CaNHVA2_pre = p["K_inh_pre"] / (p["K_inh_pre"] + x["Ca_CaNHVA_pre"])      # 1
        tau_m_CaNHVA_pre = max(p["tau_mmin_pre"] / (5 ** ((p["T_celsius"] - 25) / 10)),
                               betmt_CaNHVA_pre / (5 ** ((p["T_celsius"] - 25) / 10) *
                                                   p["a0m_pre"] * (1 + alpmt_CaNHVA_pre)))                  # ms

        return {"sigmas_Na_pre": sigmas_Na_pre,
                "h_inf_CaNHVA_pre": h_inf_CaNHVA_pre,
                "h_inf_CaNHVA2_pre": h_inf_CaNHVA2_pre,
                "h_inf_Na_pre": h_inf_Na_pre,
                "m_inf_CaNHVA_pre": m_inf_CaNHVA_pre,
                "m_inf_Na_pre": m_inf_Na_pre,
                "n_inf_K_pre": n_inf_K_pre,
                "n_inf_K2_pre": n_inf_K2_pre,
                "s_inf_Na_pre": s_inf_Na_pre,
                "tau_m_CaNHVA_pre": tau_m_CaNHVA_pre
                }

    def __init__(self, params, x0):
        print("Created new Pre")
        self.params = params    # Parameters
        self.x = x0             # Initial values of state variables

    def variable_names(self):
        return self.x.keys()

    def derivative(self, Glu_extsyn, Glu_syncleft, I_ext_pre):
        # ----------------------------------------
        # Presynaptic neuron inputs and parameters
        # ----------------------------------------
        p = self.params
        x = self.x

        # -------------------------------
        # Presynaptic algebraic equations
        # -------------------------------

        intermed = Pre.intermediate_equations(p, x)

        # Currents
        # N-type HVA Ca current
        # Safiulina et al., 2010
        nu_pre = x["V_pre"] / p["c_V_pre"]          # 1
        if abs(nu_pre) < 0.0001:
            e_pre = nu_pre / 2 - 1                  # 1
        else:
            e_pre = nu_pre / (1 - exp(nu_pre))      # 1
        f_ghk_pre = p["c_V_pre"] * (1 - (x["Ca_CaNHVA_pre"] / p["Ca_ext_pre"]) * exp(nu_pre)) * e_pre     # mV
        ICaNHVA_pre = p["gCaNHVA_pre"] * x["m_CaNHVA_pre"] ** 2 * x["h_CaNHVA_pre"] * intermed["h_inf_CaNHVA2_pre"] * \
                      f_ghk_pre             # uA/cm^2

        # K current
        # Lavzin et al., 2012
        IK_pre = p["gK_pre"] * x["n_K_pre"] ** 3 * (x["V_pre"] - p["VK_pre"]) + p["gK2_pre"] * x["n_K2_pre"] ** 3 * (
                x["V_pre"] - p["VK_pre"])   # uA/cm^2

        # Leak current
        # Lavzin et al., 2012
        IL_pre = p["gL_pre"] * (x["V_pre"] - p["VL_pre"])   # uA/cm^2

        # Na current
        # Lavzin et al., 2012
        INa_pre = p["gNa_pre"] * x["m_Na_pre"] ** 3 * x["h_Na_pre"] * x["s_Na_pre"] * (
                x["V_pre"] - p["VNa_pre"])  # uA/cm^2

        # Ca current via NMDAR
        # Schneggenburger et al., 1993; Burnashev et al., 1995
        if x["V_pre"] >= p["V_NMDAR_pre"]:
            ICa_NMDAR_pre = 0
        else:
            ICa_NMDAR_pre = 0.1 * p["gNMDAR_pre"] * x["RA2_O_pre"] * (x["V_pre"] - p["V_NMDAR_pre"])  # uA/cm^2

        # Na current via NMDAR
        # Schneggenburger et al., 1993; Burnashev et al., 1995
        if x["V_pre"] >= p["V_NMDAR_pre"]:
            INa_NMDAR_pre = 0
        else:
            INa_NMDAR_pre = 0.9 * p["gNMDAR_pre"] * x["RA2_O_pre"] * (x["V_pre"] - p["V_NMDAR_pre"])  # uA/cm^2

        # NMDAR

        # Total concentration of Glu that presynaptic NMDARs receives
        Glu_NMDAR_pre = p["f_Glu_pre"] * Glu_syncleft + Glu_extsyn

        # Rates dependent on V_pre
        # Ascher and Nowak, 1988
        kMg_f_pre = 6.1e-4 * exp(-x["V_pre"] / 17)  # 1/(uM ms); Rate of presynaptic NMDAR blocking by external Mg
        kMg_b_pre = 5.4 * exp(x["V_pre"] / 47)      # 1/ms; Rate of presynaptic NMDAR unblocking by external Mg

        # Clarke and Johnson, 2008
        ks_f_pre = p["ks_f_0_pre"] * exp((x["V_pre"] + 100) / 175)  # 1/ms; Rate of gating-associated conformational
        # changes of presynaptic GluN2 subunit of NMDAR

        # Erreger et al., 2005; Clarke and Johnson, 2008
        # No Mg
        # A + R <-> RA (2kon, koff)
        v_2kon_pre = 2 * p["kon_pre"] * Glu_NMDAR_pre * x["R_pre"]
        v_koff_pre = p["koff_pre"] * x["RA_pre"]

        # A + RA <-> RA2 (kon, 2koff)
        v_kon_pre = p["kon_pre"] * Glu_NMDAR_pre * x["RA_pre"]
        v_2koff_pre = 2 * p["koff_pre"] * x["RA2_pre"]

        # RA2 <-> RA2d1 (kd1_f, kd1_b)
        v_d1_f_pre = p["kd1_f_pre"] * x["RA2_pre"]
        v_d1_b_pre = p["kd1_b_pre"] * x["RA2d1_pre"]

        # RA2 <-> RA2d2 (kd2_f, kd2_b)
        v_d2_f_pre = p["kd2_f_pre"] * x["RA2_pre"]
        v_d2_b_pre = p["kd2_b_pre"] * x["RA2d2_pre"]

        # RA2 <-> RA2f (kf_f, kf_b)
        v_f1_f_pre = p["kf_f_pre"] * x["RA2_pre"]
        v_f1_b_pre = p["kf_b_pre"] * x["RA2f_pre"]

        # RA2 <-> RA2s (ks_f, ks_b)
        v_s1_f_pre = ks_f_pre * x["RA2_pre"]
        v_s1_b_pre = p["ks_b_pre"] * x["RA2s_pre"]

        # RA2f <-> RA2_O (ks_f, ks_b)
        v_s2_f_pre = ks_f_pre * x["RA2f_pre"]
        v_s2_b_pre = p["ks_b_pre"] * x["RA2_O_pre"]

        # RA2s <-> RA2_O (kf_f, kf_b)
        v_f2_f_pre = p["kf_f_pre"] * x["RA2s_pre"]
        v_f2_b_pre = p["kf_b_pre"] * x["RA2_O_pre"]

        # With Mg
        # A + RMg <-> RAMg (2kon, koff)
        v_2konMg_pre = 2 * p["kon_pre"] * Glu_NMDAR_pre * x["RMg_pre"]
        v_koffMg_pre = p["koff_pre"] * x["RAMg_pre"]

        # A + RAMg <-> RA2Mg (kon, 2koff)
        v_konMg_pre = p["kon_pre"] * Glu_NMDAR_pre * x["RAMg_pre"]
        v_2koffMg_pre = 2 * p["koff_pre"] * x["RA2Mg_pre"]

        # RA2Mg <-> RA2d1Mg (kd1_f, kd1_b)
        v_d1Mg_f_pre = p["kd1_f_pre"] * x["RA2Mg_pre"]
        v_d1Mg_b_pre = p["kd1_b_pre"] * x["RA2d1Mg_pre"]

        # RA2Mg <-> RA2d2Mg (kd2_f, kd2_b)
        v_d2Mg_f_pre = p["kd2_f_pre"] * x["RA2Mg_pre"]
        v_d2Mg_b_pre = p["kd2_b_pre"] * x["RA2d2Mg_pre"]

        # RA2Mg <-> RA2fMg (kf_f, kf_b)
        v_f1Mg_f_pre = p["kf_f_pre"] * x["RA2Mg_pre"]
        v_f1Mg_b_pre = p["kf_b_pre"] * x["RA2fMg_pre"]

        # RA2Mg <-> RA2sMg (ks_f, ks_b)
        v_s1Mg_f_pre = ks_f_pre * x["RA2Mg_pre"]
        v_s1Mg_b_pre = p["ks_b_pre"] * x["RA2sMg_pre"]

        # RA2fMg <-> RA2_OMg (ks_f, ks_b)
        v_s2Mg_f_pre = ks_f_pre * x["RA2fMg_pre"]
        v_s2Mg_b_pre = p["ks_b_pre"] * x["RA2_OMg_pre"]

        # RA2sMg <-> RA2_OMg (kf_f, kf_b)
        v_f2Mg_f_pre = p["kf_f_pre"] * x["RA2sMg_pre"]
        v_f2Mg_b_pre = p["kf_b_pre"] * x["RA2_OMg_pre"]

        # RA2_O <-> RA2_OMg (kMg_f, kMg_b)
        v_Mg_f_pre = kMg_f_pre * x["RA2_O_pre"]
        v_Mg_b_pre = kMg_b_pre * x["RA2_OMg_pre"]

        # --------------------------------------------
        # Differential equations in presynaptic neuron
        # --------------------------------------------
        return {
            # Membrane potential
            # Lavzin et al., 2012
            "V_pre": (-ICaNHVA_pre - IK_pre - INa_pre - IL_pre - ICa_NMDAR_pre - INa_NMDAR_pre + I_ext_pre) / \
                     p["Cm_pre"],

            # K and Na channels
            # Lavzin et al., 2012
            "h_Na_pre": (intermed["h_inf_Na_pre"] - x["h_Na_pre"]) / p["tau_h_Na_pre"],
            "m_Na_pre": (intermed["m_inf_Na_pre"] - x["m_Na_pre"]) / p["tau_m_Na_pre"],
            "n_K_pre": (intermed["n_inf_K_pre"] - x["n_K_pre"]) / p["tau_n_K_pre"],
            "n_K2_pre": (intermed["n_inf_K2_pre"] - x["n_K2_pre"]) / p["tau_n_K2_pre"],
            "s_Na_pre": (intermed["s_inf_Na_pre"] - x["s_Na_pre"]) / (
                    p["tau_s_Na_pre"] * intermed["sigmas_Na_pre"] + p["tau_sb_Na_pre"]),

            # N-type HVA Ca channel
            # Safiulina et al., 2010
            "h_CaNHVA_pre": (intermed["h_inf_CaNHVA_pre"] - x["h_CaNHVA_pre"]) / p["tau_h_CaNHVA_pre"],
            "m_CaNHVA_pre": (intermed["m_inf_CaNHVA_pre"] - x["m_CaNHVA_pre"]) / intermed["tau_m_CaNHVA_pre"],

            # NMDAR
            # Cull-Candy et al., 2001; Erreger et al., 2005; Clarke and Johnson, 2008
            "R_pre": -v_2kon_pre + v_koff_pre,
            "RA_pre": v_2kon_pre - v_koff_pre - v_kon_pre + v_2koff_pre,
            "RA2_pre": v_kon_pre - v_2koff_pre - v_d1_f_pre + v_d1_b_pre - v_d2_f_pre + v_d2_b_pre - \
                       v_f1_f_pre + v_f1_b_pre - v_s1_f_pre + v_s1_b_pre,
            "RA2d1_pre": v_d1_f_pre - v_d1_b_pre,
            "RA2d2_pre": v_d2_f_pre - v_d2_b_pre,
            "RA2f_pre": v_f1_f_pre - v_f1_b_pre - v_s2_f_pre + v_s2_b_pre,
            "RA2s_pre": v_s1_f_pre - v_s1_b_pre - v_f2_f_pre + v_f2_b_pre,
            "RA2_O_pre": v_s2_f_pre - v_s2_b_pre + v_f2_f_pre - v_f2_b_pre - v_Mg_f_pre + v_Mg_b_pre,
            "RMg_pre": -v_2konMg_pre + v_koffMg_pre,
            "RAMg_pre": v_2konMg_pre - v_koffMg_pre - v_konMg_pre + v_2koffMg_pre,
            "RA2Mg_pre": v_konMg_pre - v_2koffMg_pre - v_d1Mg_f_pre + v_d1Mg_b_pre - v_d2Mg_f_pre + v_d2Mg_b_pre - \
                         v_f1Mg_f_pre + v_f1Mg_b_pre - v_s1Mg_f_pre + v_s1Mg_b_pre,
            "RA2d1Mg_pre": v_d1Mg_f_pre - v_d1Mg_b_pre,
            "RA2d2Mg_pre": v_d2Mg_f_pre - v_d2Mg_b_pre,
            "RA2fMg_pre": v_f1Mg_f_pre - v_f1Mg_b_pre - v_s2Mg_f_pre + v_s2Mg_b_pre,
            "RA2sMg_pre": v_s1Mg_f_pre - v_s1Mg_b_pre - v_f2Mg_f_pre + v_f2Mg_b_pre,
            "RA2_OMg_pre": v_s2Mg_f_pre - v_s2Mg_b_pre + v_f2Mg_f_pre - v_f2Mg_b_pre + v_Mg_f_pre - v_Mg_b_pre,

            # Ca and Glu release
            # Badoual et al., 2008
            "Ca_CaNHVA_pre": -ICaNHVA_pre / p["c_Ca_pre"] + (p["Ca_rest_pre"] - x["Ca_CaNHVA_pre"]) / p["tau_Ca_pre"],
            "Ca_NMDAR_pre": -ICa_NMDAR_pre / p["c_Ca_pre"] + (p["Ca_rest_pre"] - x["Ca_NMDAR_pre"]) / p["tau_Ca_pre"],
            # Fiala et al., 1996
            "CaN_pre": p["k1_pre"] * (p["CaN_max_pre"] - x["CaN_pre"]) * x["Ca_NMDAR_pre"] ** 3 - \
                       p["k2_pre"] * x["CaN_pre"],
            # Modified from Tsodyks and Markram, 1997; Tsodyks et al., 1998; Lee et al., 2009;
            # De Pitta et al., 2011; De Pitta and Brunel, 2016
            "Prel_pre": -p["k_f_pre"] * x["Prel_pre"],              # Delta term added in solve_deltaf
            "Rrel_pre": p["k_recov_pre"] * (1 - x["Rrel_pre"]),     # Delta term subtracted in solve_deltaf

            "X_ac_pre": p["p1_pre"] * x["CaN_pre"] ** p["n2_pre"] / (
                    p["KA_pre"] ** p["n2_pre"] + x["CaN_pre"] ** p["n2_pre"]) * (p["X_total_pre"] - x["X_ac_pre"])
        }, {"Glu_NMDAR_pre": Glu_NMDAR_pre,
            "ICaNHVA_pre": ICaNHVA_pre,
            "ICa_NMDAR_pre": ICa_NMDAR_pre,
            "IK_pre": IK_pre,
            "IL_pre": IL_pre,
            "INa_pre": INa_pre,
            "INa_NMDAR_pre": INa_NMDAR_pre
            }

    def solve_deriv(self, deriv_pre, dt):
        for key in self.x:
            self.x[key] += deriv_pre[key] * dt

    def solve_deltaf(self, Ca_pre_old, f_pre, Prel_pre_old, Rrel_pre_old):

        # ----------------------------------------
        # Presynaptic neuron inputs and parameters
        # ----------------------------------------
        p = self.params
        x = self.x

        # -----------------------------------------------------
        # Updating those variables that include delta functions
        # -----------------------------------------------------

        # Modified from Tsodyks and Markram, 1997; Tsodyks et al., 1998; Lee et al., 2009; De Pitta et al., 2011;
        # De Pitta and Brunel, 2016
        x["Prel_pre"] += (1 - f_pre) * Ca_pre_old ** p["n1_pre"] / (
                p["K_rel_pre"] ** p["n1_pre"] + Ca_pre_old ** p["n1_pre"]) * (1 - Prel_pre_old)
        x["Rrel_pre"] -= x["Prel_pre"] * Rrel_pre_old

