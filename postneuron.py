# Simulation of tripartite synapse model
# Tiina Manninen and Ausra Saudargiene
# Reference: Tiina Manninen, Ausra Saudargiene, and Marja-Leena Linne. Astrocyte-mediated spike-timing-dependent
# long-term depression modulates synaptic properties in the developing cortex. PLoS Comput Biol, 2020.

# -----------------------------------------------------------------------

from math import exp, pi


# Postsynaptic neuron model includes the model components in the postsynaptic neuron and synaptic cleft only.
class Post:

    __DEFAULT_INITIAL_VALUES = {
        "AG_post": 0.0010453,               # uM; Postsynaptic 2-AG concentration
        "Ca_post": 0.049978,                # uM; Postsynaptic Ca concentration
        "Ca_ER_post": 62.9016,              # uM; Postsynaptic Ca concentration in ER
        "Ca_DAG_DAGL_post": 0.0052265,      # uM; Concentration of postsynaptic Ca-DAG-DAGL complex
        "Ca_DAG_GaGTP_PLC_post": 0,         # uM; Concentration of postsynaptic Ca-DAG-GaGTP-PLC complex
        "Ca_DAG_PLC_post": 8.8541e-5,       # uM; Concentration of postsynaptic Ca-DAG-PLC complex
        "Ca_DAGL_post": 0.27637,            # uM; Concentration of postsynaptic Ca-DAGL complex
        "Ca_GaGTP_PIP2_PLC_post": 0,        # uM; Concentration of postsynaptic Ca-GaGTP-PIP2-PLC complex
        "Ca_GaGTP_PLC_post": 0,             # uM; Concentration of postsynaptic Ca-GaGTP-PLC complex
        "Ca_PIP2_PLC_post": 0.00070833,     # uM; Concentration of postsynaptic Ca-PIP2-PLC complex
        "Ca_PLC_post": 0.00083161,          # uM; Concentration of postsynaptic Ca-PLC complex
        "DAG_post": 0.018912,               # uM; Postsynaptic DAG concentration
        "DAGL_post": 2.2119,                # uM; Postsynaptic DAGL concentration
        "Gabg_post": 3.5,                   # uM; Postsynaptic Gabg concentration; Kim et al., 2013
        "Gabg_Glu_mGluR_post": 0,           # uM; Concentration of postsynaptic Gabg-Glu-mGluR complex
        "GaGDP_post": 0,                    # uM; Postsynaptic GaGDP concentration
        "GaGTP_post": 0,                    # uM; Postsynaptic GaGTP concentration
        "GaGTP_PLC_post": 0,                # uM; Concentration of postsynaptic GaGTP-PLC complex
        "Glu_syncleft": 0,                  # uM; Concentration of Glu in synaptic cleft
        "Glu_mGluR_post": 0,                # uM; Concentration of postsynaptic Glu-mGluR complex
        "Glu_mGluRdesens_post": 0,          # uM; Concentration of postsynaptic Glu-mGluRdesens complex
        "h_CaLHVA_dend_post": 0.61966,      # 1; Gating variable for postsynaptic Ca_LHVA inactivation in the dendrite
        "h_CaLLVA_dend_post": 0.03206,      # 1; Gating variable for postsynaptic Ca_LLVA inactivation in the dendrite
        "h_IP3R_post": 0.97437,        	    # 1; Gating variable for postsynaptic IP3R inactivation
        "h_KA_dend_post": 0.96078,          # 1; Gating variable for postsynaptic K_A inactivation in the dendrite
        "h_Na_dend_post": 0.98073,          # 1; Gating variable for postsynaptic Na inactivation in the dendrite
        "h_Na_soma_post": 0.98059,          # 1; Gating variable for postsynaptic Na inactivation in the soma
        "IP3_post": 0.0017708,              # uM; Postsynaptic IP3 concentration
        "IP3deg_post": 0.014141,            # uM; Postsynaptic IP3deg concentration
        "IP3deg_PIKin_post": 0.017708,      # uM; Concentration of postsynaptic IP3deg-PIKin complex
        "m_AMPAR_post": 0,                  # 1; Fraction of postsynaptic AMPARs in open state
        "m_CaLHVA_dend_post": 7.0509e-5,    # 1; Gating variable for postsynaptic Ca_LHVA activation in the dendrite
        "m_CaLLVA_dend_post": 0.0090258,    # 1; Gating variable for postsynaptic Ca_LLVA activation in the dendrite
        "m_KA_dend_post": 0.035004,         # 1; Gating variable for postsynaptic K_A activation in the dendrite
        "m_KDR_soma_post": 0.022804,        # 1; Gating variable for postsynaptic K_DR activation in the soma
        "m_Na_dend_post": 0.0094356,        # 1; Gating variable for postsynaptic Na activation in the dendrite
        "m_Na_soma_post": 0.0095089,        # 1; Gating variable for postsynaptic Na activation in the soma
        "m_NMDAR_post": 0,                  # 1; Fraction of postsynaptic NMDARs in open state
        "mGluR_post": 5,                    # uM; Postsynaptic mGluR concentration; Kim et al., 2013
        "PIKin_post": 1.2523,               # uM; Postsynaptic PIKin concentration
        "PIP2_post": 49.6857,               # uM; Postsynaptic PIP2 concentration
        "PLC_post": 0.99837,                # uM; Postsynaptic PLC concentration
        "V_dend_post": -68.1916,            # mV; Postsynaptic membrane potential in the dendrite
        "V_soma_post": -68.1057,            # mV; Postsynaptic membrane potential in the soma
    }

    __DEFAULT_PARAMS = {
        # Basic parameters
        "F": 96485,                         # C/mol; Faraday constant
        "T_celsius": 36,                    # C; Temperature; Sarid et al., 2007
        "z": 2,                             # 1; Valence of Ca ion

        # Membrane capacitance
        "Cm_post": 3,                       # uF/cm^2; Postsynaptic membrane capacitance per unit area; \
                                            # Modified from Sarid et al., 2007: 2.7 uF/cm^2

        # Conductances
        "gAMPAR_post": 0.1,                 # mS/cm^2; Maximum conductance of postsynaptic AMPAR per unit area; \
                                            # Modified from Destexhe et al., 1998, Tewari and Majumdar, 2012: 3.5e-7 mS
        "gc_post": 2.1,                     # mS/cm^2; Coupling conductance between postsynaptic soma and dendrite per \
                                            # unit area; Pinsky and Rinzel, 1994
        "gCaLHVA_dend_post": 0.23,          # mS/cm^2; Maximum conductance of postsynaptic Ca_LHVA current in the \
                                            # dendrite per unit area; \
                                            # Modified from Reuveni et al., 1993, Markram et al., 2015: 0.01 mS/cm^2
        "gCaLLVA_dend_post": 0.23,          # mS/cm^2; Maximum conductance of postsynaptic Ca_LLVA current in the \
                                            # dendrite per unit area; Modified from Avery and Johnston, 1996: \
                                            # 0.01 mS/cm^2
        "gKA_dend_post": 1,                 # mS/cm^2; Maximum conductance of postsynaptic K_A current in the dendrite \
                                            # per unit area; Modified from Sarid et al., 2007: 0.001*x/100 S/cm^2
        "gKDR_soma_post": 50,               # mS/cm^2; Maximum conductance of postsynaptic K_DR current in the soma \
                                            # per unit area; Modified from Sarid et al., 2007: 154 mS/cm^2
        "gL_dend_post": 0.2,                # mS/cm^2; Leak conductance of postsynaptic dendrite per unit area; \
                                            # Modified from Pinsky and Rinzel, 1994: 0.1 mS/cm^2
        "gL_soma_post": 0.2,                # mS/cm^2; Leak conductance of postsynaptic soma per unit area; \
                                            # Modified from Sarid et al., 2007: 0.222 mS/cm^2; \
                                            # Pinsky and Rinzel, 1994: 0.1 mS/cm^2
        "gNa_dend_post": 0.06,              # mS/cm^2; Maximum conductance of postsynaptic Na current in the dendrite \
                                            # per unit area; Modified from Sarid et al., 2007: 1.78 mS/cm^2
        "gNa_soma_post": 60,                # mS/cm^2; Maximum conductance of postsynaptic Na current in the soma per \
                                            # unit area; Modified from Sarid et al., 2007: 178 mS/cm^2
        "gNaP_soma_post": 0.1,              # mS/cm^2; Maximum conductance of postsynaptic Na_P current in the soma \
                                            # per unit area; Modified from Sarid et al., 2007: 0.6 mS/cm^2
        "gNMDAR_post": 0.001,               # mS/cm^2; Maximum conductance of postsynaptic NMDAR per unit area; \
                                            # Modified from Destexhe et al., 1998: 1e-8 mS

        # Particle parameters
        "p_post": 0.5,                      # 1; Proportion of postsynaptic cell area taken by soma; \
                                            # Pinsky and Rinzel, 1994
        "tau_h_Na_post": 0.5,               # ms; Time constant for postsynaptic Na inactivation; Sarid et al., 2007
        "tau_m_KDR_post": 2,                # ms; Time constant for postsynaptic K_DR activation; Sarid et al., 2007
        "tau_m_Na_post": 0.05,              # ms; Time constant for postsynaptic Na activation; Sarid et al., 2007

        # Reversal potentials
        "V_AMPAR_post": 0,                  # mV; Reversal potential of postsynaptic AMPAR current; \
                                            # Koch, 2001; Destexhe et al., 1998
        "VCa_post": 90,                     # mV; Reversal potential of postsynaptic Ca_LHVA and Ca_LLVA currents \
                                            # Reuveni et al., 1993
        "VK_post": -85,                     # mV; Reversal potential of postsynaptic K current; \
                                            # Modified from Sarid et al., 2007: -77mV
        "VL_post": -70,                     # mV; Leak reversal potential of postsynaptic neuron; \
                                            # Modified from Sarid et al., 2007: -73.6mV
        "VNa_dend_post": 50,                # mV; Reversal potential of postsynaptic Na current in the dendrite; \
                                            # Sarid et al., 2007
        "VNa_soma_post": 50,                # mV; Reversal potential of postsynaptic Na current in the soma; \
                                            # Modified from Sarid et al., 2007: 90mV
        "V_NMDAR_post": 0,                  # mV; Reversal potential of postsynaptic NMDAR current; \
                                            # Destexhe et al., 1998

        # Other calcium related parameters
        "A_spine_post": 3.1416e-8,          # cm^2; Postsynaptic surface area of dendritic spine
        "B_post": 0.5,                      # 1; Postsynaptic fast buffering factor; \
                                            # Modified from Zachariou et al., 2013: 0.01
        "c_Ca_post": 6.4324,                # (uA ms)/(cm^2 uM); Postsynaptic scaling factor to convert from units \
                                            # uA/cm^2 to uM/ms; Zachariou et al., 2013
        "Ca_ext_post": 2015.1,              # uM; Ca concentration outside postsynaptic neuron; Kim et al., 2013
        "k_Ca_post": 1000,                  # 1; Postsynaptic scaling factor to convert from units mM/ms to uM/ms
        "K_act_post": 0.8,                  # uM; Postsynaptic IP3R dissociation constant of Ca (activation); \
                                            # Wagner et al., 2004
        "K_inh_post": 1.9,                  # uM; Postsynaptic IP3R dissociation constant of Ca (inhibition); \
                                            # Wagner et al., 2004
        "K_IP3_post": 0.15,                 # uM; Postsynaptic IP3R dissociation constant of IP3 (activation); \
                                            # Wagner et al., 2004
        "K_PMCA_post": 0.12,                # uM; Half-activation constant of postsynaptic PMCA pump; \
                                            # Politi et al., 2006
        "K_SERCA_post": 0.4,                # uM; Half-activation constant of postsynaptic SERCA pump; \
                                            # Wagner et al., 2004
        "r_ERcyt_post": 0.185,              # 1; Ratio between postsynaptic ER and cytosol volumes; \
                                            # De Young and Keizer, 1992
        "r_spine_post": 5e-5,               # cm; Postsynaptic radius of dendritic spine; 1um=1e-4cm; 0.5um=0.5e-4cm; \
                                            # Bourne and Harris, 2008
        "tau_IP3R_post": 2000,              # ms; Time constant for postsynaptic IP3R inactivation; Wagner et al., 2004
        "v_IP3R_post": 0.01,                # 1/ms; Maximum rate of Ca release via postsynaptic IP3R; \
                                            # Modified from Wagner et al., 2004: 0.0085 1/ms
        "v_PMCA_post": 8e-11,               # uMol/(ms cm^2); Maximum rate of Ca uptake by postsynaptic PMCA pump per \
                                            # unit area; Modified from Blackwell, 2002: 1e-9 uMol/(ms cm^2)
        "v_SERCA_post": 0.003,              # uM/ms; Maximum rate of Ca uptake by postsynaptic SERCA pump; \
                                            # De Young and Keizer, 1992: 9e-4 uM/ms
        "V_spine_post": 5.2360e-13,         # cm^3; Postsynaptic volume of dendritic spine

        # AMPAR parameters
        "alpha_AMPAR_post": 0.0011,         # 1/(uM ms); Rate constant of opening postsynaptic AMPAR; \
                                            # Destexhe et al., 1998
        "beta_AMPAR_post": 0.19,            # 1/ms; Rate constant of closing postsynaptic AMPAR; Destexhe et al., 1998

        # NMDAR parameters
        "alpha_NMDAR_post": 7.2e-5,         # 1/(uM ms); Rate constant of opening postsynaptic NMDAR; \
                                            # Destexhe et al., 1998
        "beta_NMDAR_post": 0.0066,          # 1/ms; Rate constant of closing postsynaptic NMDAR; Destexhe et al., 1998
        "Mg_ext_post": 1000,                # uM; Mg concentration outside postsynaptic neuron; Destexhe et al., 1998

        # mGluR -> 2-AG related parameters
        "k_Glu_f_post": 0.2,                # 1/ms; Modified from Kim et al., 2013: 0.002 1/ms
        "k_mGluR_f_post": 0.0001,           # 1/(uM ms); Kim et al., 2013
        "k_mGluR_b_post": 0.01,             # 1/ms; Kim et al., 2013
        "k_mGluR_des_f_post": 0.00025,      # 1/ms; Kim et al., 2013
        "k_mGluR_des_b_post": 1e-6,         # 1/ms; Kim et al., 2013
        "k_G_act_f_post": 0.015,            # 1/(uM ms); Kim et al., 2013
        "k_G_act_b_post": 0.0072,           # 1/ms; Kim et al., 2013
        "k_G_act_c_post": 0.0005,           # 1/ms; Kim et al., 2013
        "k_Ca_PLC1_f_post": 0.002,          # 1/(uM ms); Modified from Kim et al., 2013: 0.02 1/(uM ms)
        "k_Ca_PLC1_b_post": 0.12,           # 1/ms; Kim et al., 2013
        "k_G_PLC2_f_post": 0.1,             # 1/(uM ms); Kim et al., 2013
        "k_G_PLC2_b_post": 0.01,            # 1/ms; Kim et al., 2013
        "k_G_PLC1_f_post": 0.01,            # 1/(uM ms); Kim et al., 2013
        "k_G_PLC1_b_post": 0.012,           # 1/ms; Kim et al., 2013
        "k_Ca_PLC2_f_post": 0.08,           # 1/(uM ms); Kim et al., 2013
        "k_Ca_PLC2_b_post": 0.04,           # 1/ms; Kim et al., 2013
        "k_DAG1_f_post": 0.0006,            # 1/(uM ms); modelDB value 0.0006, article value Kim et al., 2013 is 0.006
        "k_DAG1_b_post": 0.01,              # 1/ms; Kim et al., 2013
        "k_DAG1_c_post": 0.025,             # 1/ms; Kim et al., 2013
        "k_DAG2_f_post": 0.2,               # 1/ms; Kim et al., 2013
        "k_DAG3_f_post": 0.015,             # 1/(uM ms); Kim et al., 2013
        "k_DAG3_b_post": 0.075,             # 1/ms; Kim et al., 2013
        "k_DAG3_c_post": 0.25,              # 1/ms; Kim et al., 2013
        "k_DAG4_f_post": 1,                 # 1/ms; Kim et al., 2013
        "k_degIP3_post": 0.01,              # 1/ms; Kim et al., 2013
        "k_PIP2_f_post": 0.002,             # 1/(uM ms);  Kim et al., 2013
        "k_PIP2_b_post": 0.001,             # 1/ms; Kim et al., 2013
        "k_PIP2_c_post": 0.001,             # 1/ms; Kim et al., 2013
        "k_GAP1_f_post": 0.03,              # 1/ms; Kim et al., 2013
        "k_GAP2_f_post": 0.03,              # 1/ms; Kim et al., 2013
        "k_hydrG_f_post": 0.001,            # 1/ms; Kim et al., 2013
        "k_regenG_f_post": 0.01,            # 1/ms; Kim et al., 2013
        "k_DAGL_f_post": 0.125,             # 1/(uM ms); Kim et al., 2013
        "k_DAGL_b_post": 0.05,              # 1/ms; Kim et al., 2013
        "k_prodAG_f_post": 0.0025,          # 1/(uM ms); Kim et al., 2013
        "k_prodAG_b_post": 0.0015,          # 1/ms; Kim et al., 2013
        "k_prodAG_c_post": 0.001,           # 1/ms; Kim et al., 2013
        "k_degAG_post": 0.005,              # 1/ms; Kim et al., 2013
        "k_degDAG_post": 0.00066,           # 1/ms; Politi et al., 2006; Zachariou et al., 2013
    }

    @staticmethod
    def get_parameters():
        p = dict(Post.__DEFAULT_PARAMS)
        intermed = Post.parameter_equations(p)
        p["A_spine_post"] = intermed["A_spine_post"]
        p["c_Ca_post"] = intermed["c_Ca_post"]
        p["V_spine_post"] = intermed["V_spine_post"]

        return p

    @staticmethod
    def get_initial_values(p):
        x = dict(Post.__DEFAULT_INITIAL_VALUES)
        intermed = Post.intermediate_equations(p, x)
        x["h_CaLHVA_dend_post"] = intermed["h_inf_CaLHVA_dend_post"]
        x["h_CaLLVA_dend_post"] = intermed["h_inf_CaLLVA_dend_post"]
        x["h_IP3R_post"] = intermed["h_inf_IP3R_post"]
        x["h_KA_dend_post"] = intermed["h_inf_KA_dend_post"]
        x["h_Na_dend_post"] = intermed["h_inf_Na_dend_post"]
        x["h_Na_soma_post"] = intermed["h_inf_Na_soma_post"]
        x["m_CaLHVA_dend_post"] = intermed["m_inf_CaLHVA_dend_post"]
        x["m_CaLLVA_dend_post"] = intermed["m_inf_CaLLVA_dend_post"]
        x["m_KA_dend_post"] = intermed["m_inf_KA_dend_post"]
        x["m_KDR_soma_post"] = intermed["m_inf_KDR_soma_post"]
        x["m_Na_dend_post"] = intermed["m_inf_Na_dend_post"]
        x["m_Na_soma_post"] = intermed["m_inf_Na_soma_post"]

        return x

    @staticmethod
    def parameter_equations(p):

        # -----------------------
        # Postsynaptic parameters
        # -----------------------

        A_spine_post = 4 * pi * p["r_spine_post"] ** 2
        V_spine_post = 4 / 3 * pi * p["r_spine_post"] ** 3
        c_Ca_post = p["z"] * p["F"] * V_spine_post / (p["B_post"] * A_spine_post)  # Zachariou et al., 2013

        return {"A_spine_post": A_spine_post,
                "c_Ca_post": c_Ca_post,
                "V_spine_post": V_spine_post
                }

    @staticmethod
    def intermediate_equations(p, x):

        # --------------------------------
        # Postsynaptic algebraic equations
        # --------------------------------

        # K and Na channels in soma
        # Sarid et al., 2007
        h_inf_Na_soma_post = 1 / (1 + exp((x["V_soma_post"] + 23) / 11.5))      # 1
        m_inf_Na_soma_post = 1 / (1 + exp(-(x["V_soma_post"] + 17) / 11))       # 1
        m_inf_KDR_soma_post = 1 / (1 + exp(-(x["V_soma_post"] + 17) / 13.6))    # 1
        n_inf_NaP_soma_post = 1 / (1 + exp(-(x["V_soma_post"] + 50) / 6))       # 1

        # A-type K and Na channels in dendrite
        # Sarid et al., 2007
        h_inf_KA_dend_post = 1 / (1 + exp((x["V_dend_post"] + 49) / 6))         # 1
        m_inf_KA_dend_post = 1 / (1 + exp(-(x["V_dend_post"] + 40) / 8.5))      # 1
        h_inf_Na_dend_post = 1 / (1 + exp((x["V_dend_post"] + 23) / 11.5))      # 1
        m_inf_Na_dend_post = 1 / (1 + exp(-(x["V_dend_post"] + 17) / 11))       # 1

        # L-type HVA Ca channel in dendrite
        # Reuveni et al., 1993; Markram et al., 2015
        alph_CaLHVA_post = 0.000457 * exp(-(x["V_dend_post"] + 13) / 50)        # 1/ms
        alpm_CaLHVA_post = -0.055 * (x["V_dend_post"] + 27) / (exp(-(x["V_dend_post"] + 27) / 3.8) - 1)  # 1/ms

        beth_CaLHVA_post = 0.0065 / (1 + exp(-(x["V_dend_post"] + 15) / 28))    # 1/ms
        betm_CaLHVA_post = 0.94 * exp(-(x["V_dend_post"] + 75) / 17)            # 1/ms

        h_inf_CaLHVA_dend_post = alph_CaLHVA_post / (alph_CaLHVA_post + beth_CaLHVA_post)   # 1
        m_inf_CaLHVA_dend_post = alpm_CaLHVA_post / (alpm_CaLHVA_post + betm_CaLHVA_post)   # 1

        # L-type LVA Ca channel in dendrite
        # Avery and Johnston, 1996
        h_inf_CaLLVA_dend_post = 1 / (1 + exp((x["V_dend_post"] + 10 + 80) / 6.4))          # 1
        m_inf_CaLLVA_dend_post = 1 / (1 + exp(-(x["V_dend_post"] + 10 + 30) / 6))           # 1

        # De Young and Keizer, 1992; Li and Rinzel, 1994; Wagner et al., 2004
        h_inf_IP3R_post = p["K_inh_post"] / (p["K_inh_post"] + x["Ca_post"])    # 1

        return {"alph_CaLHVA_post": alph_CaLHVA_post,
                "alpm_CaLHVA_post": alpm_CaLHVA_post,
                "beth_CaLHVA_post": beth_CaLHVA_post,
                "betm_CaLHVA_post": betm_CaLHVA_post,
                "h_inf_CaLHVA_dend_post": h_inf_CaLHVA_dend_post,
                "h_inf_CaLLVA_dend_post": h_inf_CaLLVA_dend_post,
                "h_inf_IP3R_post": h_inf_IP3R_post,
                "h_inf_KA_dend_post": h_inf_KA_dend_post,
                "h_inf_Na_dend_post": h_inf_Na_dend_post,
                "h_inf_Na_soma_post": h_inf_Na_soma_post,
                "m_inf_CaLHVA_dend_post": m_inf_CaLHVA_dend_post,
                "m_inf_CaLLVA_dend_post": m_inf_CaLLVA_dend_post,
                "m_inf_KA_dend_post": m_inf_KA_dend_post,
                "m_inf_KDR_soma_post": m_inf_KDR_soma_post,
                "m_inf_Na_dend_post": m_inf_Na_dend_post,
                "m_inf_Na_soma_post": m_inf_Na_soma_post,
                "n_inf_NaP_soma_post": n_inf_NaP_soma_post
                }

    def __init__(self, params, x0):
        print("Created new Post")
        self.params = params    # Parameters
        self.x = x0             # Initial values of state variables

    def variable_names(self):
        return self.x.keys()

    def derivative(self, f_Glu_pre, I_ext_post, r_leakCell_post, r_leakER_post):
        # -----------------------------------------
        # Postsynaptic neuron inputs and parameters
        # -----------------------------------------
        p = self.params
        x = self.x

        # --------------------------------
        # Postsynaptic algebraic equations
        # --------------------------------

        intermed = Post.intermediate_equations(p, x)

        # Time constants
        if x["V_dend_post"] < -63:
            tau_h_KA_post = (1 / (exp((x["V_dend_post"] + 46) / 5) + exp(-(x["V_dend_post"] + 238) / 37))) / (
                    3 ** ((p["T_celsius"] - 23.5) / 10))                # ms; Sarid et al., 2007
        else:
            tau_h_KA_post = 19 / (3 ** ((p["T_celsius"] - 23.5) / 10))  # ms; Huguenard and McCormick, 1992

        tau_m_KA_post = (1 / (exp((x["V_dend_post"] + 36) / 20) + exp(-(x["V_dend_post"] + 80) / 13)) + 0.37) / (
                3 ** ((p["T_celsius"] - 23.5) / 10))                    # ms; Sarid et al., 2007

        # Reuveni et al., 1993; Markram et al., 2015
        tau_h_CaLHVA_post = 1 / (intermed["alph_CaLHVA_post"] + intermed["beth_CaLHVA_post"])   # ms
        tau_m_CaLHVA_post = 1 / (intermed["alpm_CaLHVA_post"] + intermed["betm_CaLHVA_post"])   # ms

        # Avery and Johnston, 1996
        tau_h_CaLLVA_post = (20 + 50 / (1 + exp((x["V_dend_post"] + 10 + 40) / 7))) / (
                2.3 ** ((p["T_celsius"] - 21) / 10))   # ms
        tau_m_CaLLVA_post = (5 + 20 / (1 + exp((x["V_dend_post"] + 10 + 25) / 5))) / (
                2.3 ** ((p["T_celsius"] - 21) / 10))   # ms

        # Currents
        # Somatic currents
        # Somatic delayed rectifier K current
        # Sarid et al., 2007
        IKDR_soma_post = p["gKDR_soma_post"] * x["m_KDR_soma_post"] ** 2 * (x["V_soma_post"] - p["VK_post"])  # uA/cm^2

        # Somatic leak current
        # Sarid et al., 2007
        IL_soma_post = p["gL_soma_post"] * (x["V_soma_post"] - p["VL_post"])  # uA/cm^2

        # Somatic Na current
        # Sarid et al., 2007
        INa_soma_post = p["gNa_soma_post"] * x["m_Na_soma_post"] ** 2 * x["h_Na_soma_post"] * (
                x["V_soma_post"] - p["VNa_soma_post"])  # uA/cm^2

        # Somatic persistent Na current
        # Sarid et al., 2007
        INaP_soma_post = p["gNaP_soma_post"] * intermed["n_inf_NaP_soma_post"] * (
                x["V_soma_post"] - p["VNa_soma_post"])  # uA/cm^2

        # Dendritic currents
        # Dendritic A-type K current
        # Sarid et al., 2007
        IKA_dend_post = p["gKA_dend_post"] * x["m_KA_dend_post"] ** 4 * x["h_KA_dend_post"] * (
                x["V_dend_post"] - p["VK_post"])  # uA/cm^2

        # Dendritic leak current
        # Sarid et al., 2007
        IL_dend_post = p["gL_dend_post"] * (x["V_dend_post"] - p["VL_post"])  # uA/cm^2

        # Dendritic Na current
        # Sarid et al., 2007
        INa_dend_post = p["gNa_dend_post"] * x["m_Na_dend_post"] ** 2 * x["h_Na_dend_post"] * (
                x["V_dend_post"] - p["VNa_dend_post"])  # uA/cm^2

        # Dendritic AMPAR current
        # Destexhe et al., 1998; Tewari and Majumdar, 2012
        I_AMPAR_post = p["gAMPAR_post"] * x["m_AMPAR_post"] * (x["V_dend_post"] - p["V_AMPAR_post"])  # uA/cm^2

        # Coupling currents
        # Sarid et al., 2007
        # Dendritic current coupling term
        Icoupl_dend_post = p["gc_post"] / (1 - p["p_post"]) * (x["V_soma_post"] - x["V_dend_post"])  # uA/cm^2
        # Somatic current coupling term
        Icoupl_soma_post = p["gc_post"] / p["p_post"] * (x["V_dend_post"] - x["V_soma_post"])  # uA/cm^2

        # Ca fluxes in postsynaptic spine
        Ca_flux_post = Post.calcium_other_fluxes(self)
        Ca_leak_post = Post.calcium_leak_fluxes(self, r_leakCell_post, r_leakER_post)

        # Postsynaptic mGluR -> 2-AG reaction rates
        # Kim et al., 2013

        # Glu_syncleft -> empty (k_Glu_f)
        v_Glu_f_post = p["k_Glu_f_post"] * (1 - f_Glu_pre) * x["Glu_syncleft"]

        # Glu_syncleft + mGluR <-> Glu_mGluR (k_mGluR_f, k_mGluR_b)
        v_mGluR_f_post = p["k_mGluR_f_post"] * (1 - f_Glu_pre) * x["Glu_syncleft"] * x["mGluR_post"]
        v_mGluR_b_post = p["k_mGluR_b_post"] * x["Glu_mGluR_post"]

        # Glu_mGluR <-> Glu_mGluRdesens (k_mGluR_des_f, k_mGluR_des_b)
        v_mGluR_des_f_post = p["k_mGluR_des_f_post"] * x["Glu_mGluR_post"]
        v_mGluR_des_b_post = p["k_mGluR_des_b_post"] * x["Glu_mGluRdesens_post"]

        # Gabg + Glu_mGluR <-> Gabg_Glu_mGluR -> Glu_mGluR + GaGTP (k_G_act_f, k_G_act_b, k_G_act_c)
        v_G_act_f_post = p["k_G_act_f_post"] * x["Gabg_post"] * x["Glu_mGluR_post"]
        v_G_act_b_post = p["k_G_act_b_post"] * x["Gabg_Glu_mGluR_post"]
        v_G_act_c_post = p["k_G_act_c_post"] * x["Gabg_Glu_mGluR_post"]

        # Ca + PLC <-> Ca_PLC (k_Ca_PLC1_f, k_Ca_PLC1_b)
        v_Ca_PLC1_f_post = p["k_Ca_PLC1_f_post"] * x["Ca_post"] * x["PLC_post"]
        v_Ca_PLC1_b_post = p["k_Ca_PLC1_b_post"] * x["Ca_PLC_post"]

        # GaGTP + Ca_PLC <-> Ca_GaGTP_PLC (k_G_PLC2_f, k_G_PLC2_b)
        v_G_PLC2_f_post = p["k_G_PLC2_f_post"] * x["GaGTP_post"] * x["Ca_PLC_post"]
        v_G_PLC2_b_post = p["k_G_PLC2_b_post"] * x["Ca_GaGTP_PLC_post"]

        # GaGTP + PLC <-> GaGTP_PLC (k_G_PLC1_f, k_G_PLC1_b)
        v_G_PLC1_f_post = p["k_G_PLC1_f_post"] * x["GaGTP_post"] * x["PLC_post"]
        v_G_PLC1_b_post = p["k_G_PLC1_b_post"] * x["GaGTP_PLC_post"]

        # Ca + GaGTP_PLC <-> Ca_GaGTP_PLC (k_Ca_PLC2_f, k_Ca_PLC2_b)
        v_Ca_PLC2_f_post = p["k_Ca_PLC2_f_post"] * x["Ca_post"] * x["GaGTP_PLC_post"]
        v_Ca_PLC2_b_post = p["k_Ca_PLC2_b_post"] * x["Ca_GaGTP_PLC_post"]

        # PIP2 + Ca_PLC <-> Ca_PIP2_PLC -> Ca_DAG_PLC + IP3 (k_DAG1_f, k_DAG1_b, k_DAG1_c)
        v_DAG1_f_post = p["k_DAG1_f_post"] * x["PIP2_post"] * x["Ca_PLC_post"]
        v_DAG1_b_post = p["k_DAG1_b_post"] * x["Ca_PIP2_PLC_post"]
        v_DAG1_c_post = p["k_DAG1_c_post"] * x["Ca_PIP2_PLC_post"]

        # Ca_DAG_PLC -> Ca_PLC + DAG (k_DAG2_f)
        v_DAG2_f_post = p["k_DAG2_f_post"] * x["Ca_DAG_PLC_post"]

        # PIP2 + Ca_GaGTP_PLC <-> Ca_GaGTP_PIP2_PLC -> Ca_DAG_GaGTP_PLC + IP3 (k_DAG3_f, k_DAG3_b, k_DAG3_c)
        v_DAG3_f_post = p["k_DAG3_f_post"] * x["Ca_GaGTP_PLC_post"] * x["PIP2_post"]
        v_DAG3_b_post = p["k_DAG3_b_post"] * x["Ca_GaGTP_PIP2_PLC_post"]
        v_DAG3_c_post = p["k_DAG3_c_post"] * x["Ca_GaGTP_PIP2_PLC_post"]

        # Ca_DAG_GaGTP_PLC -> Ca_GaGTP_PLC + DAG (k_DAG4_f)
        v_DAG4_f_post = p["k_DAG4_f_post"] * x["Ca_DAG_GaGTP_PLC_post"]

        # IP3 <-> IP3deg (k_degIP3)
        v_degIP3_post = p["k_degIP3_post"] * x["IP3_post"]

        # IP3deg + PIKin <-> IP3deg_PIKin -> PIP2 + PIKin (k_PIP2_f, k_PIP2_b, k_PIP2_c)
        v_PIP2_f_post = p["k_PIP2_f_post"] * x["IP3deg_post"] * x["PIKin_post"]
        v_PIP2_b_post = p["k_PIP2_b_post"] * x["IP3deg_PIKin_post"]
        v_PIP2_c_post = p["k_PIP2_c_post"] * x["IP3deg_PIKin_post"]

        # GaGTP_PLC -> PLC + GaGDP (k_GAP1_f)
        v_GAP1_f_post = p["k_GAP1_f_post"] * x["GaGTP_PLC_post"]

        # Ca_GaGTP_PLC -> Ca_PLC + GaGDP (k_GAP2_f)
        v_GAP2_f_post = p["k_GAP2_f_post"] * x["Ca_GaGTP_PLC_post"]

        # GaGTP -> GaGDP (k_hydrG_f)
        v_hydrG_f_post = p["k_hydrG_f_post"] * x["GaGTP_post"]

        # GaGDP -> Gabg (k_regenG_f)
        v_regenG_f_post = p["k_regenG_f_post"] * x["GaGDP_post"]

        # Ca + DAGL <-> Ca_DAGL (k_DAGL_f, k_DAGL_b)
        v_DAGL_f_post = p["k_DAGL_f_post"] * x["Ca_post"] * x["DAGL_post"]
        v_DAGL_b_post = p["k_DAGL_b_post"] * x["Ca_DAGL_post"]

        # DAG + Ca_DAGL <-> Ca_DAG_DAGL -> Ca_DAGL + 2-AG (k_prodAG_f, k_prodAG_b, k_prodAG_c)
        v_prodAG_f_post = p["k_prodAG_f_post"] * x["DAG_post"] * x["Ca_DAGL_post"]
        v_prodAG_b_post = p["k_prodAG_b_post"] * x["Ca_DAG_DAGL_post"]
        v_prodAG_c_post = p["k_prodAG_c_post"] * x["Ca_DAG_DAGL_post"]

        # 2-AG -> empty (k_degAG)
        v_degAG_post = p["k_degAG_post"] * x["AG_post"]

        # DAG -> empty (k_degDAG)
        # Zachariou et al., 2013
        v_degDAG_post = p["k_degDAG_post"] * x["DAG_post"]

        # ---------------------------------------------
        # Differential equations in postsynaptic neuron
        # ---------------------------------------------
        return {
                # Membrane potential
                # Pinsky and Rinzel, 1994
                "V_dend_post": (-IKA_dend_post - Ca_flux_post["ICaLHVA_dend_post"] - Ca_flux_post["ICaLLVA_dend_post"] -
                                INa_dend_post - IL_dend_post - I_AMPAR_post - Ca_flux_post["ICa_NMDAR_post"] +
                                Icoupl_dend_post) / p["Cm_post"],
                "V_soma_post": (-IKDR_soma_post - INa_soma_post - INaP_soma_post - IL_soma_post + Icoupl_soma_post +
                                I_ext_post) / p["Cm_post"],

                # K and Na channels in soma
                # Sarid et al., 2007
                "h_Na_soma_post": (intermed["h_inf_Na_soma_post"] - x["h_Na_soma_post"]) / p["tau_h_Na_post"],
                "m_Na_soma_post": (intermed["m_inf_Na_soma_post"] - x["m_Na_soma_post"]) / p["tau_m_Na_post"],
                "m_KDR_soma_post": (intermed["m_inf_KDR_soma_post"] - x["m_KDR_soma_post"]) / p["tau_m_KDR_post"],

                # K and Na channels in dendrite
                # Sarid et al., 2007
                "h_KA_dend_post": (intermed["h_inf_KA_dend_post"] - x["h_KA_dend_post"]) / tau_h_KA_post,
                "m_KA_dend_post": (intermed["m_inf_KA_dend_post"] - x["m_KA_dend_post"]) / tau_m_KA_post,
                "h_Na_dend_post": (intermed["h_inf_Na_dend_post"] - x["h_Na_dend_post"]) / p["tau_h_Na_post"],
                "m_Na_dend_post": (intermed["m_inf_Na_dend_post"] - x["m_Na_dend_post"]) / p["tau_m_Na_post"],

                # Ca channels in dendrite
                # Reuveni et al., 1993; Avery and Johnston, 1996; Markram et al., 2015
                "h_CaLHVA_dend_post": (intermed["h_inf_CaLHVA_dend_post"] - x["h_CaLHVA_dend_post"]) / \
                                      tau_h_CaLHVA_post,
                "m_CaLHVA_dend_post": (intermed["m_inf_CaLHVA_dend_post"] - x["m_CaLHVA_dend_post"]) / \
                                      tau_m_CaLHVA_post,
                "h_CaLLVA_dend_post": (intermed["h_inf_CaLLVA_dend_post"] - x["h_CaLLVA_dend_post"]) / \
                                      tau_h_CaLLVA_post,
                "m_CaLLVA_dend_post": (intermed["m_inf_CaLLVA_dend_post"] - x["m_CaLLVA_dend_post"]) / \
                                      tau_m_CaLLVA_post,

                # AMPAR and NMDAR
                # Destexhe et al., 1998
                "m_AMPAR_post": p["alpha_AMPAR_post"] * (1 - f_Glu_pre) * x["Glu_syncleft"] * (
                        1 - x["m_AMPAR_post"]) - p["beta_AMPAR_post"] * x["m_AMPAR_post"],
                "m_NMDAR_post": p["alpha_NMDAR_post"] * (1 - f_Glu_pre) * x["Glu_syncleft"] * (
                        1 - x["m_NMDAR_post"]) - p["beta_NMDAR_post"] * x["m_NMDAR_post"],

                # mGluR -> 2-AG
                # Mostly Kim et al., 2013, but added model components from De Young and Keizer, 1992;
                # Li and Rinzel, 1994; Blackwell, 2002; Zachariou et al., 2013
                "AG_post": v_prodAG_c_post - v_degAG_post,
                "Ca_post": -v_Ca_PLC1_f_post + v_Ca_PLC1_b_post - v_Ca_PLC2_f_post + v_Ca_PLC2_b_post - \
                           v_DAGL_f_post + v_DAGL_b_post + Ca_flux_post["J_IP3R_post"] - \
                           Ca_flux_post["J_SERCA_post"] + Ca_leak_post["J_leakER_post"] + \
                           Ca_flux_post["J_CaL_post"] + Ca_flux_post["J_NMDAR_post"] - \
                           Ca_flux_post["J_PMCA_post"] + Ca_leak_post["J_leakCell_post"],
                "Ca_ER_post": (-Ca_flux_post["J_IP3R_post"] + Ca_flux_post["J_SERCA_post"] -
                               Ca_leak_post["J_leakER_post"]) / p["r_ERcyt_post"],
                "Ca_DAG_DAGL_post": v_prodAG_f_post - v_prodAG_b_post - v_prodAG_c_post,
                "Ca_DAG_GaGTP_PLC_post": v_DAG3_c_post - v_DAG4_f_post,
                "Ca_DAG_PLC_post": v_DAG1_c_post - v_DAG2_f_post,
                "Ca_DAGL_post": v_DAGL_f_post - v_DAGL_b_post - v_prodAG_f_post + v_prodAG_b_post + v_prodAG_c_post,
                "Ca_GaGTP_PIP2_PLC_post": v_DAG3_f_post - v_DAG3_b_post - v_DAG3_c_post,
                "Ca_GaGTP_PLC_post": v_G_PLC2_f_post - v_G_PLC2_b_post + v_Ca_PLC2_f_post - v_Ca_PLC2_b_post - \
                                    v_DAG3_f_post + v_DAG3_b_post + v_DAG4_f_post - v_GAP2_f_post,
                "Ca_PIP2_PLC_post": v_DAG1_f_post - v_DAG1_b_post - v_DAG1_c_post,
                "Ca_PLC_post": v_Ca_PLC1_f_post - v_Ca_PLC1_b_post - v_G_PLC2_f_post + v_G_PLC2_b_post - \
                               v_DAG1_f_post + v_DAG1_b_post + v_DAG2_f_post + v_GAP2_f_post,
                "DAG_post": v_DAG2_f_post + v_DAG4_f_post - v_prodAG_f_post + v_prodAG_b_post - v_degDAG_post,
                "DAGL_post": -v_DAGL_f_post + v_DAGL_b_post,
                "Gabg_post": -v_G_act_f_post + v_G_act_b_post + v_regenG_f_post,
                "Gabg_Glu_mGluR_post": v_G_act_f_post - v_G_act_b_post - v_G_act_c_post,
                "GaGDP_post": v_GAP1_f_post + v_GAP2_f_post + v_hydrG_f_post - v_regenG_f_post,
                "GaGTP_post": v_G_act_c_post - v_G_PLC2_f_post + v_G_PLC2_b_post - v_G_PLC1_f_post + \
                              v_G_PLC1_b_post - v_hydrG_f_post,
                "GaGTP_PLC_post": v_G_PLC1_f_post - v_G_PLC1_b_post - v_Ca_PLC2_f_post + v_Ca_PLC2_b_post - \
                                  v_GAP1_f_post,
                "Glu_syncleft": - v_Glu_f_post - v_mGluR_f_post + v_mGluR_b_post,   # Delta term added in solve_deltaf
                "Glu_mGluR_post": v_mGluR_f_post - v_mGluR_b_post - v_mGluR_des_f_post + v_mGluR_des_b_post - \
                                  v_G_act_f_post + v_G_act_b_post + v_G_act_c_post,
                "Glu_mGluRdesens_post": v_mGluR_des_f_post - v_mGluR_des_b_post,
                "h_IP3R_post": (intermed["h_inf_IP3R_post"] - x["h_IP3R_post"]) / p["tau_IP3R_post"],
                "IP3_post": v_DAG1_c_post + v_DAG3_c_post - v_degIP3_post,
                "IP3deg_post": v_degIP3_post - v_PIP2_f_post + v_PIP2_b_post,
                "IP3deg_PIKin_post": v_PIP2_f_post - v_PIP2_b_post - v_PIP2_c_post,
                "mGluR_post": -v_mGluR_f_post + v_mGluR_b_post,
                "PIKin_post": -v_PIP2_f_post + v_PIP2_b_post + v_PIP2_c_post,
                "PIP2_post": -v_DAG1_f_post + v_DAG1_b_post - v_DAG3_f_post + v_DAG3_b_post + v_PIP2_c_post,
                "PLC_post": -v_Ca_PLC1_f_post + v_Ca_PLC1_b_post - v_G_PLC1_f_post + v_G_PLC1_b_post + v_GAP1_f_post

               }, {"I_AMPAR_post": I_AMPAR_post,
                   "ICaLHVA_dend_post": Ca_flux_post["ICaLHVA_dend_post"],
                   "ICaLLVA_dend_post": Ca_flux_post["ICaLLVA_dend_post"],
                   "ICa_NMDAR_post": Ca_flux_post["ICa_NMDAR_post"],
                   "Icoupl_dend_post": Icoupl_dend_post,
                   "Icoupl_soma_post": Icoupl_soma_post,
                   "IKA_dend_post": IKA_dend_post,
                   "IKDR_soma_post": IKDR_soma_post,
                   "IL_dend_post": IL_dend_post,
                   "IL_soma_post": IL_soma_post,
                   "INa_dend_post": INa_dend_post,
                   "INa_soma_post": INa_soma_post,
                   "INaP_soma_post": INaP_soma_post,
                   "J_CaL_post": Ca_flux_post["J_CaL_post"],
                   "J_IP3R_post": Ca_flux_post["J_IP3R_post"],
                   "J_leakCell_post": Ca_leak_post["J_leakCell_post"],
                   "J_leakER_post": Ca_leak_post["J_leakER_post"],
                   "J_NMDAR_post": Ca_flux_post["J_NMDAR_post"],
                   "J_PMCA_post":  Ca_flux_post["J_PMCA_post"],
                   "J_SERCA_post": Ca_flux_post["J_SERCA_post"]
                   }

    def calcium_leak_parameters(self, J_CaL_post, J_IP3R_post, J_NMDAR_post, J_PMCA_post, J_SERCA_post):
        # -----------------------------------------
        # Postsynaptic neuron inputs and parameters
        # -----------------------------------------
        p = self.params
        x = self.x

        # ---------------------------------
        # Adjusting Ca leak flux parameters
        # ---------------------------------

        # r_leakER_post adjusted such that net Ca flux across the postsynaptic ER membrane
        # is zero at the resting Ca concentration (Blackwell 2002)
        # (J_SERCA_post-J_IP3R_post-J_leakER_post)=0
        r_leakER_post = (J_SERCA_post - J_IP3R_post) / (x["Ca_ER_post"] - x["Ca_post"])
        print(r_leakER_post)

        # r_leakCell_post adjusted such that net Ca flux across the postsynaptic plasma membrane
        # is zero at the basal Ca value (Blackwell 2002)
        # (J_PMCA_post-J_CaL_post-J_NMDAR_post-J_leakCell_post)=0
        r_leakCell_post = (J_PMCA_post - J_CaL_post - J_NMDAR_post) / (p["Ca_ext_post"] - x["Ca_post"])
        print(r_leakCell_post)

        return {"r_leakCell_post": r_leakCell_post,
                "r_leakER_post": r_leakER_post
                }

    def calcium_leak_fluxes(self, r_leakCell_post, r_leakER_post):
        # -----------------------------------------
        # Postsynaptic neuron inputs and parameters
        # -----------------------------------------
        p = self.params
        x = self.x

        # ---------------------------
        # Postsynaptic Ca leak fluxes
        # ---------------------------

        # De Young and Keizer, 1992; Li and Rinzel, 1994
        J_leakER_post = r_leakER_post * (x["Ca_ER_post"] - x["Ca_post"])  # uM/ms

        # Blackwell, 2002
        J_leakCell_post = r_leakCell_post * (p["Ca_ext_post"] - x["Ca_post"])  # uM/ms

        return{"J_leakCell_post": J_leakCell_post,
               "J_leakER_post": J_leakER_post
               }

    def calcium_other_fluxes(self):
        # -----------------------------------------
        # Postsynaptic neuron inputs and parameters
        # -----------------------------------------
        p = self.params
        x = self.x

        # ---------------------------------------------
        # Postsynaptic Ca fluxes other than leak fluxes
        # ---------------------------------------------

        # Dendritic L-type HVA Ca current
        # Reuveni et al., 1993; Markram et al., 2015
        ICaLHVA_dend_post = p["gCaLHVA_dend_post"] * x["m_CaLHVA_dend_post"] ** 2 * x["h_CaLHVA_dend_post"] * (
                x["V_dend_post"] - p["VCa_post"])  # uA/cm^2

        # Dendritic L-type LVA Ca current
        # Avery and Johnston, 1996
        ICaLLVA_dend_post = p["gCaLLVA_dend_post"] * x["m_CaLLVA_dend_post"] ** 2 * x["h_CaLLVA_dend_post"] * (
                x["V_dend_post"] - p["VCa_post"])  # uA/cm^2

        # Dendritic L-type Ca flux
        J_CaL_post = -(ICaLHVA_dend_post + ICaLLVA_dend_post) / p["c_Ca_post"]  # uM/ms

        # IP3R on ER membrane
        # De Young and Keizer, 1992; Li and Rinzel, 1994; Wagner et al., 2004
        m_inf_IP3R_post = x["IP3_post"] / (p["K_IP3_post"] + x["IP3_post"])     # 1
        n_inf_IP3R_post = x["Ca_post"] / (p["K_act_post"] + x["Ca_post"])       # 1
        J_IP3R_post = p["v_IP3R_post"] * m_inf_IP3R_post ** 3 * n_inf_IP3R_post ** 3 * x["h_IP3R_post"] ** 3 * (
                x["Ca_ER_post"] - x["Ca_post"])  # uM/ms

        # Dendritic NMDAR current and flux
        # Destexhe et al., 1998
        B_NMDAR_post = 1 / (1 + p["Mg_ext_post"] / 3570 * exp(-0.062 * x["V_dend_post"]))
        ICa_NMDAR_post = p["gNMDAR_post"] * B_NMDAR_post * x["m_NMDAR_post"] * (
                x["V_dend_post"] - p["V_NMDAR_post"])       # uA/cm^2
        J_NMDAR_post = -ICa_NMDAR_post / p["c_Ca_post"]     # uM/ms

        # Dendritic PMCA pump
        # Blackwell, 2002
        J_PMCA_post = p["k_Ca_post"] * p["A_spine_post"] * p["v_PMCA_post"] / p["V_spine_post"] * x["Ca_post"] ** 2 / (
                p["K_PMCA_post"] ** 2 + x["Ca_post"] ** 2)  # uM/ms

        # SERCA pump on ER membrane
        # De Young and Keizer, 1992; Li and Rinzel, 1994
        J_SERCA_post = p["v_SERCA_post"] * x["Ca_post"] ** 2 / (p["K_SERCA_post"] ** 2 + x["Ca_post"] ** 2)  # uM/ms

        return{"ICaLHVA_dend_post": ICaLHVA_dend_post,
               "ICaLLVA_dend_post": ICaLLVA_dend_post,
               "ICa_NMDAR_post": ICa_NMDAR_post,
               "J_CaL_post": J_CaL_post,
               "J_IP3R_post": J_IP3R_post,
               "J_NMDAR_post": J_NMDAR_post,
               "J_PMCA_post": J_PMCA_post,
               "J_SERCA_post": J_SERCA_post
               }

    def solve_deriv(self, deriv_post, dt):
        for key in self.x:
            self.x[key] += deriv_post[key] * dt

    def solve_deltaf(self, pre, Rrel_pre_old):

        # --------------------------------------------------
        # Pre- and postsynaptic neuron inputs and parameters
        # --------------------------------------------------
        prep = pre.params
        x = self.x
        prex = pre.x

        # -----------------------------------------------------
        # Updating those variables that include delta functions
        # -----------------------------------------------------

        # Modified from Lee et al., 2009; De Pitta et al., 2011; De Pitta and Brunel, 2016
        x["Glu_syncleft"] += prep["G_pre"] * prep["N_pre"] * prex["Prel_pre"] * Rrel_pre_old / (
                prep["k_Glu_pre"] * prep["N_A"] * prep["V_syncleft"])

