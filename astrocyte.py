# Simulation of tripartite synapse model
# Tiina Manninen and Ausra Saudargiene
# Reference: Tiina Manninen, Ausra Saudargiene, and Marja-Leena Linne. Astrocyte-mediated spike-timing-dependent
# long-term depression modulates synaptic properties in the developing cortex. PLoS Comput Biol, 2020.

# -----------------------------------------------------------------------

# Astrocyte model includes the model components in the astrocyte and extrasynaptic space only.
class Astro:

    __DEFAULT_INITIAL_VALUES = {
        "Ca_astro": 0.15002,        # uM; Astrocytic Ca concentration
        "Glu_extsyn": 0,            # uM; Concentration of Glu in extrasynaptic space
        "h_astro": 0.7009,          # 1; Gating variable of astrocytic IP3R inactivation
        "IP3_astro": 0.28,          # uM; Astrocytic IP3 concentration
        "Rrel_astro": 1             # 1; Fraction of releasable astrocytic vesicles
    }

    __DEFAULT_PARAMS = {
        "AG_star_post": 0.0010453,  # uM; Postsynaptic resting 2-AG concentration
        "C_thr_astro": 0.3,         # uM; Ca threshold concentration of Glu exocytosis in astrocytes; Pasti et al., 2001
        "Ca_tot_astro": 2,          # uM; Total free astrocytic Ca concentration; De Young and Keizer, 1992
        "G_astro": 50000,           # uM; Glu concentration per astrocytic vesicle; De Pitta et al., 2011
        "IP3_star_astro": 0.28,     # uM; Astrocytic resting IP3 concentration; \
                                    # Modified from Nadkarni and Jung, 2003: 0.16 uM
        "k_recov_astro": 0.0006,    # 1/ms; Astrocytic recovery rate constant from empty to releasable state; \
                                    # De Pitta et al., 2011
        "K_act_astro": 0.08234,     # uM; Astrocytic IP3R dissociation constant of Ca (activation); \
                                    # De Young and Keizer, 1992
        "K_inh_astro": 1.049,       # uM; Astrocytic IP3R dissociation constant of Ca (inhibition); \
                                    # De Young and Keizer, 1992
        "K_IP3_1_astro": 0.13,      # uM; Astrocytic IP3R dissociation constant of IP3; De Young and Keizer, 1992
        "K_IP3_2_astro": 0.9434,    # uM; Astrocytic IP3R dissociation constant of IP3; De Young and Keizer, 1992
        "K_SERCA_astro": 0.1,       # uM; Half-activation constant of astrocytic SERCA pump; De Young and Keizer, 1992
        "N_astro": 4,               # 1; Number of readily releasable astrocytic vesicles; De Pitta et al., 2011
        "Prel_astro": 0.6,          # 1; Basal release probability of astrocytic Glu vesicles; \
                                    # De Pitta et al., 2011; De Pitta and Brunel, 2016
        "r_astro": 0.005,           # 1/ms; Glu clearance rate from extrasynaptic space; De Pitta and Brunel, 2016
        "r_ERcyt_astro": 0.185,     # 1; Ratio between astrocytic ER and cytosol volumes; De Young and Keizer, 1992
        "r_IP3_astro": 0.0008,      # 1/ms; Rate constant of astrocytic IP3 production; Nadkarni and Jung, 2003
        "r_IP3R_astro": 0.0002,     # 1/(uM ms); Astrocytic IP3R binding constant for Ca (inhibition); \
                                    # De Young and Keizer, 1992
        "r_vesext_astro": 6.5e-4,   # 1; Ratio between astrocytic vesicular volume and volume of \
                                    # extrasynaptic space; De Pitta et al., 2011; De Pitta and Brunel, 2016
        "tau_IP3_astro": 7000,      # ms; Time constant for astrocytic IP3 degradation; Nadkarni and Jung, 2003
        "v_IP3R_astro": 0.006,      # 1/ms; Maximum rate of Ca release via astrocytic IP3R; De Young and Keizer, 1992
        "v_SERCA_astro": 0.0007     # uM/ms; Maximum rate of Ca uptake by astrocytic SERCA pump; \
                                    # Modified from De Young and Keizer, 1992: 9e-4 uM/ms;
    }

    @staticmethod
    def get_parameters():
        return dict(Astro.__DEFAULT_PARAMS)

    @staticmethod
    def get_initial_values(p):
        x = dict(Astro.__DEFAULT_INITIAL_VALUES)
        intermed = Astro.intermediate_equations(p, x)
        x["h_astro"] = intermed["h_inf_astro"]

        return x

    @staticmethod
    def intermediate_equations(p, x):

        # --------------------------------
        # Astrocyte IP3R related equations
        # --------------------------------

        # De Young and Keizer, 1992; Li and Rinzel, 1994
        Q_astro = p["K_inh_astro"] * (x["IP3_astro"] + p["K_IP3_1_astro"]) / (x["IP3_astro"] + p["K_IP3_2_astro"])  # uM
        h_inf_astro = Q_astro / (Q_astro + x["Ca_astro"])  # 1
        tau_h_astro = 1 / (p["r_IP3R_astro"] * (Q_astro + x["Ca_astro"]))  # ms

        return {"h_inf_astro": h_inf_astro,
                "tau_h_astro": tau_h_astro
                }

    def __init__(self, params, x0):
        print("Created new Astro")
        self.params = params    # Parameters
        self.x = x0             # Initial values of state variables

    def variable_names(self):
        return self.x.keys()

    def derivative(self, AG_post, r_leakER_astro):
        # -------------------------------
        # Astrocyte inputs and parameters
        # -------------------------------
        p = self.params
        x = self.x

        # -----------------------------
        # Astrocyte algebraic equations
        # -----------------------------

        # Ca fluxes in astrocyte
        Ca_flux_astro = Astro.calcium_other_fluxes(self)
        Ca_leak_astro = Astro.calcium_leak_fluxes(self, r_leakER_astro)

        # IP3R equations
        intermed = Astro.intermediate_equations(p, x)

        # -----------------------------------------------------------
        # Differential equations in astrocyte and extrasynaptic space
        # -----------------------------------------------------------
        return {
                   # De Young and Keizer, 1992; Li and Rinzel, 1994
                   "Ca_astro": Ca_flux_astro["J_IP3R_astro"] - Ca_flux_astro["J_SERCA_astro"] + \
                               Ca_leak_astro["J_leakER_astro"],
                   "h_astro": (intermed["h_inf_astro"] - x["h_astro"]) / intermed["tau_h_astro"],

                   # Modified from Nadkarni and Jung 2003; Wade et al., 2012
                   "IP3_astro": (p["IP3_star_astro"] - x["IP3_astro"]) / p["tau_IP3_astro"] + p["r_IP3_astro"] * (
                           AG_post - p["AG_star_post"]),

                   # Tsodyks and Markram, 1997; Tsodyks et al., 1998; Lee et al., 2009; De Pitta et al., 2011;
                   # De Pitta and Brunel, 2016
                   "Rrel_astro":  p["k_recov_astro"] * (1 - x["Rrel_astro"]),  # Delta term subtracted in solve_deltaf

                   # De Pitta et al., 2011; De Pitta and Brunel, 2016
                   "Glu_extsyn": -p["r_astro"] * x["Glu_extsyn"],  # Delta term added in solve_deltaf
               }, {"J_IP3R_astro": Ca_flux_astro["J_IP3R_astro"],
                   "J_leakER_astro": Ca_leak_astro["J_leakER_astro"],
                   "J_SERCA_astro": Ca_flux_astro["J_SERCA_astro"]
                   }

    def calcium_leak_parameters(self, J_IP3R_astro, J_SERCA_astro):
        # -------------------------------
        # Astrocyte inputs and parameters
        # -------------------------------
        p = self.params
        x = self.x

        # ---------------------------------
        # Adjusting Ca leak flux parameters
        # ---------------------------------

        # r_leakER_astro adjusted such that net Ca flux across the astrocytic ER membrane
        # is zero at the resting Ca concentration (Blackwell, 2002)
        # (J_SERCA_astro-J_IP3R_astro-J_leakER_astro)=0
        r_leakER_astro = (J_SERCA_astro - J_IP3R_astro) / (p["Ca_tot_astro"] - (1 + p["r_ERcyt_astro"]) * x["Ca_astro"])
        print(r_leakER_astro)

        return {"r_leakER_astro": r_leakER_astro}

    def calcium_leak_fluxes(self, r_leakER_astro):
        # -------------------------------
        # Astrocyte inputs and parameters
        # -------------------------------
        p = self.params
        x = self.x

        # -----------------------
        # Astrocytic Ca leak flux
        # -----------------------

        # De Young and Keizer, 1992; Li and Rinzel, 1994
        J_leakER_astro = r_leakER_astro * (p["Ca_tot_astro"] - (1 + p["r_ERcyt_astro"]) * x["Ca_astro"])  # uM/ms

        return{"J_leakER_astro": J_leakER_astro}

    def calcium_other_fluxes(self):
        # -------------------------------
        # Astrocyte inputs and parameters
        # -------------------------------
        p = self.params
        x = self.x

        # -----------------------------------------
        # Astrocytic Ca fluxes other than leak flux
        # -----------------------------------------

        # IP3R on ER membrane
        # De Young and Keizer, 1992; Li and Rinzel, 1994
        m_inf_astro = x["IP3_astro"] / (p["K_IP3_1_astro"] + x["IP3_astro"])    # 1
        n_inf_astro = x["Ca_astro"] / (p["K_act_astro"] + x["Ca_astro"])        # 1
        J_IP3R_astro = p["v_IP3R_astro"] * m_inf_astro ** 3 * n_inf_astro ** 3 * x["h_astro"] ** 3 * (
                p["Ca_tot_astro"] - (1 + p["r_ERcyt_astro"]) * x["Ca_astro"])   # uM/ms

        # SERCA pump on ER membrane
        # De Young and Keizer, 1992; Li and Rinzel, 1994
        J_SERCA_astro = p["v_SERCA_astro"] * x["Ca_astro"] ** 2 / (
                p["K_SERCA_astro"] ** 2 + x["Ca_astro"] ** 2)                   # uM/ms

        return{"J_IP3R_astro": J_IP3R_astro,
               "J_SERCA_astro": J_SERCA_astro
               }

    def solve_deriv(self, deriv_ast, dt):
        for key in self.x:
            self.x[key] += deriv_ast[key] * dt

    def solve_deltaf(self, Rrel_astro_old):

        # -------------------------------
        # Astrocyte inputs and parameters
        # -------------------------------
        p = self.params
        x = self.x

        # -----------------------------------------------------
        # Updating those variables that include delta functions
        # -----------------------------------------------------

        # Tsodyks and Markram, 1997; Tsodyks et al., 1998; Lee et al., 2009; De Pitta et al., 2011;
        # De Pitta and Brunel, 2016
        x["Rrel_astro"] -= p["Prel_astro"] * Rrel_astro_old

        # De Pitta et al., 2011; De Pitta and Brunel, 2016
        x["Glu_extsyn"] += p["r_vesext_astro"] * p["G_astro"] * p["N_astro"] * p["Prel_astro"] * Rrel_astro_old

