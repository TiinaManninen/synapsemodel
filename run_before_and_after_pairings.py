# Simulation of tripartite synapse model
# Tiina Manninen and Ausra Saudargiene
# Reference: Tiina Manninen, Ausra Saudargiene, and Marja-Leena Linne. Astrocyte-mediated spike-timing-dependent
# long-term depression modulates synaptic properties in the developing cortex. PLoS Comput Biol, 2020.

# -----------------------------------------------------------------------

from datetime import datetime
import os
from preneuron import Pre
from postneuron import Post
from astrocyte import Astro
from tqdm import tqdm
from scipy import io
from collections import defaultdict


def state_var_to_be_saved(pre, post, astro):

    return{
        "Ca_CaNHVA_pre": pre["Ca_CaNHVA_pre"],
        "Ca_NMDAR_pre": pre["Ca_NMDAR_pre"],
        "CaN_pre": pre["CaN_pre"],
        "Glu_syncleft": post["Glu_syncleft"],
        "Prel_pre": pre["Prel_pre"],
        "RA2_O_pre":  pre["RA2_O_pre"],
        "Rrel_pre": pre["Rrel_pre"],
        "V_pre": pre["V_pre"],
        "AG_post": post["AG_post"],
        "Ca_post": post["Ca_post"],
        "Ca_ER_post": post["Ca_ER_post"],
        "Ca_DAG_GaGTP_PLC_post": post["Ca_DAG_GaGTP_PLC_post"],
        "Ca_DAG_PLC_post": post["Ca_DAG_PLC_post"],
        "Ca_GaGTP_PLC_post": post["Ca_GaGTP_PLC_post"],
        "Ca_PLC_post": post["Ca_PLC_post"],
        "DAG_post": post["DAG_post"],
        "GaGTP_PLC_post": post["GaGTP_PLC_post"],
        "h_IP3R_post": post["h_IP3R_post"],
        "IP3_post": post["IP3_post"],
        "PLC_post": post["PLC_post"],
        "V_dend_post": post["V_dend_post"],
        "V_soma_post": post["V_soma_post"],
        "Ca_astro": astro["Ca_astro"],
        "Glu_extsyn": astro["Glu_extsyn"],
        "h_astro": astro["h_astro"],
        "IP3_astro": astro["IP3_astro"],
        "Rrel_astro": astro["Rrel_astro"]
           }


def main(path, f_pre):
    # Start to count the time spent in simulation
    time_start = datetime.now()
    print("Simulation started at", time_start)

    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    # --------------------
    # External stimulation
    # --------------------
    stim_start = 20000          # ms; Stimulation start time
    trainlengthtime = 25000     # ms; Stimulation lasting time
    restlengthtime = 20000      # ms; Resting time after stimulation ends
    no_trains = 1               # 1; Number of trains
    pulserate = 0.2             # Hz, 1/s; Frequency of stimulus
    pulselengthtime = 10        # ms; Length of the pulse

    A_stim_post = 0             # uA/cm^2; External current amplitude to postsynaptic neuron per unit area
    A_stim_pre = 10             # uA/cm^2; External current amplitude to presynaptic neuron per unit area

    dt = 0.05                   # ms; Simulation time step
    T_end = stim_start + no_trains * (trainlengthtime + restlengthtime)  # ms; Simulation end time
    Nsteps = round(T_end / dt)  # Number of simulation steps

    t = [i * dt for i in range(Nsteps + 1)]  # ms; Time vector

    pulselength = round(pulselengthtime / dt)
    no_pulses = round(pulserate * trainlengthtime * 1e-3)
    pauselengthtime = (trainlengthtime - no_pulses * pulselengthtime) / no_pulses  # ms
    pauselength = round(pauselengthtime / dt)
    restlength = round(restlengthtime / dt)

    # No need for T_shift in the simulations because no pairing protocol, only presynaptic stimulus
    steps_T_shift = 0

    # Only presynaptic stimulus, postsynaptic stimulus is zero
    stim_pause_post = ([A_stim_post] * pulselength + [0] * pauselength) * no_pulses
    stim_pause_pre = ([A_stim_pre] * pulselength + [0] * pauselength) * no_pulses

    # Postsynaptic stimulus is zero
    I_ext_post = [0] * (round(stim_start / dt) + 1) + (stim_pause_post + [0] * restlength) * no_trains

    # Presynaptic stimulus
    I_ext_pre = [0] * (round(stim_start / dt) + steps_T_shift + 1) + (stim_pause_pre + [0] * restlength) * no_trains

    # ---------------------------------
    # Presynaptic neuron initialization
    # ---------------------------------

    pre_params = Pre.get_parameters()
    pre_init = Pre.get_initial_values(pre_params)

    pre = Pre(pre_params, pre_init)

    # ----------------------------------
    # Postsynaptic neuron initialization
    # ----------------------------------

    post_params = Post.get_parameters()
    post_init = Post.get_initial_values(post_params)

    post = Post(post_params, post_init)

    # ------------------------
    # Astrocyte initialization
    # ------------------------

    astro_params = Astro.get_parameters()
    astro_init = Astro.get_initial_values(astro_params)

    astro = Astro(astro_params, astro_init)

    # -----------------------------------------
    # Initialization of Ca leak flux parameters
    # -----------------------------------------
    Ca_flux_post = post.calcium_other_fluxes()
    Ca_par_post = post.calcium_leak_parameters(Ca_flux_post["J_CaL_post"],
                                               Ca_flux_post["J_IP3R_post"],
                                               Ca_flux_post["J_NMDAR_post"],
                                               Ca_flux_post["J_PMCA_post"],
                                               Ca_flux_post["J_SERCA_post"])

    Ca_flux_astro = astro.calcium_other_fluxes()
    Ca_par_astro = astro.calcium_leak_parameters(Ca_flux_astro["J_IP3R_astro"],
                                                 Ca_flux_astro["J_SERCA_astro"])

    # ----------------------------------------------------------
    # Variables to be saved in a dictionary and then into a file
    # ----------------------------------------------------------
    state_var = state_var_to_be_saved(pre.x, post.x, astro.x)
    saved_state_var = {key: [values] for key, values in state_var.items()}
    saved_other_var = defaultdict(list)

    t_spike = 10     # ms

    for i in tqdm(range(Nsteps)):

        if i == stim_start * 1 / 2 / dt or i == stim_start * 3 / 4 / dt:

            # ---------------------------------
            # Adjusting Ca leak flux parameters
            # ---------------------------------
            Ca_flux_post = post.calcium_other_fluxes()
            Ca_par_post = post.calcium_leak_parameters(Ca_flux_post["J_CaL_post"],
                                                       Ca_flux_post["J_IP3R_post"],
                                                       Ca_flux_post["J_NMDAR_post"],
                                                       Ca_flux_post["J_PMCA_post"],
                                                       Ca_flux_post["J_SERCA_post"])

            Ca_flux_astro = astro.calcium_other_fluxes()
            Ca_par_astro = astro.calcium_leak_parameters(Ca_flux_astro["J_IP3R_astro"],
                                                         Ca_flux_astro["J_SERCA_astro"])

        # ---------------------------------------------
        # Saving old values for certain state variables
        # ---------------------------------------------
        Ca_pre_old = pre.x["Ca_CaNHVA_pre"]
        Prel_pre_old = pre.x["Prel_pre"]
        Rrel_pre_old = pre.x["Rrel_pre"]
        V_pre_old = pre.x["V_pre"]
        Ca_astro_old = astro.x["Ca_astro"]
        Rrel_astro_old = astro.x["Rrel_astro"]

        # ----------------------
        # Differential equations
        # ----------------------
        deriv_pre, other_var_pre = pre.derivative(
            astro.x["Glu_extsyn"], post.x["Glu_syncleft"], I_ext_pre[i+1])

        deriv_post, other_var_post = post.derivative(
            pre.params["f_Glu_pre"], I_ext_post[i+1], Ca_par_post["r_leakCell_post"], Ca_par_post["r_leakER_post"])

        deriv_ast, other_var_ast = astro.derivative(post.x["AG_post"], Ca_par_astro["r_leakER_astro"])

        # ----------------------------------
        # Solving the differential equations
        # ----------------------------------
        pre.solve_deriv(deriv_pre, dt)
        post.solve_deriv(deriv_post, dt)
        astro.solve_deriv(deriv_ast, dt)

        # -----------------------------------------------------
        # Updating those variables that include delta functions
        # -----------------------------------------------------

        # Counting the time from previous presynaptic spike
        if (pre.x["V_pre"] >= 0) and (V_pre_old < 0):
            t_spike = 0
        else:
            t_spike = t_spike + dt

        # Glu release from presynaptic neuron
        if (pre.x["Ca_CaNHVA_pre"] >= pre.params["C_thr_pre"]) and (t_spike < 10):
            pre.solve_deltaf(Ca_pre_old, f_pre, Prel_pre_old, Rrel_pre_old)
            post.solve_deltaf(pre, Rrel_pre_old)
            t_spike = 10

        # Glu release from astrocyte
        if (astro.x["Ca_astro"] >= astro.params["C_thr_astro"]) and (Ca_astro_old < astro.params["C_thr_astro"]):
            astro.solve_deltaf(Rrel_astro_old)

        # -----------
        # Saving data
        # -----------
        state_var = state_var_to_be_saved(pre.x, post.x, astro.x)

        for key, values in state_var.items():
            saved_state_var[key].append(values)

        other_var = {"f_pre": f_pre,
                     "Glu_NMDAR_pre": other_var_pre["Glu_NMDAR_pre"],
                     "ICaNHVA_pre": other_var_pre["ICaNHVA_pre"],
                     "ICa_NMDAR_pre": other_var_pre["ICa_NMDAR_pre"],
                     "I_AMPAR_post": other_var_post["I_AMPAR_post"],
                     "ICaLHVA_dend_post": other_var_post["ICaLHVA_dend_post"],
                     "ICaLLVA_dend_post": other_var_post["ICaLLVA_dend_post"],
                     "ICa_NMDAR_post": other_var_post["ICa_NMDAR_post"],
                     "J_CaL_post": other_var_post["J_CaL_post"],
                     "J_IP3R_post": other_var_post["J_IP3R_post"],
                     "J_leakCell_post": other_var_post["J_leakCell_post"],
                     "J_leakER_post": other_var_post["J_leakER_post"],
                     "J_NMDAR_post": other_var_post["J_NMDAR_post"],
                     "J_PMCA_post": other_var_post["J_PMCA_post"],
                     "J_SERCA_post": other_var_post["J_SERCA_post"]}

        for key, values in other_var.items():
            saved_other_var[key].append(values)

    # Saving dictionaries to mat files
    io.savemat(os.path.join(path, "state_var_results.mat"), saved_state_var)
    io.savemat(os.path.join(path, "other_var_results.mat"), saved_other_var)
    io.savemat(os.path.join(path, "time_stimuli.mat"),
               {**{"time": [tp / 1000 for tp in t]}, **{"I_ext_pre": [I_ext_pre]}, **{"I_ext_post": [I_ext_post]}})
    io.savemat(os.path.join(path, "stimulation_parameters.mat"),
               {**{"dt": dt}, **{"pulserate": pulserate}})

    # Simulation time
    time_end = datetime.now()
    total_time = (time_end - time_start).seconds / 60.  # min
    print("\n")
    print("Simulation finished at", time_end)
    print("Total time = {0:.2f} minutes".format(total_time))


if __name__ == "__main__":

    # -------------------------
    # Before pairing simulation
    # -------------------------

    # Fraction of presynaptic Glu release inhibition
    f_pre_base = 0

    # Define the name of the result directory to be created
    path = "./results_before_pairing/"

    # Calling the main function for simulation
    main(path, f_pre_base)

    # -------------------------
    # After pairing simulations
    # -------------------------

    # Fraction of presynaptic Glu release inhibition
    # End values from post-pre pairing simulations
    f_pre = {"10": 0.4968,
             "20": 0.4872,
             "30": 0.4690,
             "40": 0.4592,
             "50": 0.4483,
             "60": 0.4269,
             "70": 0.4066,
             "80": 0.3846,
             "90": 0.3613,
             "100": 0.3380,
             "110": 0.3131,
             "120": 0.2878,
             "130": 0.2601,
             "140": 0.2327,
             "150": 0.1893,
             "160": 0.1740,
             "170": 0.1283,
             "180": 0.1126,
             "190": 0.0630,
             "200": 0.0279
             }

    for key in f_pre:   # ms; Temporal difference between post and pre activation in pairing protocol, \
                        # needed here just to save the data to correct folders

        # Define the name of the result directory to be created
        path_template = "./results_after_pairing/%sms/"
        path = path_template % key

        # Calling the main function for simulation
        main(path, f_pre[key])




