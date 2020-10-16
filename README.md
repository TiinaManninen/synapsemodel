# Synapse model for developing somatosensory cortex

Computational model of a tripartite synapse in developing somatosensory cortex, including a presynaptic neuron, a 
postsynaptic neuron, and an astrocyte.

## Prerequisites

The code was developed and tested on Linux with Python 3.7. The prerequisites include numpy, scipy, and tqdm. 
To install these libraries on Linux type in terminal:

```
pip3 install -r requirements.txt
```
## Usage

1. The command ```python run_pairings.py``` will run the t-LTD induction protocol, thus 20 simulations with the 
different temporal differences of post-pre pairing protocol (-10 ms, -20 ms,..., -200 ms) and saves the data to separate 
folders (altogether 77.8 GB). Please, modify the path where to save the data, if needed. Note that the temporal 
differences of the post-pre pairings are implemented in the code as positive values, whereas they have negative values 
in the article. The reason for this is that we set the postsynaptic stimuli to occur at fixed time points in all the 
simulations whereas the presynaptic stimuli occurred in varying time points after the postsynaptic stimuli depending on 
temporal difference used, and not the other way around.

2. The command  ```python run_before_and_after_pairings.py``` will run the protocols before (baseline) and after t-LTD 
induction and saves the data to separate folders (altogether 9.9 GB). Please, modify the path where to save the data, 
if needed. Baseline protocol is run only once, but the protocol after t-LTD induction is run 20 times, thus once for 
each corresponding temporal difference in the t-LTD induction protocol. The protocol after t-LTD induction needs the 
actual end values of f_pre from the t-LTD induction protocol, and the values are given in 
run_before_and_after_pairings.py file. If you modify the model, you need to change these values in the code to the 
values your model gives after running run_pairings.py.

Plotting can be done with the given example MATLAB code: plotting_post_pre_pairing_data.m. It will plot in separate 
figures, the inputs, the saved model state variables, and the other saved output variables for a certain temporal 
difference of the post-pre pairing protocol. You can easily modify the path to your data to plot also the simulation 
results from the protocols before and after t-LTD induction.

In addition to the above two python files, the model includes presynaptic (preneuron.py) and postsynaptic 
(postneuron.py) neurons and an astrocyte (astrocyte.py). More details of the model are given below. Description of the 
whole model is given in the article and its appendix.

## Citation

Please, cite our article when using our code:

Tiina Manninen, Ausra Saudargiene, and Marja-Leena Linne. Astrocyte-mediated spike-timing-dependent long-term 
depression modulates synaptic properties in the developing cortex. PLoS Comput. Biol. 2020.
DOI: [10.1371/journal.pcbi.1008360](https://doi.org/10.1371/journal.pcbi.1008360)

## License

This work is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Additional information about the model

PRESYNAPTIC NEURON MODEL: Layer 4 spiny stellate cell of barrel cortex

Currents in axon: Ca_NHVA (high-voltage-activated N-type Ca), K, Na, GluN2C/D-containing NMDAR

Models adopted:

HH model including K and Na channels - from L4 spiny stellate cell model (Lavzin et al., 2012)

Ca_NHVA channel  - from pyramidal cell model (Safiulina et al., 2010)

NMDAR (Erreger et al., 2005; Clarke and Johnson, 2008)

CaN signaling (Fiala et al., 1996)

Glutamate release by presynaptic neuron (Combined and modified from Tsodyks et al., 1998; Lee et al., 2009; 
De Pitta et al., 2011; De Pitta and Brunel, 2016)

HH model:

Lavzin M, Rapoport S, Polsky A, Garion L, Schiller J (2012).
Nonlinear dendritic processing determines angular tuning of barrel cortex neurons in vivo.
Nature 490:397-401.
https://doi.org/10.1038/nature11451
https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=146565#tabs-1

Ca_NHVA channel:

Safiulina VF, Caiati MD, Sivakumaran S, Bisson G, Migliore M, Cherubini E (2010).
Control of GABA release at mossy fiber-CA3 connections in the developing hippocampus.
Front Synaptic Neuroscience 2:1.
https://doi.org/10.3389/neuro.19.001.2010
https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=126814#tabs-1

NMDAR:

Erreger K, Dravid SM, Banke TG, Wyllie DJA, Traynelis SF (2005).
Subunit-specific gating controls rat NR1/NR2A and NR1/NR2B NMDA channel kinetics and synaptic signalling profiles.
J Physiol 563.2: 345–358.
https://doi.org/10.1113/jphysiol.2004.080028

Clarke RJ, Johnson JW (2008).
Voltage-dependent gating of NR1/2B NMDA receptors.
J Physiol 586.23: 5727–5741.
https://doi.org/10.1113/jphysiol.2008.160622

CaN signaling:

Fiala JC, Grossberg S, Bullock D (1996). 
Metabotropic glutamate receptor activation in cerebellar Purkinje cells as substrate for adaptive timing of the 
classically conditioned eye-blink response.
J Neurosci 16(11):3760–3774.
https://doi.org/10.1523/JNEUROSCI.16-11-03760.1996

Glutamate release by presynaptic neuron:

Tsodyks M, Pawelzik K, Markram H (1998).
Neural networks with dynamic synapses.
Neural Comput 10(4):821–835.
https://doi.org/10.1162/089976698300017502

Lee C-CJ, Anton M, Poon C-S, McRae GJ (2009).
A kinetic model unifying presynaptic short-term facilitation and depression.
J Comput Neurosci 26:459–473.
https://doi.org/10.1007/s10827-008-0122-6
https://senselab.med.yale.edu/modeldb/ShowModel.cshtml?model=120184#tabs-1

De Pittà M, Volman V, Berry H, Ben-Jacob E (2011).
A tale of two stories: astrocyte regulation of synaptic depression and facilitation.
PLoS Comput Biol 7(12):e1002293.
https://doi.org/10.1371/journal.pcbi.1002293

De Pittà M, Brunel N (2016).
Modulation of synaptic plasticity by glutamatergic gliotransmission: a modeling study.
Neural Plast 2016:7607924.
https://doi.org/10.1155/2016/7607924

---

POSTSYNAPTIC NEURON MODEL: Layer 2/3 pyramidal cell of barrel cortex

Two compartmental model (Pinsky and Rinzel, 1994)

Currents in soma: K_DR (delayed rectifier K), Na,  Na_P (persistent Na)

Currents in dendrite: K_A (A-type K), AMPAR, Ca_LHVA (high-voltage-activated L-type Ca), 
Ca_LLVA (low-voltage-activated L-type Ca), Na, GluN2B-containing NMDAR

Models adopted:

HH model including K_A, K_DR and Na, Na_P channels - from L2/3 pyramidal cell model (Sarid et al., 2007)

Ca_LHVA channel - from neocortical pyramidal cell model (Reuveni et al., 1993, Markram et al., 2015)

Ca_LLVA channel - from hippocampal CA3 pyramidal cell model (Avery and Johnston, 1996)

AMPAR (Destexhe et al., 1998)

NMDAR (Destexhe et al., 1998)

mGluR -> 2-AG reactions (Mostly from Kim et al., 2013, but added model components from De Young and Keizer, 1992; 
Li and Rinzel, 1994; Blackwell, 2002; Zachariou et al., 2013)

HH model:

Sarid L, Bruno R, Sakmann B, Segev I, Feldmeyer D (2007).
Modeling a layer 4-to-layer 2/3 module of a single column in rat neocortex:
Interweaving in vitro and in vivo experimental observations.
Proc Natl Acad Sci USA 104(41): 16353-16358.
https://doi.org/10.1073/pnas.0707853104

Ca_L channels:

Reuveni I, Friedman A, Amitai Y, Gutnick MJ (1993).
Stepwise repolarization from Ca2+ plateaus in neocortical pyramidal cells: evidence for nonhomogeneous
distribution of HVA Ca2+ channels in dendrites.
J Neurosci 13(11): 4609-4621.
https://doi.org/10.1523/JNEUROSCI.13-11-04609.1993

Avery RB, Johnston D (1996).
Multiple channel types contribute to the low-voltage-activated calcium current in hippocampal CA3 pyramidal neurons.
J Neurosci 16(18): 5567–5582.
https://doi.org/10.1523/JNEUROSCI.16-18-05567.1996

Markram H, Muller E, Ramaswamy S, Reimann MW, Abdellah M, et al. (2015).
Reconstruction and simulation of neocortical microcircuitry.
Cell 163(2): 456-492.
https://doi.org/10.1016/j.cell.2015.09.029
https://senselab.med.yale.edu/modeldb/ShowModel?model=188543#tabs-1

AMPAR and NMDAR:

Destexhe A, Mainen ZF, Sejnowski TJ (1998).
Kinetic Models of Synaptic Transmission.
In Book: Methods in Neuronal Modeling, Koch C and Segev I, MIT Press, Cambridge, MA.

mGluR -> 2-AG reactions:

Kim B, Hawes SL, Gillani F, Wallace LJ, Blackwell KT (2013).
Signaling pathways involved in striatal synaptic plasticity are sensitive to temporal pattern and exhibit spatial 
specificity.
PLoS Comput Biol 9(3): e1002953.
https://doi.org/10.1371/journal.pcbi.1002953
http://krasnow1.gmu.edu/CENlab/publications.html

De Young GW, Keizer J (1992).
A single-pool inositol 1,4,5-trisphosphate-receptor-based model for agonist-stimulated oscillations in Ca2+ 
concentration.
Proc Natl Acad Sci USA 89(20): 9895–9899.
https://doi.org/10.1073/pnas.89.20.9895

Li YX, Rinzel J (1994).
Equations for InsP3 receptor-mediated [Ca2+]i oscillations derived from a detailed kinetic model: a Hodgkin-Huxley 
like formalism.
J Theor Biol 166(4): 461–473.
https://doi.org/10.1006/jtbi.1994.1041

Blackwell KT (2002).
Calcium waves and closure of potassium channels in response to GABA stimulation in Hermissenda type B photoreceptors.
J. Neurophysiol. 87: 776-792.
https://doi.org/10.1152/jn.00867.2000

Zachariou M, Alexander SPH, Coombes S, Christodoulou C (2013).
A biophysical model of endocannabinoid-mediated short term depression in hippocampal inhibition.
PLoS ONE 8(3): e58926.
https://doi.org/10.1371/journal.pone.0058926

---

ASTROCYTE MODEL

Models adopted:

Calcium equations (De Young and Keizer, 1992; Li and Rinzel, 1994)

Glutamate release by astrocyte (Combined from Tsodyks et al., 1998; Lee et al., 2009; De Pitta et al., 2011; 
Wade et al., 2012; De Pitta and Brunel, 2016)

Wade J, McDaid L, Harkin J, Crunelli V, Kelso S (2012).
Self-repair in a bidirectionally coupled astrocyte-neuron (AN) system based on retrograde signaling.
Front Comput Neurosci 6:76.
https://doi.org/10.3389/fncom.2012.00076
