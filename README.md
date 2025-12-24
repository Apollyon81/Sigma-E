# Sigma-E

Project: Simulation of Coupled Oscillators with Energy-Dependent Plasticity


This repository contains a minimal simulation of a system of coupled nonlinear oscillators, with memristive-like plasticity, homeostatic modulation, and adaptation dependent on a global energy proxy.
The goal is not to demonstrate a finished, functional system, but to test whether a structural hypothesis survives computational friction.

Core idea
The hypothesis explored here is simple:
A small set of oscillators, when exposed to structured external stimuli, can dynamically reorganize their couplings in a nontrivial way, modulated by global energy and local adaptation mechanisms.
The system includes:
• 	Van der Pol–type oscillators
• 	Adaptive (memristive) coupling
• 	Modulation by local activity
• 	A global energy proxy that affects the learning rate
Everything has been intentionally kept small and imperfect.

What this code is NOT
• 	Not a validated biological model
• 	Not a proven learning system
• 	Not optimized or scalable
• 	Not claiming practical usefulness in its current state
This project exists to fail fast or survive with minimal evidence.

Explicit failure criterion
I will consider the hypothesis invalidated at this stage if:
• 	the system does not show consistent reorganization of the mean couplings (w) in response to external stimuli,
or
• 	the coherence index shows no clear temporal correlation with the input pulses.
If the observed patterns are indistinguishable from numerical artifact or noise, the model does not pass this point.

Current outcome
The current code produces:
• 	time series of position and velocity
• 	evolution of adaptive parameters
• 	mean coupling matrix
• 	a global energy proxy and a simple coherence index
The plots are not proof — they are only the current state of the experiment.

Public commitment 72 hours
👉 Within 72 hours of this publication I will run one of the tests below and post the result here, regardless of outcome:
• 	increase the number of oscillators (N) and check whether the behavior persists
or
• 	remove homeostasis (b) and observe whether the system collapses
or
• 	freeze the couplings after an interval and analyze the subsequent dynamics
If the test contradicts the hypothesis, this will be recorded without retroactive adjustment of the narrative.

Open uncertainties
• 	I do not know whether the coherence index measures something structural or only numerical stability
• 	I do not know whether the observed behavior scales
• 	I do not know whether the dependence on global energy is essential or redundant
These uncertainties will not be resolved by explanation, only by testing or rejection.

Status
Exploratory.
No guarantees.
No promise of continuity beyond the next experiment.
