# Mechanistic formalism

This document holds the equations that define the model so every coefficient is auditable. The implementation in `src/dynamics/` mirrors what is written here term-for-term, and `src/dynamics/forcing_functions.py` is the canonical source for the EPL weights.

## State space

Let $z_t \in \mathbb{R}^6$ denote the latent state of a participant on day $t$, with components

$$
z_t = \bigl(z_t^{(1)}, z_t^{(2)}, z_t^{(3)}, z_t^{(4)}, z_t^{(5)}, z_t^{(6)}\bigr)
= \bigl(\text{autonomic recovery},\ \text{circadian alignment},\ \text{stress load},\ \text{environmental burden},\ \text{behavioral instability},\ \text{missingness pressure}\bigr).
$$

Each axis is z-scored at the cohort level so $\mathbb{E}[z_t^{(j)}] \approx 0$ and $\mathrm{Var}[z_t^{(j)}] \approx 1$ in a healthy reference population.

## Continuous-time dynamics

The generative model is a forced, contractive stochastic differential equation on the latent manifold:

$$
\mathrm{d}z_t = f_\theta(z_t, E_t, B_t, P_t)\,\mathrm{d}t + \sigma\,\mathrm{d}W_t,
$$

with environmental input $E_t$, behavioral input $B_t$, perturbation input $P_t$, and Brownian residual $W_t$. The drift decomposes additively:

$$
f_\theta(z, E, B, P) = -\kappa\,z + \alpha_E\,F_{\text{env}}(E) + \alpha_B\,F_{\text{beh}}(B) + P.
$$

In the discrete-time implementation, forward Euler with step $\Delta t = 1$ day:

$$
z_{t+1} = (1 - \kappa)\,z_t + \alpha_E\,F_{\text{env}}(E_t) + \alpha_B\,F_{\text{beh}}(B_t) + P_t + \epsilon_t,\qquad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I).
$$

Defaults are $\kappa = 0.1$, $\alpha_E = \alpha_B = 0.04$, $\sigma = 0.01$. An RK4 integrator and a learned-residual `NeuralODEStep` are also available.

## Environmental Physiological Load (EPL)

EPL is a deterministic composite of four environmental components, each z-scored at the cohort level:

$$
\mathrm{EPL}_t = w_d\,h^{\text{day}}_t + w_n\,h^{\text{night}}_t + w_a\,a_t + w_h\,H_t,
$$

with $h^{\text{day}}_t$ = daytime heat, $h^{\text{night}}_t$ = nighttime heat, $a_t$ = AQI, $H_t$ = consecutive heat-wave exposure days, and default weights $(w_d, w_n, w_a, w_h) = (0.30,\ 0.30,\ 0.25,\ 0.15)$. Weighting nighttime heat as heavily as daytime heat reflects the literature on impaired thermoregulation during sleep. The heat-wave term is a regime indicator rather than a temperature.

## Environmental forcing

The forcing $F_{\text{env}}: \mathbb{R}^4 \to \mathbb{R}^6$ projects EPL components onto the latent dimensions:

$$
F_{\text{env}}(E_t) = v \cdot \begin{pmatrix}
-\mathrm{EPL}_t \\
-h^{\text{night}}_t \\
+\mathrm{EPL}_t \\
+\mathrm{EPL}_t \\
+0.4 \cdot \mathrm{EPL}_t \\
+0.2 \cdot H_t
\end{pmatrix},
$$

where $v$ is the participant-level vulnerability coefficient (centered on 1, larger for climate-sensitive participants). Behavioral forcing $F_{\text{beh}}$ projects sleep, activity, screen, and social-rhythm z-scores onto the latent axes via an analogous linear map.

## Perturbation operator

A perturbation specification is a triple $(\tau, m, H)$ of type, magnitude, and horizon. The operator returns an additive sequence $P_{1:H} \in \mathbb{R}^{H \times 6}$ with geometric decay:

$$
P_t = m \cdot d_\tau \cdot \rho^{\,t-1},\qquad \rho = \exp\!\bigl(-\log 2 / \tau_{1/2}\bigr),\qquad t = 1, \dots, H,
$$

where $d_\tau \in \mathbb{R}^6$ is the unit-direction vector for perturbation type $\tau$ and $\tau_{1/2}$ is the participant's resilience half-life (default 4 days). The heat-wave shock reverses the sign of $\log\rho$ so the magnitude ramps up rather than decays. Counterfactual trajectories are produced by injecting $P_{1:H}$ into the dynamics; baseline and counterfactual integrate from the same $z_0$ with the same noise realization.

## Resilience half-life

For a participant whose post-perturbation stress excursion $s_t = z_t^{(3)}$ is observed for $t = 0, \dots, T$, the resilience half-life is estimated by regressing $\log s_t$ on $t$ via least squares and recovering $\tau_{1/2} = \log 2 / \hat{\beta}$. The estimator is robust to small $T$ by clipping at a configurable maximum.

## Uncertainty

For a downstream task head $g_\phi$ there are two uncertainty options. MC dropout with $K$ stochastic forward passes yields

$$
\hat{\mu}(x) = \tfrac{1}{K}\sum_{k=1}^K g_\phi(x;\,\xi_k),\qquad \hat{\sigma}^2(x) = \tfrac{1}{K-1}\sum_{k=1}^K \bigl(g_\phi(x;\,\xi_k) - \hat{\mu}(x)\bigr)^2.
$$

A lightweight ensemble averages $M$ independent heads, with $\hat{\sigma}^2$ the across-head variance. Prediction intervals are formed under a Gaussian assumption with confidence $1 - \alpha$.

## Critical-transition warning score

Three dynamical-systems early-warning signals are computed over a rolling window $W$ on the latent stress dimension $z_t^{(3)}$: increasing variance, increasing lag-1 autocorrelation, and an instability index that combines the two. A fourth term, the Mahalanobis distance to the dysregulated cluster centroid, is added when the regime detector is available. The warning score is

$$
S_t = \tfrac{1}{4}\bigl(\tilde V_t + \tilde A_t + \tilde I_t + \tilde D_t\bigr) \in [0, 1],
$$

with tildes denoting per-day min-max normalization. The score is descriptive, not prescriptive: there is no claim that it predicts clinical events.
