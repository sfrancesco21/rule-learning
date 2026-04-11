# Mathematical Description of the Models

## Task Setup

Consider a sequential binary-choice task with trials indexed by $t=1,\dots,T$. On each trial, the participant observes a two-dimensional stimulus

$$
\mathbf{x}_t \in \mathbb{R}^2,
$$

and emits a binary response

$$
a_t \in \{0,1\}.
$$

There is an underlying correct category or response label

$$
y_t \in \{0,1\}.
$$

The experiment has two phases:

- **Training phase**: after responding, the participant receives feedback and can therefore learn the correct label $y_t$.
- **Test phase**: no feedback is provided, so the participant must rely on what was learned during training.

All models below assume that the participant forms, before responding on trial $t$, an internal belief

$$
q_t = P(y_t = 1 \mid \mathbf{x}_t, \mathcal{H}_{t-1}),
$$

where $\mathcal{H}_{t-1}$ denotes the information accumulated up to the previous trial.

To account for occasional random responding, all models include a lapse parameter $\lambda \in [0,1]$, yielding the choice rule

$$
P(a_t = 1 \mid \mathbf{x}_t, \mathcal{H}_{t-1})
= (1-\lambda)\, q_t + \lambda \cdot \tfrac12.
$$

Equivalently,

$$
P(a_t = 0 \mid \mathbf{x}_t, \mathcal{H}_{t-1})
= (1-\lambda)\,(1-q_t) + \lambda \cdot \tfrac12.
$$

For any non-missed trial, the contribution to the log-likelihood is therefore

$$
\log P(a_t \mid \mathbf{x}_t,\mathcal{H}_{t-1})
= a_t \log \tilde q_t + (1-a_t)\log(1-\tilde q_t),
$$

with

$$
\tilde q_t = (1-\lambda) q_t + \lambda \cdot \tfrac12.
$$

The central modeling question is thus how $q_t$ is computed and how $\mathcal{H}_{t-1}$ is updated over time.

## Working-Memory Models

The working-memory models assume that participants store episodic associations between previously seen stimuli and their correct responses. Let

$$
\mathcal{M}_{t-1} = \{(\mathbf{x}_i, y_i): i < t,\; i \text{ was in training and feedback was observed}\}
$$

denote the available memory set before trial $t$. During test, this memory is no longer updated, because no feedback is available.

For a query stimulus $\mathbf{x}_t$, memory contributes category-specific evidence

$$
s_c(\mathbf{x}_t)
= \beta + \sum_{i \in \mathcal{M}_{t-1}: y_i = c}
\frac{w_i}{\|\mathbf{x}_t - \mathbf{x}_i\|_2 + \varepsilon},
\qquad c \in \{0,1\},
$$

where:

- $\varepsilon > 0$ is a small distance regularizer,
- $\beta > 0$ is a baseline mass preventing degenerate zero evidence,
- $w_i \ge 0$ is the memory weight assigned to episode $i$.

These evidences are normalized into a predictive probability

$$
q_t
= P(y_t=1 \mid \mathbf{x}_t,\mathcal{M}_{t-1})
= \frac{s_1(\mathbf{x}_t)}{s_0(\mathbf{x}_t) + s_1(\mathbf{x}_t)}.
$$

The variants differ only in how the weights $w_i$ are defined.

### WM-2: Last-$M$ Memory

The simplest working-memory model assumes that only the most recent $M$ feedback-available training episodes are retained. If $\mathcal{R}_{t-1}(M)$ denotes those most recent $M$ episodes, then

$$
w_i =
\begin{cases}
1, & \text{if } i \in \mathcal{R}_{t-1}(M),\\
0, & \text{otherwise}.
\end{cases}
$$

Thus the model bases its decision on a finite recency-limited episodic buffer.

This variant has **2 free parameters**. These are $M$, the capacity of the episodic memory buffer, and $\lambda$, the lapse rate. The parameter $M$ controls how many recent training episodes remain behaviorally available at decision time, whereas $\lambda$ captures stimulus-independent random responding.

### WM-2C: Category-Specific Last-$M$ Memory

This variant allows different memory capacities for the two categories. Let $M_0$ and $M_1$ denote the number of retained episodes for categories $0$ and $1$, respectively. Then

$$
w_i =
\begin{cases}
1, & \text{if episode } i \text{ is among the last } M_{y_i} \text{ memories of its category},\\
0, & \text{otherwise}.
\end{cases}
$$

This allows asymmetric memory retention across the two response classes.

This variant has **3 free parameters**. These are $M_0$, the memory capacity for category-$0$ episodes, $M_1$, the memory capacity for category-$1$ episodes, and $\lambda$, the lapse rate. The pair $(M_0,M_1)$ allows the model to express asymmetric retention across the two categories.

### WM-1: Exponential Recency Decay

Instead of a hard memory cutoff, this model assumes that all stored episodes contribute, but their influence decays exponentially with recency. Let $\tau_i(t)$ denote the age of memory $i$ at trial $t$, measured as the number of more recent feedback-available memories. Then

$$
w_i = \exp\big(-\gamma\, \tau_i(t)\big),
\qquad \gamma \ge 0.
$$

Large $\gamma$ implies rapid forgetting; $\gamma \approx 0$ implies nearly uniform weighting over all stored episodes.

This variant has **2 free parameters**. These are $\gamma$, the exponential recency-decay rate, and $\lambda$, the lapse rate. The decay parameter $\gamma$ determines how quickly older episodes lose influence relative to more recent ones.

### WM-1D: Dual Exponential Decay

The dual-decay model separates two forms of forgetting:

1. decay with elapsed time or recency,
2. decay with the number of intervening memories from the same category.

Let:

- $\tau_i(t)$ be the usual recency age of episode $i$,
- $n_i^{\text{same}}(t)$ be the number of more recent stored episodes with the same label $y_i$.

Then the memory weight is

$$
w_i
= \exp\big(-\gamma_t \tau_i(t) - \gamma_c n_i^{\text{same}}(t)\big),
\qquad \gamma_t,\gamma_c \ge 0.
$$

This allows memory strength to decay both as a function of absolute recency and as a function of within-category interference.

This variant has **3 free parameters**. These are $\gamma_t$, the decay rate for absolute recency, $\gamma_c$, the decay rate for within-category interference, and $\lambda$, the lapse rate. The first parameter governs forgetting over time, while the second governs how strongly additional same-category memories suppress older ones.

## Gaussian-Mixture Rule Models

The Gaussian-mixture models assume that participants attempt to infer a latent category structure in stimulus space rather than merely memorizing episodes. Each response class $c \in \{0,1\}$ is represented by a class-conditional mixture density

$$
p_t(\mathbf{x} \mid y=c)
= \sum_{k=1}^{K_c} \pi_{c,k,t}\,
\mathcal{N}\!\big(\mathbf{x}; \boldsymbol{\mu}_{c,k,t}, \mathbf{\Sigma}_{c,k,t}\big),
$$

where:

- $K_c$ is the number of mixture components for class $c$,
- $\pi_{c,k,t}$ are the mixture weights,
- $\boldsymbol{\mu}_{c,k,t}$ are component means,
- $\mathbf{\Sigma}_{c,k,t}$ are full covariance matrices.

Given the current stimulus $\mathbf{x}_t$, the model computes class evidence

$$
Z_{c,t} = p_t(\mathbf{x}_t \mid y=c),
$$

and transforms it into a binary decision probability

$$
q_t
= P(y_t=1 \mid \mathbf{x}_t,\mathcal{H}_{t-1})
= \frac{Z_{1,t}}{Z_{0,t}+Z_{1,t}}.
$$

### Initial State

Each class is initialized with $K_c$ components that share a class-specific initial mean

$$
\boldsymbol{\mu}_{c,k,0} = \mathbf{m}_c,
$$

and a common isotropic initial covariance

$$
\mathbf{\Sigma}_{c,k,0} = \sigma_0^2 \mathbf{I}.
$$

Initial mixture weights are determined by a concentration parameter $\alpha > 0$. Using the mean of a stick-breaking construction, define

$$
v = \frac{1}{1+\alpha}.
$$

Then, for $k=1,\dots,K_c-1$,

$$
\pi_{c,k,0} = v(1-v)^{k-1},
$$

and the final component receives the remainder,

$$
\pi_{c,K_c,0} = (1-v)^{K_c-1}.
$$

Thus $\alpha$ controls the effective initial distribution of representational mass across components.

### Online Inference and Learning

On each trial, the model computes soft assignments (responsibilities) within each class:

$$
r_{c,k,t}
= P(z_t = k \mid \mathbf{x}_t, y=c, \mathcal{H}_{t-1})
\propto
\pi_{c,k,t-1}\,
\mathcal{N}\!\big(\mathbf{x}_t; \boldsymbol{\mu}_{c,k,t-1}, \mathbf{\Sigma}_{c,k,t-1}\big).
$$

After feedback in the training phase, only the mixture corresponding to the correct class $y_t$ is updated. The incorrect class is left unchanged. Conceptually, the update is an online, responsibility-weighted version of mixture learning:

- mixture mass is reassigned according to $r_{c,k,t}$,
- component means move toward the new stimulus,
- component covariances are updated from responsibility-weighted prediction errors.

The test phase uses the learned class densities but performs no further learning.

### Forgetting

Some variants include a forgetting parameter $\rho \in [0,1]$ acting on class-specific mixture mass before the new observation is incorporated. If $n_{c,k,t}$ denotes the effective mass assigned to component $k$ of class $c$, then the forgetting step is

$$
n_{c,k,t}^{-} = \rho\, n_{c,k,t-1},
$$

followed by the addition of current responsibility:

$$
n_{c,k,t} = n_{c,k,t}^{-} + r_{c,k,t}.
$$

When $\rho=1$, no forgetting occurs.

### Mean and Covariance Learning Rates

The model can also include explicit learning-rate parameters for the component means and covariances.

For component means, a generic update may be written as

$$
\boldsymbol{\mu}_{c,k,t}
= \boldsymbol{\mu}_{c,k,t-1}
+ \eta_\mu\, r_{c,k,t}\,
\big(\mathbf{x}_t - \boldsymbol{\mu}_{c,k,t-1}\big),
$$

where $\eta_\mu \in [0,1]$ controls how strongly the component center shifts toward the new observation.

For covariances, an analogous learning-rate parameter $\eta_\Sigma \in [0,1]$ controls how rapidly second-order statistics are updated. In variants without explicit learning rates, the updates reduce to standard normalized online mixture updates.

## The Four GMM Variants

All four GMM variants share the same observation model and differ only in whether they include forgetting and explicit learning-rate parameters.

### GMM-0: No Forgetting, No Explicit Learning Rates

$$
\rho = 1,
$$

and the model uses standard responsibility-weighted online updates for means and covariances, with no additional free learning-rate parameters.

This variant has **7 free parameters**. These are $\alpha$, the mixture concentration parameter controlling the initial distribution of component mass; $\lambda$, the lapse rate; $\sigma_0$, the initial covariance scale; and the two class-specific initial mean vectors $\mathbf{m}_0,\mathbf{m}_1 \in \mathbb{R}^2$, contributing four scalar parameters in total. Thus the fitted parameter set is

$$
\{\alpha,\lambda,\sigma_0,\mathbf{m}_0,\mathbf{m}_1\},
$$

with dimensionality $1+1+1+2+2=7$.

### GMM-1: No Forgetting, Tied Learning Rates

$$
\rho = 1,
\qquad
\eta_\mu = \eta_\Sigma = \eta.
$$

The single parameter $\eta$ controls how rapidly both means and covariances adapt.

This variant has **8 free parameters**. Relative to GMM-0, it adds the shared learning-rate parameter $\eta$, which jointly determines the rate of adaptation of component means and covariances. The fitted parameter set is

$$
\{\alpha,\lambda,\sigma_0,\eta,\mathbf{m}_0,\mathbf{m}_1\},
$$

with dimensionality $1+1+1+1+2+2=8$.

### GMM-2: Forgetting, No Explicit Learning Rates

$$
\rho \in [0,1]
$$

is estimated, but mean and covariance updates otherwise follow the standard normalized online mixture-learning rule.

This variant has **8 free parameters**. Relative to GMM-0, it adds the forgetting parameter $\rho$, which determines how strongly past mixture mass is discounted before incorporating each new feedback-bearing observation. The fitted parameter set is

$$
\{\alpha,\lambda,\sigma_0,\rho,\mathbf{m}_0,\mathbf{m}_1\},
$$

with dimensionality $1+1+1+1+2+2=8$.

### GMM-3: Forgetting, Tied Learning Rates

$$
\rho \in [0,1],
\qquad
\eta_\mu = \eta_\Sigma = \eta.
$$

This is the most flexible GMM variant: it allows both decay of past mixture mass and explicit control over the speed of representational adaptation.

This variant has **9 free parameters**. It includes both $\rho$, the forgetting parameter, and $\eta$, the shared mean/covariance learning-rate parameter, in addition to $\alpha$, $\lambda$, $\sigma_0$, and the two initial class means $\mathbf{m}_0,\mathbf{m}_1 \in \mathbb{R}^2$. The fitted parameter set is

$$
\{\alpha,\lambda,\sigma_0,\rho,\eta,\mathbf{m}_0,\mathbf{m}_1\},
$$

with dimensionality $1+1+1+1+1+2+2=9$.

## Interpretation

The two model families formalize different hypotheses about participants' learning strategies:

- **Working-memory models** assume decisions are driven by retrieval of previously seen stimulus-response episodes, with different assumptions about recency limits and forgetting.
- **Gaussian-mixture models** assume participants are constructing an internal category representation in stimulus space and generalizing from that representation to new stimuli.

In this sense, the working-memory models are fundamentally **episodic**, whereas the GMM models are fundamentally **rule-extrapolative**.
