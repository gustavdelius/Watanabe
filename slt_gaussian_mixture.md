# Illustrating Watanabe's Singular Learning Theory with a Gaussian Mixture Model

This document provides a walk-through of Sumio Watanabe's *Singular Learning Theory* (up to Chapter 6), using a two-component Gaussian Mixture Model (GMM) as a concrete pedagogical example. 

## 1. Introduction: The Model and the True Distribution (Chapter 1)

In statistical formulation, we consider a learning machine defined by a parametric statistical model $p(x|w)$ and a true data-generating distribution $q(x)$.

### The Two-Component Gaussian Mixture
Let's define a simple 2-parameter Gaussian Mixture Model where one component is fixed at the origin:
$$ p(x|w) = (1-a) \mathcal{N}(x|0, 1) + a \mathcal{N}(x|\mu, 1) $$
Here, the parameter vector is $w = (a, \mu) \in W = [0, 1] \times [-c, c]$.

Suppose the **true distribution** is simply a standard normal distribution:
$$ q(x) = \mathcal{N}(x|0, 1) $$

![True Distribution vs Model](gmm_distributions.png)

### The Set of True Parameters
The true distribution is realized by the model whenever $p(x|w) = q(x)$. By inspecting the equation:
$$ (1-a) \mathcal{N}(x|0, 1) + a \mathcal{N}(x|\mu, 1) = \mathcal{N}(x|0, 1) $$
$$ a (\mathcal{N}(x|\mu, 1) - \mathcal{N}(x|0, 1)) = 0 $$

This holds true if and only if **$a = 0$** (the mixing proportion is zero) OR **$\mu = 0$** (both components are identical).
Thus, the set of true parameters $W_0$ is:
$$ W_0 = \{ (a, \mu) \in W : a = 0 \text{ or } \mu = 0 \} $$

This means that $W_0$ is not a single point, but the **union of two intersecting lines**. In classical (regular) statistical theory, $W_0$ is assumed to be a single point, and the Fisher Information Matrix is positive definite. Here, the Fisher Information Matrix degenerates on $W_0$, making this a **singular model**.

### The Kullback-Leibler Divergence
The log-likelihood ratio (empirical loss) is driven by the Kullback-Leibler (KL) divergence from $q(x)$ to $p(x|w)$:
$$ K(w) = \int q(x) \log \frac{q(x)}{p(x|w)} dx $$

Using a Taylor expansion for small $a$ and $\mu$, we can approximate $p(x|w)$:
$$ p(x|w) = \mathcal{N}(x|0, 1) \left[ 1 + a (e^{\mu x - \mu^2/2} - 1) \right] \approx \mathcal{N}(x|0, 1) \left[ 1 + a\left(\mu x + \frac{1}{2}\mu^2(x^2 - 1)\right) \right] $$

Plugging this into the KL divergence and using $-\log(1+z) \approx -z + z^2/2$, the linear terms integrate to 0 under $q(x)$, leaving the leading non-zero term:
$$ K(w) \approx \frac{1}{2} a^2 \mu^2 $$

![Parameter Space and KL Divergence](gmm_kl_divergence.png)
*(Notice how the valley of $K(w)=0$ forms a cross at $a=0$ and $\mu=0$)*

---

## 2. Resolution of Singularities (Chapter 2)

Because the set of true parameters $W_0$ has a singularity (an intersection forming a cross), standard asymptotic expansions (like the Laplace approximation) fail. Watanabe employs **Hironaka's Theorem on the Resolution of Singularities** from algebraic geometry to resolve this.

The theorem states that there exists a real analytic manifold $\mathcal{M}$ and a proper analytic map $g: \mathcal{M} \to W$ (a "blow-up") such that the composition $K(g(u))$ has a simple normal crossing form.

### The Blow-Up Transformation
For the approximation $K(w) \approx \frac{1}{2} a^2 \mu^2$, the true parameters $W_0$ correspond to the crossing lines $a=0$ and $\mu=0$. To resolve this intersection, we apply a "blow-up" transformation. A blow-up geometrically replaces the problematic intersection point (the origin) with an entire line (called the exceptional divisor), separating the paths that cross there.

We can reparameterize the space by keeping track of the *slope* of lines passing through the origin. We define a local coordinate chart (a directional blow-up) as:
$$ a = u_1 $$
$$ \mu = u_1 u_2 $$

Here, $u_1$ simply represents the original $a$ coordinate, while $u_2 = \mu / a$ represents the slope of a line passing through the origin in the $(a, \mu)$ parameter space.

- A single point in the original space—the origin $(a=0, \mu=0)$—now corresponds to the entire line $u_1 = 0$ for any value of $u_2$ in the new space.
- The crossing lines in the $(a, \mu)$ space have been pulled apart. Actually, our new coordinate system does not extend to the line $a=0$, but we can get arbitrarily close to it. We would need another patch with coordinates $\mu, a/\mu$ to represent the line $a=0$. 

The KL divergence in these new coordinates $u = (u_1, u_2)$ is
$$ K(g(u)) \approx \frac{1}{2} (u_1)^2 (u_1 u_2)^2 = \frac{1}{2} u_1^4 u_2^2 $$

We must also account for the distortion of the volume measure, dictated by the Jacobian of $g$:
$$ dw = |g'(u)| du = \left| \det \begin{pmatrix} 1 & 0 \\ u_2 & u_1 \end{pmatrix} \right| du_1 du_2 = |u_1| du_1 du_2 $$

---

## 3. Standard Form and Real Log Canonical Threshold (Chapter 3)

In Chapter 3, Watanabe introduces the concept of the **Real Log Canonical Threshold (RLCT)**, denoted by $\lambda$, and its multiplicity $m$. These two algebraic invariants completely govern the asymptotic behavior of the learning machine.

To understand why the blow-up was a strictly necessary algebraic maneuver, we have to look at how $\lambda$ is formally calculated. By definition, the RLCT is found by examining the analytic continuation of the **zeta function** of the statistical model, given by the integral:
$$ \zeta(z) = \int (K(w))^z \varphi(w) dw $$
where $\varphi(w)$ is the prior distribution and $z \in \mathbb{C}$ ($\Re(z) > 0$). The RLCT $\lambda$ is defined such that $-\lambda$ is the largest (closest to zero) real pole of this function, and its multiplicity $m$ is the order of this pole.

Without the blow-up, evaluating this integral and finding its poles is mathematically intractable. The true KL divergence $K(w)$ isn't just a simple polynomial like $a^2\mu^2$; it contains an infinite series of higher-order terms from the Taylor expansion. Because the variables are coupled in a highly non-linear way at the singularity (the cross $a=0, \mu=0$), you cannot separate the variables to evaluate the integral.

**This is where the blow-up of the singularity resolves the integration problem.** Hironaka's Theorem guarantees that after passing to the resolved coordinates $u$, the fully complex divergence $K(g(u))$ perfectly factors into a single monomial multiplied by a non-vanishing positive analytic function $b(u) > 0$. The prior measure and Jacobian of the blow-up also become a simple monomial multiplied by a strictly positive function $c(u) > 0$. 

When we substitute this **Standard Form** into the zeta function integral using our resolved coordinates $u$, the variables completely decouple near the origin:
$$ \zeta(z) = \int \left( u_1^{2k_{1}} u_2^{2k_{2}} b(u) \right)^z \left( u_1^{h_1} u_2^{h_2} c(u) \right) du_1 du_2 $$
$$ \zeta(z) \approx C \left( \int u_1^{2k_1 z + h_1} du_1 \right) \left( \int u_2^{2k_2 z + h_2} du_2 \right) $$

Evaluating these independent 1D integrals yields formulas of the type:
$$ \int_0^\epsilon u^{2k z + h} du = \frac{\epsilon^{2k z + h + 1}}{2k z + h + 1} $$
This expression clearly has a pole exactly when the denominator is zero, i.e., at $z = -\frac{h + 1}{2k}$. 

By separating the variables, the blow-up perfectly isolates the poles of the zeta function!

Using our specific factors from the GMM blow-up:
1. Divergence function: $K(u) = u_1^4 u_2^2$ (so $k_1 = 2$, $k_2 = 1$)
2. Prior measure / Jacobian: $\Phi(u) = u_1^1 u_2^0$ (so $h_1 = 1$, $h_2 = 0$)

The candidate poles $z = -\lambda_j$ along each coordinate axis $j$ give us:
$$ \lambda_j = \frac{h_j + 1}{2k_j} $$

For our $u_1$ and $u_2$:
- $\lambda_1 = \frac{1 + 1}{4} = \frac{1}{2}$
- $\lambda_2 = \frac{0 + 1}{2} = \frac{1}{2}$

The overall RLCT $\lambda$ is dictated by the pole closest to zero, which is the minimum of these values:
$$ \lambda = \min(\lambda_1, \lambda_2) = \min(1/2, 1/2) = \frac{1}{2} $$

The **multiplicity** $m$ is the order of this leading pole, which equals the number of coordinate indices $j$ that achieve this minimum. Here, both $\lambda_1$ and $\lambda_2$ equal $1/2$. Thus:
$$ m = 2 $$

*Note: In a regular model with parameter dimension $d=2$, the RLCT is always $\lambda = d/2 = 1$. Our singular GMM has $\lambda = 1/2 < 1$, showcasing the mathematical definition of a singular learning machine.*

---

## 4. Singular Fluctuation and Free Energy (Chapter 4)

In Bayesian evaluation, the **Stochastic Complexity** or **Free Energy** $F_n$ represents the negative log-marginal likelihood (evidence) of the data $X^n$:
$$ F_n = -\log Z_n = -\log \int e^{-n L_n(w)} \varphi(w) dw $$
where $L_n(w) = -\frac{1}{n} \sum \log p(X_i|w)$ is the empirical loss.

Chapter 4 defines how the parameter posterior behaves under singular fluctuations. Using the algebraic invariants we just found, SLT proves that the free energy asymptotically expands as:

$$ F_n \approx n L_n(w_0) + \lambda \log n - (m-1) \log \log n + O_p(1) $$

For our two-component Gaussian Mixture Model:
$$ F_n \approx n L_n(w_0) + \frac{1}{2} \log n - \log \log n + O_p(1) $$

![Asymptotic Free Energy](gmm_free_energy.png)
*(The penalty term for the singular model is significantly smaller than for a regular model, making singular models heavily preferred by the marginal likelihood).*

This result breaks the classical Bayesian Information Criterion (BIC), which assumes a penalty of $\frac{d}{2} \log n$. BIC would have penalized this 2-parameter model by $1.0 \log n$, entirely missing the true Bayesian Occam's Razor effect on the singular manifold.

---

## 5. Generalization and Training Errors (Chapter 5)

Chapter 5 links the geometry of the parameter space to the average errors. 
Let $G_n$ be the **Generalization Error** (expected loss on new data) and $T_n$ be the **Training Error** (empirical loss on the training set). 

In traditional regular statistics (like AIC), the expectation of these errors relies on $d$:
$$ \mathbb{E}[G_n] = L(w_0) + \frac{d}{2n} $$
$$ \mathbb{E}[T_n] = L(w_0) - \frac{d}{2n} $$

In Singular Learning Theory, Watanabe elegantly proves a symmetry relation using $\lambda$:
$$ \mathbb{E}[G_n] = L(w_0) + \frac{\lambda}{n} $$
$$ \mathbb{E}[T_n] = L(w_0) - \frac{\lambda}{n} $$

For our continuous GMM, $\lambda = 1/2$. Therefore:
- Expected Generalization Error converges with rate $\frac{0.5}{n}$.
- Expected Training Error converges with rate $-\frac{0.5}{n}$.

This explains why heavily overparameterized singular models (like deep neural networks and complex mixtures) often generalize better than their parameter count $d$ would imply; their learning dynamics are constrained by the smaller geometric invariant $\lambda < d/2$.

---

## 6. Asymptotic Expansion and WAIC (Chapter 6)

Finally, Chapter 6 resolves a crucial practical problem. Since calculating $\lambda$ analytically requires algebraic blow-ups (which is impossible for massive modern models like LLMs), we cannot use it directly to estimate the generalization error. Furthermore, cross-validation can be unstable in singular models.

Watanabe introduces the **Widely Applicable Information Criterion (WAIC)**:
$$ \text{WAIC} = T_n + \frac{1}{n} \sum_{i=1}^n V_w \left( \log p(X_i | w) \right) $$
Where $V_w$ is the posterior variance of the log-likelihood for data point $X_i$.

The core theorem of Chapter 6 proves that WAIC is an asymptotically unbiased estimator of the generalization error, **even for singular models**:
$$ \mathbb{E}[\text{WAIC}] \approx \mathbb{E}[G_n] $$

In the context of our GMM, evaluating the posterior variance computationally via Markov Chain Monte Carlo (MCMC) allows us to estimate the true hold-out performance without needing to analytically find $\lambda = 1/2$ or $m=2$, proving WAIC's universal applicability in modern deep learning and mixture modeling.
