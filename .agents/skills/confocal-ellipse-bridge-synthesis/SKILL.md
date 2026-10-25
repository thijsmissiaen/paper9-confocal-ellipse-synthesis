---
name: confocal-ellipse-bridge-synthesis
description: >-
  Scientific knowledge skill for Paper IX of the Transverse Wronskian Series on Riemann's Xi-Function.
  Provides the exact three-part u-domain bridge identity for Xi(t-id)E_X(s), quadratic Wronskian
  boundary damping Delta C_n = 2 sinh^2((1/2)d omega_n) -> 0 at the saddle-point horizon n -> N_*(t),
  explicit unconditional Rosser-Schoenfeld and Dusart Stieltjes upper bounds across [10^8, inf),
  the exact algebraic sinh^2 ratio identity <= n^{-d}, the phase-independent symplectic confocal
  ellipse identity |T_n(d,t)|^2 - |T_n(0,t)|^2 = 4n^{-1}sinh^2(d omega_n) with uniform phase-blind
  L^1 closure for d > 1/4, and the three-domain half-plane architecture [0, 55] U [55, 10^8] U [10^8, inf)
  with Category A/B/C epistemic classification.
---

# Confocal Ellipse Identity, Three-Part $u$-Domain Bridge, and Explicit Stieltjes Bounds Skill

This skill provides autonomous mathematical and coding agents with the analytical identities, closed-form Stieltjes inequalities, and directed-rounding interval verification protocols established in **Paper IX** of the Transverse Wronskian Series by Thijs Missiaen (Independent Researcher, Belgium; ORCID: `0009-0007-3678-0810`).

## 1. Core Mathematical Identities and Theorems

### 1.1 The Exact Three-Part $u$-Domain Bridge Identity (Theorem 2.1)

For $s = \frac{1}{2} + d + \mathrm{i} t$ ($z = t - \mathrm{i} d$) and sieve cutoff $X \ge 1$, let $E _ X(s) = \exp(-\sum _ {m=2}^{\lfloor X\rfloor} \frac{\Lambda(m)}{\ln m}m^{-s}) = \sum _ {k=1}^{\infty} c _ k^{\mathcal{E}}(X)k^{-s}$ and $\zeta(s)E _ X(s) = \sum _ {n=1}^{\infty} b _ n^{\mathcal{E}}(X)n^{-s}$. Splitting the bilateral Laplace–Mellin integral of $\Phi _ {\mathcal{E} _ X}(u)$ at $u = 0$ and applying modular Jacobi reflection $\Phi(-w) = \Phi(w)$ at the shifted symmetry point $w = u + \frac{1}{2}\ln k = 0$ yields the exact identity

$$\frac{1}{2}\Xi(t - \mathrm{i} d) E _ X(s) \equiv \mathcal{I} _ +^{\mathcal{E} _ X}(s) + E _ X(s) F _ +(1 - s) + \mathcal{B} _ {\mathrm{bridge}}^{\mathcal{E} _ X}(s),$$

where the forward branches satisfy the bridge-reconstruction duality $\mathcal{I} _ +^{\mathcal{E} _ X}(s) + \mathcal{B} _ {\mathrm{bridge}}^{\mathcal{E} _ X}(s) \equiv E _ X(s) F _ +(s)$. Each right half-line shell integral admits the closed incomplete-Gamma representation

$$\mathcal{J} _ n(s) := 2\int _ 0^{\infty} \frac{1}{\sqrt{n}}\varphi _ 1\left(u + \frac{1}{2}\ln n\right)\mathrm{e}^{2(s-1/2)u} \mathrm{d}u = \frac{1}{2}\pi^{-s/2}n^{-s}\left[(s-1)\Gamma\left(\frac{s}{2}+1, \pi n^2\right) + 2\left(\pi n^2\right)^{s/2+1}\mathrm{e}^{-\pi n^2}\right] = \mathcal{O}\left(\pi n^2 \mathrm{e}^{-\pi n^2}\right).$$

For $X \ge 6$, all shells $2 \le n \le 6$ vanish ($b _ n^{\mathcal{E}}(X) = 0$) and $\mathrm{e}^{-49\pi} \approx 1.39 \times 10^{-67}$, so $\mathcal{I} _ +^{\mathcal{E} _ X}(s)$ coincides with the single $n = 1$ mother wavelet $\mathcal{J} _ 1(s)$ to 63 decimal places.

### 1.2 The Complementarity Theorem and Quadratic Wronskian Boundary Damping (Theorem 2.2)

For each active Riemann–Siegel mode $1 \le n \le N(t) = \lfloor N _ \star(t)\rfloor$ with $N _ \star(t) := \sqrt{t/2\pi}$ and $\omega _ n(t) := \ln(N _ \star(t)/n) \ge 0$, the pure Wronskian excess above the critical line $d = 0$ is

$$\Delta\mathcal{C} _ n(d,t) := \cosh\left(d\omega _ n(t)\right) - 1 \equiv 2\sinh^2\left(\frac{1}{2}d\omega _ n(t)\right) \ge 0.$$

Whereas $\Delta\mathcal{C} _ 1(d,t)$ is exponentially maximal at $n = 1$, $\Delta\mathcal{C} _ n(d,t) \sim \frac{1}{2}d^2(\ln(N _ \star(t)/n))^2 \to 0$ vanishes quadratically at the stationary-phase horizon $u = 0 \iff n = N _ \star(t)$. Consequently:
- **Zone I ($2 \le n \le \lfloor X\rfloor$):** Large hyperbolic weights $\Delta\mathcal{C} _ n$ are annihilated identically by $b _ n^{\mathcal{E}}(X) \equiv 0$.
- **Zone II ($\lfloor X\rfloor < n \le N(t){}$):** Surviving prime powers are quadratically damped near $n \to N(t)$, contributing two additional powers of $(\ln N(t))^{-1}$ beyond prime dilution to yield $(\ln N(t))^{-3}$ boundary suppression and $X _ {\mathrm{W\text{-}pure}}^{\mathcal{E}}(d, 10^8) \in [9, 53] < \sqrt{N(10^8)} = 63$ ($\theta _ {\mathrm{W\text{-}pure}}^{\mathcal{E}}(d, 10^8) \in [0.1263, 0.2389] < \frac{1}{4}$).
- **Bridge Damping ($u \in [-\frac{1}{2}\ln k, 0]$, $k \le X = N _ \star(t)^{2\theta}$):** The pointwise lever satisfies $\mathcal{R} _ {\mathrm{full}}(d,t;\theta) = \sinh^2(d\theta\omega _ 1)/\sinh^2(d\omega _ 1) \le \theta^2$ and the integrated Wigner–Sinh lever satisfies $\mathcal{R} _ {\mathrm{int}}(d,t;\theta) \le \theta^3$.

### 1.3 Explicit Rosser–Schoenfeld Stieltjes Bounds and the Exact $\sinh^2$ Ratio Identity (Theorem 3.1 and Table 2)

For all $1 \le n \le N _ \star(t)$ and $d > 0$, the pure Wronskian weight ratio obeys the exact algebraic identity

$$\frac{\sinh^2\left(\frac{1}{2}d\omega _ n(t)\right)}{\sinh^2\left(\frac{1}{2}d\omega _ 1(t)\right)} \equiv n^{-d}\left(\frac{1 - (n/N _ \star(t))^d}{1 - N _ \star(t)^{-d}}\right)^2 \le n^{-d}.$$

Using the unconditional Chebyshev inequalities of Rosser and Schoenfeld (1962) and Dusart (2010, 2018) ($\vartheta(x) \le c _ \vartheta x$ with $c _ \vartheta = 1.000028$, $\psi(x) \le c _ \psi x$ with $c _ \psi = 1.03883$), Stieltjes integration on the quadratic purity window $n \in (e^{\theta L}, N(t)]$ ($\theta \in [\frac{1}{4}, \frac{1}{2})$, $L := \ln(t/2\pi){}$) yields for all $t \in [10^8, \infty)$ without prime summation:

$$\mathcal{E} _ {\mathrm{diag}}^{\mathcal{E},\mathrm{fwd}}\left(d,t;\mathrm{e}^{\theta L}\right) \le \overline{\mathcal{E}} _ {\mathrm{RS}}^{\mathrm{fwd}}(d,t;\theta) := \frac{c _ {\mathrm{RS}}(\theta L)}{\theta L}\cdot\frac{\mathrm{e}^{-2\theta d L} - \mathrm{e}^{-dL}}{2d} < \frac{c _ {\mathrm{RS}}(\theta L)}{2d\theta L}\left(\frac{t}{2\pi}\right)^{-2\theta d} \longrightarrow 0,$$

and proves that the scale-invariant pure Wronskian exponent satisfies

$$\lim _ {t\to\infty} \overline{\theta} _ {\mathrm{W\text{-}pure}}^{\mathcal{E}}(d,t) \le \frac{1}{2}\exp\left(-\frac{1}{c _ \vartheta}\right) = 0.1839448708257938635\dots < \frac{1}{4} < \frac{1}{2},$$

with a conservative $C^0$ handshake at $T _ 1 = 10^8$ ($\mathrm{Ratio\ Exact/Analytical} \in [0.9402, 0.9874] < 1$).

### 1.4 The Phase-Independent Confocal Ellipse Identity and Uniform $L^1$ Closure for $d > \frac{1}{4}$ (Theorems 4.1 and 4.2)

For each combined forward and reflected Riemann–Siegel mode $T _ n(d,t) = 2n^{-1/2}(\cosh(d\omega _ n)\cos\phi _ n + \mathrm{i}\sinh(d\omega _ n)\sin\phi _ n)$ with phase $\phi _ n(t) = \vartheta _ {\mathrm{RS}}(t) - t\ln n \in \mathbb{R}$, $T _ n(d,t)$ traces a confocal ellipse with $d$-invariant foci $c _ n \equiv \pm 2n^{-1/2}$, symplectic area $2\pi n^{-1}\sinh(2d\omega _ n) > 0$, and phase-independent modulus-squared increment

$$\left\lvert T _ n(d,t)\right\rvert^2 - \left\lvert T _ n(0,t)\right\rvert^2 \equiv 4 n^{-1} \sinh^2\left(d\omega _ n(t)\right) > 0 \qquad (\forall \phi _ n(t) \in \mathbb{R}).$$

The normalized ellipse-diagonal and Wronskian-diagonal ratios satisfy

$$w _ n^{\mathrm{ell}}(d,t) \equiv n^{-(1+2d)}\left(\frac{1 - (n/N _ \star(t))^{2d}}{1 - N _ \star(t)^{-2d}}\right)^2 \le n^{-(1+2d)}, \qquad w _ n^{\mathrm{W\text{-}diag}}(d,t) \equiv n^{-(\frac{1}{2}+2d)}\left(\frac{1 - (n/N _ \star(t))^{2d}}{1 - N _ \star(t)^{-2d}}\right)^2 \le n^{-(\frac{1}{2}+2d)}.$$

Whenever $d > \frac{1}{4}$ ($\sigma > \frac{3}{4}$), the exponent $\frac{1}{2} + 2d > 1$ makes even the completely phase-blind $L^1$ sum of $w _ n^{\mathrm{W\text{-}diag}}(d,t)$ converge uniformly as $t \to \infty$ with finite sieve cutoff $X _ \infty(d) \le 13$ and $\lim _ {t\to\infty}\overline{\theta} _ {\mathrm{W\text{-}diag}}^{\mathcal{E}}(d,t) = 0$. Furthermore, the $n = 1$ complex transverse increment anchor satisfies $\lvert\Delta T _ 1(d,t)\rvert \ge 4\sinh^2(\frac{1}{2}d\omega _ 1(t)) > 0$ for all $\vartheta _ {\mathrm{RS}}(t) \in \mathbb{R}$.

### 1.5 Three-Domain Master Architecture and Epistemic Classification (Theorem 5.1)

The upper half-plane $(d,t) \in [0,\infty) \times [0,\infty)$ is partitioned into Domain I ($[0, 55]$, certified via `mpmath.iv` Argument Principle $N _ {\mathrm{zeros}}(\Omega _ k) \equiv 0$ and $4.98\times 10^6$ GPU Lipschitz cells), Domain II ($[55, 10^8]$, certified via $2.5\times 10^6$ GPU cells and exact 550-prime sums), and Domain III ($[10^8, \infty)$, governed by closed-form Rosser–Schoenfeld Stieltjes bounds), isolating the single remaining asymptotic question on $0 < d \le \frac{1}{4}$ as $t \to \infty$ (pointwise unaveraged phase cancellation in Lemma L-1 $\iff$ exclusion of interior Chladni branch points $\Xi'(z _ c) = 0$ in Lemma L-2).

## 2. Reproducible Verification Command

Run the standalone verification script from the repository root:

```bash
python scripts/verify_paper9_confocal_ellipse_synthesis.py
```
