---
name: confocal-ellipse-bridge-synthesis
description: Mathematical reference and red-team audit guardrails for Paper IX on Riemann's shifted Mellin bridge decomposition for Xi * E_X, L^2 Stieltjes prime-tail bounds, Mertens' harmonic benchmark, and the four structural obstructions (branch asymmetry, L^1 divergence, off-diagonal cancellation, and Voronin universality) preventing phase-blind closure on d > 1/4.
---

# Paper IX Skill: Shifted Mellin Bridge, Diagonal Stieltjes Bounds, and Off-Diagonal Obstructions

**Author:** Thijs Missiaen (ORCID: `0009-0007-3678-0810`)

## When to Activate

Activate this skill whenever inspecting, extending, or citing Paper IX (`paper9-confocal-ellipse-bridge-synthesis`), finite Euler-sieve products $\Xi(t - \mathrm{i}d)E_X(s)$, incomplete-Gamma shells $\mathcal{J}_n(s)$, Stieltjes prime-tail bounds, or confocal-ellipse mode decompositions.

## Canonical Normalization and Proven Identities

1. **Canonical Transverse Wronskian:**

$$\Xi(t - \mathrm{i}d) = 8\int _ 0^{\infty} \Phi(u)\cos\bigl(2(t - \mathrm{i}d)u\bigr)\,\mathrm{d}u, \qquad W(d, t) = \frac{1}{2}\frac{\partial}{\partial d}\lvert \Xi(t - \mathrm{i}d)\rvert^2 = 16\int _ 0^{\infty} y\sinh(2dy)\,I(y, t)\,\mathrm{d}y,$$

where $I(y, t) = \int _ {-\infty}^{\infty} \Phi(\frac{y+x}{2})\Phi(\frac{y-x}{2})\cos(2tx)\,\mathrm{d}x$.

2. **Riemann-Type Shifted Mellin Decomposition (Theorem 2.1):** Adapting Riemann's (1859) classical split at $u = 0$ to $\Phi _ {\mathcal{E}}(u) = \sum _ {k \mid P _ X} \mu(k)k^{-1/2}\Phi(u + \frac{1}{2}\ln k)$ gives

$$\frac{1}{2}\Xi(t - \mathrm{i}d)\,E _ X(s) = \mathcal{I} _ +^{\mathcal{E}}(s) + E _ X(s)\,F _ +(1 - s) + \mathcal{B} _ {\mathrm{bridge}}^{\mathcal{E}}(s),$$

with closed incomplete-Gamma shells $\mathcal{J} _ n(s) = \frac{1}{2}\pi^{-s/2}n^{-s}[(s - 1)\Gamma(\frac{s}{2}+1,\pi n^2) + 2(\pi n^2)^{s/2+1}\mathrm{e}^{-\pi n^2}]$.

3. **Hyperbolic Mode Ratios, $L^2$ Stieltjes Bounds, and Mertens' Benchmark (Theorem 3.1):** With $N _ {\star}(t) = \sqrt{t/(2\pi)}$ and $\omega _ n(t) = \ln(N _ {\star}(t)/n)$, one has $\sinh^2(\frac{1}{2}d\omega _ n)/\sinh^2(\frac{1}{2}d\omega _ 1) \le n^{-d}$ and $\sinh^2(d\omega _ n)/\sinh^2(d\omega _ 1) \le n^{-2d}$. On $X \ge \max(2, \sqrt{N(t)})$, every integer $n \in (\lfloor X\rfloor, N(t)]$ with $b _ n^{\mathcal{E}}(X) \neq 0$ is strictly a prime $p \in (X, N(t)]$, and Stieltjes integration by parts against $\vartheta(u) \le c _ {\vartheta}u$ (Rosser–Schoenfeld 1962) gives $\mathcal{E} _ {\mathrm{diag}}^{\mathcal{E},L^2}(d,t; X) = \sum _ {X < p \le N(t)} p^{-1-2d} \le c _ {\vartheta}[X^{-2d}/\ln X + \int _ X^{N(t)} u^{-1-2d}/\ln u\,\mathrm{d}u]$, while Mertens' theorem (1874) explains the harmonic threshold $\theta _ {\mathrm{Mertens}} = \frac{1}{2}\mathrm{e}^{-1} \approx 0.18394$.

4. **Elementary Single-Mode Identity (Proposition 4.1):** The relation $\cosh^2(a)\cos^2\phi + \sinh^2(a)\sin^2\phi = \cos^2\phi + \sinh^2(a)$ is an elementary trigonometric-hyperbolic identity for a single mode.

## Mandatory Guardrails Against Falsified Overclaims

- **Never switch $n^{-1/2} \mapsto n^{-(1-d)}$:** With the true Riemann–Siegel factor $n^{-1/2}$, the $L^1$ prime sum weighted by $\sinh^2(\frac{1}{2}d\omega_n)/\sinh^2(\frac{1}{2}d\omega_1) \asymp n^{-(1/2+d)}$ diverges as $\frac{N(t)^{1/2-d}}{(1/2-d)\ln N(t)} \to +\infty$ for every $0 < d < \frac{1}{2}$ (Proposition 3.2).
- **Never claim phase-blind zero-free closure on $d > \frac{1}{4}$:** Mixing linear $n^{-1/2}$ with quadratic $\sinh^2(d\omega_n)$ violates homogeneity; positive $L^2$ diagonals do not prevent destructive off-diagonal cancellation $\mathcal{C} _ {\mathrm{off}}(d, t) < 0$; and Voronin's universality theorem (1975) together with Bohr density proves $\inf _ {t > 0}\lvert \zeta(\sigma + \mathrm{i}t)E_X(\sigma + \mathrm{i}t)\rvert = 0$ on $\frac{3}{4} < \sigma < 1$ (Proposition 4.3).
- **Never treat symmetric mode sums as approximate functional equations for $\Xi E_X$:** Because $E_X(s) \neq E_X(1 - s)$, the reflected branch in Theorem 2.1 is $E_X(s)F _ +(1 - s)$, not $F _ +^{\mathcal{E}}(1 - s)$ (Proposition 2.3).
- **Never claim continuum grid verification to astronomical heights:** Evaluating $N(t) \asymp \sqrt{t}$ modes scales exponentially in $\ln t$; unconditional zero verification currently reaches $t \le 3 \times 10^{12}$ (Platt–Trudgian 2021).

## Verification Command

```bash
python scripts/verify_paper9_confocal_ellipse_synthesis.py
```
