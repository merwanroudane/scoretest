# Mathematical Theory

## The Model

### Basic Regression Model (Equation 2.1)

$$y_t = x_t'\beta + u_t \quad (t = 1, \ldots, T)$$

where:
- $y_t$ is the observable response at time $t$
- $x_t' = (x_{t1}, \ldots, x_{tp})$ is a $1 \times p$ vector of known nonstochastic regressors
- $\beta = (\beta_1, \ldots, \beta_p)'$ is the $p \times 1$ vector of unknown parameters
- $u_t$ is the unobservable random error

### First-Order Autoregressive Error Process (Equation 2.2)

$$u_t = \rho u_{t-1} + e_t \quad (t = 2, \ldots, T)$$

with initial condition $u_1 = e_1$.

Here $\rho$ is the unknown autocorrelation coefficient, and the innovations $e_t$ are normally and independently distributed with mean 0.

### Heteroscedastic Variance (Equation 2.3)

$$\text{var}(e_t) = w_t \sigma^2 = w(z_t, \lambda)\sigma^2$$

where:
- $\lambda = (\lambda_1, \ldots, \lambda_q)'$ is a $q \times 1$ vector of parameters
- $z_t = (z_{t1}, \ldots, z_{tq})'$ is a known $q \times 1$ vector
- $w(\cdot, \cdot)$ is a twice-differentiable weight function
- There exists a unique $\lambda^*$ such that $w(z_t, \lambda^*) = 1$ for all $t$

---

## The Score Test

### Null Hypothesis

$$H_0: \rho = 0 \text{ and } \lambda = \lambda^*$$

This corresponds to the usual linear regression model with i.i.d. normal errors.

### Alternative Hypothesis

$$H_1: \rho \neq 0 \text{ or } \lambda \neq \lambda^*$$

Either autocorrelation or heteroscedasticity (or both) is present.

### Test Statistic (Equation 2.4)

$$S = S_1 + S_2$$

#### Component S₁ (Autocorrelation)

$$S_1 = \frac{(T\hat{\rho})^2}{T - 1}$$

where $\hat{\rho}$ is computed from Equation (2.5):

$$\hat{\rho} = \frac{\sum_{t=2}^{T} \hat{e}_t \hat{e}_{t-1}}{\sum_{t=1}^{T} \hat{e}_t^2}$$

Here $\hat{e}_t = y_t - x_t'\hat{\beta}$ are the OLS residuals, with:
- $\hat{\beta} = (X'X)^{-1}X'y$ (OLS estimator)
- $\hat{\sigma}^2 = \sum_{t=1}^{T} \hat{e}_t^2 / T$ (MLE of variance)

#### Component S₂ (Heteroscedasticity)

$$S_2 = \frac{1}{2} V'\bar{D}(\bar{D}'\bar{D})^{-1}\bar{D}'V$$

where:
- $V$ is a $T \times 1$ vector with element $\hat{e}_t^2 / \hat{\sigma}^2$
- $D$ is a $T \times q$ matrix with $t$th row $\partial w(z_t, \lambda)/\partial \lambda$ evaluated at $\lambda = \lambda^*$
- $\bar{D} = D - \mathbf{1}\mathbf{1}'D/T$ (centered D matrix)
- $\mathbf{1}$ is a $T \times 1$ vector of ones

### Asymptotic Distributions

Under $H_0$:
- $S_1 \stackrel{d}{\to} \chi^2(1)$
- $S_2 \stackrel{d}{\to} \chi^2(q)$
- $S \stackrel{d}{\to} \chi^2(q+1)$

The test rejects $H_0$ when $S > \chi^2_{q+1,\alpha}$.

---

## Derivation (From Appendix)

### Log-Likelihood Function

$$L = -\frac{T}{2}\log(2\pi) - \frac{T}{2}\log\sigma^2 - \frac{1}{2}\sum_{t=1}^{T}\log w_t + \frac{1}{2}\log(1-\rho^2)$$
$$- \frac{1}{2\sigma^2}\left\{(y_1 - x_1'\beta)^2(1-\rho^2)/w_1 + \sum_{t=2}^{T}(y_t^* - x_t^{*'}\beta)^2/w_t\right\}$$

where:
- $y_t^* = y_t - \rho y_{t-1}$
- $x_t^* = x_t - \rho x_{t-1}$

### Score Vector

At $w^* = (0, \lambda^{*'})'$:

$$U_{w_0^*}(\hat{\beta}, \hat{\sigma}^2) = \left[\frac{\sum_{t=2}^{T} \hat{e}_t \hat{e}_{t-1}}{\hat{\sigma}^2}, \frac{1}{2}(V - \mathbf{1})'D\right]'$$

### Expected Fisher Information

$$I = \text{diag}\left(\frac{1}{T-1}, \frac{1}{2}D'D\right)$$

$$K = \text{diag}\left(\frac{X'X}{\sigma^2}, \frac{T}{2\sigma^4}\right)$$

$$J = \text{diag}\left(0, \frac{1}{2}D'\mathbf{1}/\sigma^2\right)$$

### Score Statistic Formula

$$I^* = \text{diag}\left(\frac{1}{T-1}, 2(\bar{D}'\bar{D})^{-1}\right)$$

Therefore:
$$S = U'_{w_0^*}I^*U_{w_0^*} = \frac{(T\hat{\rho})^2}{T-1} + \frac{1}{2}V'\bar{D}(\bar{D}'\bar{D})^{-1}\bar{D}'V$$

---

## Normal Curvature (Section 3)

### Influence Graph (Equation 3.1)

$$M(w^*) = [L(w^*), w^{*'}]'$$

where $L(w^*) = 2\{L(\hat{\beta}) - L(\hat{\beta}_{w^*}|w^*)\}$.

### Curvature (Equation 3.2)

$$C_l = \frac{2|l'\hat{J}\hat{K}^{-1}\hat{J}'l|}{l'l}$$

where:
- $l = (l_1, \ldots, l_r)'$ is a direction vector
- $\hat{J}$ and $\hat{K}$ are defined in the paper

The maximum curvature $C_{\max} = \max_l C_l$ indicates the most influential perturbation direction.

### Parameter Sensitivity (Equation 3.4)

$$\left[\frac{\partial\hat{\beta}(w^*)}{\partial w^*}\right]_{w^*=w_0^*} = -\tilde{K}^{-1}\tilde{J}'$$

where $\tilde{K} = X'X$ and $\tilde{J}' = [\sum'(\hat{e}_{t-1}x_t + \hat{e}_tx_{t-1}), \sum\{\hat{e}_tx_t(\partial w_t/\partial\lambda)'\}]$.

---

## Robustness Properties

1. **S₁ is robust to heteroscedasticity** when $\hat{\rho}(\lambda - \lambda^*) \approx 0$

2. **S₂ is robust to autocorrelation** when $\rho/\sigma^2 \approx 0$

These properties mean that:
- S₁ can detect autocorrelation even in the presence of mild heteroscedasticity
- S₂ can detect heteroscedasticity even in the presence of mild autocorrelation

---

## Weight Function Examples

### Exponential (Default)

$$w(z, \lambda) = \exp(\lambda'z)$$

At $\lambda^* = 0$: $w = 1$, $\partial w/\partial\lambda = z$

### Linear

$$w(z, \lambda) = 1 + \lambda'z$$

At $\lambda^* = 0$: $w = 1$, $\partial w/\partial\lambda = z$

### Power

$$w(z, \lambda) = |z|^{2\lambda}$$

At $\lambda^* = 0$: $w = 1$, $\partial w/\partial\lambda = 2\log|z|$

---

## References

1. Tsai, C.-L. (1986). Score test for the first-order autoregressive model with heteroscedasticity. *Biometrika*, 73(2), 455-460.

2. Cook, R.D. & Weisberg, S. (1983). Diagnostics for heteroscedasticity in regression. *Biometrika*, 70(1), 1-10.

3. Durbin, J. & Watson, G.S. (1950, 1951, 1971). Testing for serial correlation in least squares regression. *Biometrika*.

4. Cox, D.R. & Hinkley, D.V. (1974). *Theoretical Statistics*. London: Chapman and Hall.

5. Bates, D.M. & Watts, D.G. (1980). Relative curvature measures of nonlinearity. *J.R. Statist. Soc. B*, 42, 1-25.
