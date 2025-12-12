# API Reference

## Core Module

### TsaiScoreTest

```python
class TsaiScoreTest(y, X, Z=None, weight_func=None, weight_deriv=None)
```

Score Test for the First-Order Autoregressive Model with Heteroscedasticity.

**Parameters:**
- `y` : array-like of shape (T,) - Response variable observations
- `X` : array-like of shape (T, p) - Design matrix of regressors
- `Z` : array-like of shape (T, q), optional - Variables for heteroscedasticity
- `weight_func` : callable, optional - Weight function w(z_t, λ)
- `weight_deriv` : callable, optional - Derivative ∂w(z_t, λ)/∂λ at λ = λ*

**Methods:**
- `fit()` → `ScoreTestResult` - Perform the score test

**Attributes:**
- `result` : ScoreTestResult - Test results after calling fit()
- `beta_hat` : np.ndarray - OLS estimate of β
- `residuals` : np.ndarray - OLS residuals

### ScoreTestResult

```python
@dataclass
class ScoreTestResult
```

Container for score test results.

**Attributes:**
- `S` : float - Joint score test statistic (Equation 2.4)
- `S1` : float - Score statistic for autocorrelation
- `S2` : float - Score statistic for heteroscedasticity
- `p_value` : float - P-value for joint test
- `p_value_S1` : float - P-value for S₁
- `p_value_S2` : float - P-value for S₂
- `df_S1` : int - Degrees of freedom for S₁ (always 1)
- `df_S2` : int - Degrees of freedom for S₂ (equals q)
- `df_total` : int - Degrees of freedom for S (equals q + 1)
- `rho_hat` : float - Estimated autocorrelation ρ̂
- `sigma2_hat` : float - Estimated error variance σ̂²
- `T` : int - Sample size
- `q` : int - Number of heteroscedasticity parameters
- `residuals` : np.ndarray - OLS residuals

**Methods:**
- `summary()` → str - Publication-ready summary
- `to_dict()` → dict - Convert to dictionary

---

## Convenience Functions

### score_test_joint

```python
score_test_joint(y, X, Z=None) → ScoreTestResult
```

Joint score test for autocorrelation and heteroscedasticity.

**Parameters:**
- `y` : array-like of shape (T,) - Response variable
- `X` : array-like of shape (T, p) - Design matrix
- `Z` : array-like of shape (T, q), optional - Heteroscedasticity variables

**Returns:**
- `result` : ScoreTestResult - Complete test results

### score_test_autocorrelation

```python
score_test_autocorrelation(y, X, return_rho=False) → tuple
```

Score test for first-order autocorrelation only.

**Parameters:**
- `y` : array-like of shape (T,) - Response variable
- `X` : array-like of shape (T, p) - Design matrix
- `return_rho` : bool - If True, also return ρ̂

**Returns:**
- `S1` : float - Score test statistic
- `p_value` : float - P-value from χ²(1)
- `rho_hat` : float - (only if return_rho=True)

### score_test_heteroscedasticity

```python
score_test_heteroscedasticity(y, X, Z) → tuple
```

Score test for heteroscedasticity only (Cook-Weisberg test).

**Parameters:**
- `y` : array-like of shape (T,) - Response variable
- `X` : array-like of shape (T, p) - Design matrix
- `Z` : array-like of shape (T, q) - Heteroscedasticity variables

**Returns:**
- `S2` : float - Score test statistic
- `p_value` : float - P-value from χ²(q)
- `df` : int - Degrees of freedom

---

## Weight Functions Module

### exponential_weight

```python
exponential_weight() → WeightFunction
```

Exponential weight function: w(z, λ) = exp(λ'z)

Default from Cook & Weisberg (1983). At λ* = 0: w = 1.

### linear_weight

```python
linear_weight() → WeightFunction
```

Linear weight function: w(z, λ) = 1 + λ'z

Breusch-Pagan style. At λ* = 0: w = 1.

### power_weight

```python
power_weight() → WeightFunction
```

Power weight function: w(z, λ) = |z|^{2λ}

For multiplicative heteroscedasticity.

---

## Diagnostics Module

### normal_curvature

```python
normal_curvature(y, X, Z) → CurvatureResult
```

Compute normal curvature for influence graph (Section 3, Equation 3.2).

**Parameters:**
- `y`, `X`, `Z` : array-like - Data

**Returns:**
- `CurvatureResult` with:
  - `C_max` : float - Maximum curvature
  - `l_max` : np.ndarray - Direction of maximum curvature
  - `J_hat` : np.ndarray - Ĵ matrix
  - `K_hat` : np.ndarray - K̂ matrix

### parameter_sensitivity

```python
parameter_sensitivity(y, X, Z) → SensitivityResult
```

Compute parameter sensitivity (Equation 3.4).

**Returns:**
- `SensitivityResult` with:
  - `sensitivity_matrix` : np.ndarray - ∂β̂/∂w*
  - `sensitivity_norms` : np.ndarray - Column norms

---

## Simulation Module

### simulate_critical_values

```python
simulate_critical_values(
    T, p, q,
    n_simulations=10000,
    alpha_levels=[0.01, 0.05, 0.10],
    seed=None,
    n_jobs=-1
) → SimulationResults
```

Monte Carlo simulation for finite-sample critical values.

**Parameters:**
- `T` : int - Sample size
- `p` : int - Number of regressors
- `q` : int - Number of heteroscedasticity parameters
- `n_simulations` : int - Number of Monte Carlo replications
- `alpha_levels` : list - Significance levels
- `seed` : int, optional - Random seed
- `n_jobs` : int - Number of parallel jobs (-1 for all CPUs)

**Returns:**
- `SimulationResults` with critical values and size analysis

### simulate_power

```python
simulate_power(
    T, p, q,
    rho_values,
    n_simulations=1000,
    alpha=0.05
) → PowerResults
```

Simulate power of the score test.

**Returns:**
- `PowerResults` with power curves for S, S₁, S₂

---

## Utilities Module

### ols_residuals

```python
ols_residuals(y, X, return_all=False) → np.ndarray or tuple
```

Compute OLS residuals.

### compute_rho_hat

```python
compute_rho_hat(residuals) → float
```

Compute ρ̂ from Equation (2.5).

### durbin_watson_statistic

```python
durbin_watson_statistic(residuals) → float
```

Compute Durbin-Watson statistic for comparison.

### breusch_godfrey_statistic

```python
breusch_godfrey_statistic(y, X, order=1) → tuple
```

Compute Breusch-Godfrey LM statistic.
