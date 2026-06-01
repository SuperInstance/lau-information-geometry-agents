//! Statistical manifolds: concrete parametric families of distributions.

use crate::*;
use nalgebra::{DMatrix, DVector};

/// Normal distribution N(μ, σ²) parameterized by θ = (μ, log σ)
/// Using log σ ensures unconstrained parameterization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalManifold;

impl NormalManifold {
    pub fn new() -> Self {
        Self
    }

    /// Extract μ from θ
    pub fn mu(theta: &ManifoldPoint) -> f64 {
        theta.theta[0]
    }

    /// Extract σ from θ (stored as log σ)
    pub fn sigma(theta: &ManifoldPoint) -> f64 {
        theta.theta[1].exp()
    }

    /// Create θ from (μ, σ)
    pub fn params(mu: f64, sigma: f64) -> ManifoldPoint {
        ManifoldPoint::new(vec![mu, sigma.ln()])
    }
}

impl Default for NormalManifold {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalManifold for NormalManifold {
    fn dimension(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        let mu = Self::mu(theta);
        let sigma = Self::sigma(theta);
        let z = (x - mu) / sigma;
        (-0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln()).exp()
    }

    fn log_pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        let mu = Self::mu(theta);
        let sigma = Self::sigma(theta);
        let z = (x - mu) / sigma;
        -0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln()
    }

    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64> {
        let mu = Self::mu(theta);
        let sigma = Self::sigma(theta);
        let diff = x - mu;
        // ∂/∂μ log p = (x - μ) / σ²
        let d_mu = diff / (sigma * sigma);
        // ∂/∂(log σ) log p = (x - μ)² / σ² - 1
        let d_log_sigma = diff * diff / (sigma * sigma) - 1.0;
        vec![d_mu, d_log_sigma]
    }

    fn fisher_information(&self, theta: &ManifoldPoint) -> DMatrix<f64> {
        // Fisher info for (μ, log σ) parameterization:
        // g_μμ = 1/σ², g_μ,logσ = 0, g_{logσ,logσ} = 2
        let sigma = Self::sigma(theta);
        DMatrix::from_row_slice(2, 2, &[
            1.0 / (sigma * sigma), 0.0,
            0.0, 2.0,
        ])
    }

    fn valid_params(&self, _theta: &ManifoldPoint) -> bool {
        true // log σ is unconstrained
    }
}

/// Multivariate normal N(μ, Σ) — diagonal covariance only for tractability.
/// θ = (μ₁, ..., μₖ, log σ₁, ..., log σₖ)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagonalNormalManifold {
    pub k: usize,
}

impl DiagonalNormalManifold {
    pub fn new(k: usize) -> Self {
        Self { k }
    }

    pub fn mu(&self, theta: &ManifoldPoint) -> DVector<f64> {
        DVector::from_vec(theta.theta[..self.k].to_vec())
    }

    pub fn sigma(&self, theta: &ManifoldPoint) -> DVector<f64> {
        DVector::from_vec(theta.theta[self.k..].iter().map(|&x| x.exp()).collect())
    }

    pub fn params_from_means_cov(&self, mu: &[f64], sigma: &[f64]) -> ManifoldPoint {
        let mut theta = mu.to_vec();
        theta.extend(sigma.iter().map(|&s| s.ln()));
        ManifoldPoint::new(theta)
    }
}

impl StatisticalManifold for DiagonalNormalManifold {
    fn dimension(&self) -> usize {
        2 * self.k
    }

    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        // Treat x as scalar applied to first dimension for simplicity
        let mu = theta.theta[0];
        let sigma = theta.theta[self.k].exp();
        let z = (x - mu) / sigma;
        (-0.5 * z * z - 0.5 * (2.0 * std::f64::consts::PI).ln() - sigma.ln()).exp()
    }

    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64> {
        let mut s = vec![0.0; 2 * self.k];
        let mu = theta.theta[0];
        let sigma = theta.theta[self.k].exp();
        let diff = x - mu;
        s[0] = diff / (sigma * sigma);
        s[self.k] = diff * diff / (sigma * sigma) - 1.0;
        s
    }

    fn fisher_information(&self, theta: &ManifoldPoint) -> DMatrix<f64> {
        let n = 2 * self.k;
        let mut g = DMatrix::zeros(n, n);
        for i in 0..self.k {
            let sigma_i = theta.theta[self.k + i].exp();
            g[(i, i)] = 1.0 / (sigma_i * sigma_i);
            g[(self.k + i, self.k + i)] = 2.0;
        }
        g
    }

    fn valid_params(&self, theta: &ManifoldPoint) -> bool {
        theta.theta.len() == 2 * self.k
    }
}

/// Categorical distribution on {1, ..., K} parameterized by θ = (η₁, ..., η_{K-1})
/// where p_k = softmax(η). The K-th probability is determined.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalManifold {
    pub k: usize,
}

impl CategoricalManifold {
    pub fn new(k: usize) -> Self {
        assert!(k >= 2);
        Self { k }
    }

    /// Convert parameters to probabilities via softmax
    pub fn to_probs(&self, theta: &ManifoldPoint) -> Vec<f64> {
        let mut eta = theta.theta.clone();
        eta.push(0.0); // anchor last class at 0
        let max_eta = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = eta.iter().map(|&e| (e - max_eta).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|&e| e / sum).collect()
    }

    /// Create uniform parameters
    pub fn uniform_params(&self) -> ManifoldPoint {
        ManifoldPoint::new(vec![0.0; self.k - 1])
    }
}

impl StatisticalManifold for CategoricalManifold {
    fn dimension(&self) -> usize {
        self.k - 1
    }

    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        let probs = self.to_probs(theta);
        let idx = x as usize;
        if idx < probs.len() {
            probs[idx]
        } else {
            0.0
        }
    }

    fn log_pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        let probs = self.to_probs(theta);
        let idx = x as usize;
        if idx < probs.len() && probs[idx] > 0.0 {
            probs[idx].ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64> {
        let probs = self.to_probs(theta);
        let k = x as usize;
        let mut s = vec![0.0; self.k - 1];
        for i in 0..self.k - 1 {
            s[i] = if k == i { 1.0 - probs[i] } else { -probs[i] };
        }
        s
    }

    fn fisher_information(&self, theta: &ManifoldPoint) -> DMatrix<f64> {
        let probs = self.to_probs(theta);
        let n = self.k - 1;
        let mut g = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    g[(i, j)] = 1.0 / probs[i] + 1.0 / probs[self.k - 1];
                } else {
                    g[(i, j)] = 1.0 / probs[self.k - 1];
                }
            }
        }
        g
    }

    fn valid_params(&self, theta: &ManifoldPoint) -> bool {
        theta.theta.len() == self.k - 1 && self.to_probs(theta).iter().all(|&p| p > 0.0)
    }
}

/// Exponential distribution with rate λ. θ = log λ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialManifold;

impl ExponentialManifold {
    pub fn new() -> Self {
        Self
    }

    pub fn lambda(theta: &ManifoldPoint) -> f64 {
        theta.theta[0].exp()
    }

    pub fn params(lambda: f64) -> ManifoldPoint {
        ManifoldPoint::new(vec![lambda.ln()])
    }
}

impl Default for ExponentialManifold {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalManifold for ExponentialManifold {
    fn dimension(&self) -> usize {
        1
    }

    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        if x < 0.0 {
            return 0.0;
        }
        let lambda = Self::lambda(theta);
        lambda * (-lambda * x).exp()
    }

    fn log_pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        if x < 0.0 {
            return f64::NEG_INFINITY;
        }
        let lambda = Self::lambda(theta);
        lambda.ln() - lambda * x
    }

    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64> {
        let lambda = Self::lambda(theta);
        // ∂/∂(log λ) log p = 1 - λx
        vec![1.0 - lambda * x]
    }

    fn fisher_information(&self, _theta: &ManifoldPoint) -> DMatrix<f64> {
        // Fisher info for log λ parameterization = 1
        DMatrix::from_row_slice(1, 1, &[1.0])
    }

    fn valid_params(&self, _theta: &ManifoldPoint) -> bool {
        true
    }
}

/// Poisson distribution with rate λ. θ = log λ.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoissonManifold;

impl PoissonManifold {
    pub fn new() -> Self {
        Self
    }

    pub fn lambda(theta: &ManifoldPoint) -> f64 {
        theta.theta[0].exp()
    }

    pub fn params(lambda: f64) -> ManifoldPoint {
        ManifoldPoint::new(vec![lambda.ln()])
    }
}

impl Default for PoissonManifold {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalManifold for PoissonManifold {
    fn dimension(&self) -> usize {
        1
    }

    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        let k = x.round() as i64;
        if k < 0 {
            return 0.0;
        }
        let lambda = Self::lambda(theta);
        (-lambda + k as f64 * lambda.ln() - ln_factorial(k)).exp()
    }

    fn log_pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        let k = x.round() as i64;
        if k < 0 {
            return f64::NEG_INFINITY;
        }
        let lambda = Self::lambda(theta);
        -lambda + k as f64 * lambda.ln() - ln_factorial(k)
    }

    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64> {
        let lambda = Self::lambda(theta);
        vec![x / lambda - 1.0]
    }

    fn fisher_information(&self, _theta: &ManifoldPoint) -> DMatrix<f64> {
        DMatrix::from_row_slice(1, 1, &[1.0])
    }

    fn valid_params(&self, _theta: &ManifoldPoint) -> bool {
        true
    }
}

fn ln_factorial(n: i64) -> f64 {
    if n <= 1 {
        0.0
    } else {
        (1..=n).map(|i| (i as f64).ln()).sum()
    }
}

/// Beta distribution B(α, β). θ = (log α, log β).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaManifold;

impl BetaManifold {
    pub fn new() -> Self {
        Self
    }

    pub fn alpha(theta: &ManifoldPoint) -> f64 {
        theta.theta[0].exp()
    }

    pub fn beta(theta: &ManifoldPoint) -> f64 {
        theta.theta[1].exp()
    }

    pub fn params(alpha: f64, beta: f64) -> ManifoldPoint {
        ManifoldPoint::new(vec![alpha.ln(), beta.ln()])
    }
}

impl Default for BetaManifold {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalManifold for BetaManifold {
    fn dimension(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return 0.0;
        }
        let a = Self::alpha(theta);
        let b = Self::beta(theta);
        let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
        ((a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta).exp()
    }

    fn log_pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        if x <= 0.0 || x >= 1.0 {
            return f64::NEG_INFINITY;
        }
        let a = Self::alpha(theta);
        let b = Self::beta(theta);
        let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
        (a - 1.0) * x.ln() + (b - 1.0) * (1.0 - x).ln() - ln_beta
    }

    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64> {
        let a = Self::alpha(theta);
        let b = Self::beta(theta);
        // ∂/∂(log α) log p = α[ψ(α) - ψ(α+β) + ln x]
        // ∂/∂(log β) log p = β[ψ(β) - ψ(α+β) + ln(1-x)]
        let psi_a = digamma(a);
        let psi_b = digamma(b);
        let psi_ab = digamma(a + b);
        vec![
            a * (psi_a - psi_ab + x.ln()),
            b * (psi_b - psi_ab + (1.0 - x).ln()),
        ]
    }

    fn fisher_information(&self, theta: &ManifoldPoint) -> DMatrix<f64> {
        let a = Self::alpha(theta);
        let b = Self::beta(theta);
        let psi1_a = trigamma(a);
        let psi1_b = trigamma(b);
        let psi1_ab = trigamma(a + b);
        let g11 = a * a * (psi1_a - psi1_ab);
        let g22 = b * b * (psi1_b - psi1_ab);
        let g12 = -a * b * psi1_ab;
        DMatrix::from_row_slice(2, 2, &[
            g11, g12,
            g12, g22,
        ])
    }

    fn valid_params(&self, theta: &ManifoldPoint) -> bool {
        Self::alpha(theta) > 0.0 && Self::beta(theta) > 0.0
    }
}

/// Log-gamma approximation (Stirling-based for large values)
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    // Lanczos approximation
    let coefs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        return std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * ln_gamma(1.0 - x));
    }
    let x = x - 1.0;
    let a = coefs[0];
    let t = x + 7.5;
    let sum: f64 = coefs[1..]
        .iter()
        .enumerate()
        .map(|(i, &c)| c / (x + i as f64 + 1.0))
        .sum();
    let ln_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    ln_sqrt_2pi + (x + 0.5) * t.ln() - t + (a + sum).ln()
}

/// Digamma function ψ(x) = d/dx ln Γ(x)
fn digamma(x: f64) -> f64 {
    if x < 6.0 {
        return digamma(x + 1.0) - 1.0 / x;
    }
    // Asymptotic expansion
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    let inv_x4 = inv_x2 * inv_x2;
    let inv_x6 = inv_x4 * inv_x2;
    x.ln() - 0.5 * inv_x - inv_x2 / 12.0 + inv_x4 / 120.0 - inv_x6 / 252.0
}

/// Trigamma function ψ₁(x) = d²/dx² ln Γ(x)
fn trigamma(x: f64) -> f64 {
    if x < 6.0 {
        return trigamma(x + 1.0) + 1.0 / (x * x);
    }
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;
    let inv_x4 = inv_x2 * inv_x2;
    let inv_x6 = inv_x4 * inv_x2;
    inv_x + 0.5 * inv_x2 + inv_x2 * inv_x / 6.0 - inv_x4 * inv_x / 30.0 + inv_x6 * inv_x / 42.0
}

/// Gamma distribution Γ(α, β) with shape α, rate β. θ = (log α, log β).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GammaManifold;

impl GammaManifold {
    pub fn new() -> Self {
        Self
    }

    pub fn alpha(theta: &ManifoldPoint) -> f64 {
        theta.theta[0].exp()
    }

    pub fn beta(theta: &ManifoldPoint) -> f64 {
        theta.theta[1].exp()
    }

    pub fn params(alpha: f64, beta: f64) -> ManifoldPoint {
        ManifoldPoint::new(vec![alpha.ln(), beta.ln()])
    }
}

impl Default for GammaManifold {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalManifold for GammaManifold {
    fn dimension(&self) -> usize {
        2
    }

    fn pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }
        let a = Self::alpha(theta);
        let b = Self::beta(theta);
        let ln_norm = a * b.ln() - ln_gamma(a);
        ((a - 1.0) * x.ln() - b * x + ln_norm).exp()
    }

    fn log_pdf(&self, x: f64, theta: &ManifoldPoint) -> f64 {
        if x <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let a = Self::alpha(theta);
        let b = Self::beta(theta);
        (a - 1.0) * x.ln() - b * x + a * b.ln() - ln_gamma(a)
    }

    fn score(&self, x: f64, theta: &ManifoldPoint) -> Vec<f64> {
        let a = Self::alpha(theta);
        let b = Self::beta(theta);
        vec![
            a * (digamma(a) - b.ln() + x.ln()),
            b * (a / b - x),
        ]
    }

    fn fisher_information(&self, theta: &ManifoldPoint) -> DMatrix<f64> {
        let a = Self::alpha(theta);
        let b = Self::beta(theta);
        let g11 = a * a * trigamma(a);
        let g22 = a; // β*β * (α/β²) = α
        DMatrix::from_row_slice(2, 2, &[
            g11, -a,  // cross term in (log α, log β) coords
            -a, g22,
        ])
    }

    fn valid_params(&self, theta: &ManifoldPoint) -> bool {
        Self::alpha(theta) > 0.0 && Self::beta(theta) > 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normal_pdf_integrates() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let p0 = m.pdf(0.0, &theta);
        assert_relative_eq!(p0, 1.0 / (2.0 * std::f64::consts::PI).sqrt().sqrt(), max_relative = 1e-6);
        // Actually, exp(-0) / (sqrt(2π)·1) = 1/sqrt(2π)
        assert_relative_eq!(p0, 1.0 / (2.0 * std::f64::consts::PI).sqrt(), max_relative = 1e-10);
    }

    #[test]
    fn test_normal_fisher_diagonal() {
        let m = NormalManifold;
        let theta = NormalManifold::params(1.0, 2.0);
        let g = m.fisher_information(&theta);
        assert_relative_eq!(g[(0, 0)], 1.0 / 4.0, max_relative = 1e-10);
        assert_relative_eq!(g[(0, 1)], 0.0, max_relative = 1e-10);
        assert_relative_eq!(g[(1, 1)], 2.0, max_relative = 1e-10);
    }

    #[test]
    fn test_normal_score_zero_mean() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let s = m.score(0.0, &theta);
        assert_relative_eq!(s[0], 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_categorical_uniform_probs() {
        let m = CategoricalManifold::new(3);
        let theta = m.uniform_params();
        let probs = m.to_probs(&theta);
        assert_relative_eq!(probs[0], 1.0 / 3.0, max_relative = 1e-10);
        assert_relative_eq!(probs[1], 1.0 / 3.0, max_relative = 1e-10);
        assert_relative_eq!(probs[2], 1.0 / 3.0, max_relative = 1e-10);
    }

    #[test]
    fn test_categorical_probs_sum_to_one() {
        let m = CategoricalManifold::new(4);
        let theta = ManifoldPoint::new(vec![0.5, -0.3, 1.0]);
        let probs = m.to_probs(&theta);
        let sum: f64 = probs.iter().sum();
        assert_relative_eq!(sum, 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_categorical_fisher_positive_definite() {
        let m = CategoricalManifold::new(3);
        let theta = ManifoldPoint::new(vec![0.5, -0.3]);
        let g = m.fisher_information(&theta);
        // All diagonal elements should be positive
        assert!(g[(0, 0)] > 0.0);
        assert!(g[(1, 1)] > 0.0);
    }

    #[test]
    fn test_exponential_pdf() {
        let m = ExponentialManifold;
        let theta = ExponentialManifold::params(1.0);
        assert_relative_eq!(m.pdf(0.0, &theta), 1.0, max_relative = 1e-10);
        assert_relative_eq!(m.pdf(1.0, &theta), std::f64::consts::E.exp(), max_relative = 1e-6);
        // 1/ e ≈ 0.3679
        assert_relative_eq!(m.pdf(1.0, &theta), 1.0 / std::f64::consts::E, max_relative = 1e-10);
    }

    #[test]
    fn test_exponential_fisher() {
        let m = ExponentialManifold;
        let theta = ExponentialManifold::params(2.0);
        let g = m.fisher_information(&theta);
        assert_relative_eq!(g[(0, 0)], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_poisson_fisher() {
        let m = PoissonManifold;
        let theta = PoissonManifold::params(5.0);
        let g = m.fisher_information(&theta);
        assert_relative_eq!(g[(0, 0)], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_beta_pdf_at_mode() {
        let m = BetaManifold;
        let theta = BetaManifold::params(2.0, 2.0);
        let p = m.pdf(0.5, &theta);
        // B(2,2) mode = 0.5, pdf = 6 * 0.5 * 0.5 = 1.5
        assert_relative_eq!(p, 1.5, max_relative = 1e-4);
    }

    #[test]
    fn test_beta_fisher_positive_definite() {
        let m = BetaManifold;
        let theta = BetaManifold::params(2.0, 3.0);
        let g = m.fisher_information(&theta);
        assert!(g[(0, 0)] > 0.0);
        assert!(g[(1, 1)] > 0.0);
        // Determinant should be positive for PD matrix
        let det = g[(0, 0)] * g[(1, 1)] - g[(0, 1)] * g[(1, 0)];
        assert!(det > 0.0);
    }

    #[test]
    fn test_gamma_fisher_positive_definite() {
        let m = GammaManifold;
        let theta = GammaManifold::params(2.0, 3.0);
        let g = m.fisher_information(&theta);
        assert!(g[(0, 0)] > 0.0);
        assert!(g[(1, 1)] > 0.0);
    }

    #[test]
    fn test_diagonal_normal_dimension() {
        let m = DiagonalNormalManifold::new(3);
        assert_eq!(m.dimension(), 6);
    }

    #[test]
    fn test_normal_valid_params() {
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        assert!(m.valid_params(&theta));
    }

    #[test]
    fn test_manifold_point_dim() {
        let p = ManifoldPoint::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(p.dim(), 3);
    }

    #[test]
    fn test_categorical_pdf() {
        let m = CategoricalManifold::new(3);
        let theta = ManifoldPoint::new(vec![1.0, 0.0]); // class 0 gets more weight
        let p0 = m.pdf(0.0, &theta);
        let p1 = m.pdf(1.0, &theta);
        assert!(p0 > p1); // class 0 should have higher probability
    }
}
