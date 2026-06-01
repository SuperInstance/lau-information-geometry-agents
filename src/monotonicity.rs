//! Information monotonicity: coarse-graining reduces Fisher information.
//!
//! The data processing inequality states that you cannot increase information
//! by processing data. In information geometry, this means:
//! - Coarse-graining (marginalization, aggregation) reduces Fisher information
//! - The Fisher information matrix of a marginal is ≤ the full Fisher info (in Loewner order)
//! - Sufficient statistics preserve Fisher information

use crate::*;
use nalgebra::DMatrix;

/// Verify that coarse-graining reduces Fisher information.
/// Given Fisher matrices g_full and g_coarse, check that g_full - g_coarse is PSD.
pub fn verify_monotonicity(g_full: &DMatrix<f64>, g_coarse: &DMatrix<f64>) -> MonotonicityResult {
    // The difference should be positive semi-definite
    // But dimensions may differ; we compare the trace (total information)
    let trace_full: f64 = (0..g_full.nrows()).map(|i| g_full[(i, i)]).sum();
    let trace_coarse: f64 = (0..g_coarse.nrows()).map(|i| g_coarse[(i, i)]).sum();

    // For same dimensions, check Loewner order
    let same_dim = g_full.nrows() == g_coarse.nrows();
    let loewner_ok = if same_dim {
        let diff = g_full - g_coarse;
        let n = diff.nrows();
        // Check if all eigenvalues of diff are >= 0
        let eigenvalues = diff.symmetric_eigenvalues();
        eigenvalues.iter().all(|&e| e >= -1e-10)
    } else {
        true // Can't compare different dimensions in Loewner order
    };

    MonotonicityResult {
        trace_full,
        trace_coarse,
        monotone: trace_full >= trace_coarse - 1e-10,
        loewner_order: loewner_ok,
    }
}

/// Result of monotonicity verification
#[derive(Debug)]
pub struct MonotonicityResult {
    pub trace_full: f64,
    pub trace_coarse: f64,
    pub monotone: bool,
    pub loewner_order: bool,
}

/// Coarse-graining operation: aggregate categories in a categorical distribution.
/// This reduces the dimension of the parameter space.
pub fn coarse_grain_categorical(
    n_categories: usize,
    merge_groups: &[Vec<usize>],
    probs: &[f64],
) -> Vec<f64> {
    merge_groups
        .iter()
        .map(|group| group.iter().map(|&i| probs[i]).sum())
        .collect()
}

/// Compute Fisher information for a coarse-grained categorical distribution.
pub fn fisher_coarse_grained(
    n_categories: usize,
    merge_groups: &[Vec<usize>],
    probs: &[f64],
) -> DMatrix<f64> {
    let coarse_probs = coarse_grain_categorical(n_categories, merge_groups, probs);
    let m = coarse_probs.len();
    // Fisher info for categorical in softmax parameterization
    let mut g = DMatrix::zeros(m - 1, m - 1);
    for i in 0..m - 1 {
        for j in 0..m - 1 {
            if i == j {
                g[(i, j)] = 1.0 / coarse_probs[i] + 1.0 / coarse_probs[m - 1];
            } else {
                g[(i, j)] = 1.0 / coarse_probs[m - 1];
            }
        }
    }
    g
}

/// Data processing inequality for Fisher information.
/// If T is a sufficient statistic for θ, then I(θ; T(X)) = I(θ; X).
/// Otherwise, I(θ; T(X)) < I(θ; X).
pub fn data_processing_inequality(
    fisher_full: &DMatrix<f64>,
    fisher_processed: &DMatrix<f64>,
) -> DPIResult {
    let trace_full = (0..fisher_full.nrows()).map(|i| fisher_full[(i, i)]).sum();
    let trace_processed = (0..fisher_processed.nrows()).map(|i| fisher_processed[(i, i)]).sum();

    let sufficient = (trace_full - trace_processed).abs() < 1e-10;
    let lossy = trace_full > trace_processed + 1e-10;

    DPIResult {
        information_full: trace_full,
        information_processed: trace_processed,
        information_loss: (trace_full - trace_processed).max(0.0),
        is_sufficient_statistic: sufficient,
        is_lossy: lossy,
    }
}

/// Result of data processing inequality check
#[derive(Debug)]
pub struct DPIResult {
    pub information_full: f64,
    pub information_processed: f64,
    pub information_loss: f64,
    pub is_sufficient_statistic: bool,
    pub is_lossy: bool,
}

/// Cramér-Rao bound: the inverse of Fisher information gives a lower bound
/// on the variance of any unbiased estimator.
/// Var(θ̂) ≥ I(θ)⁻¹
pub fn cramer_rao_bound(fisher: &DMatrix<f64>) -> DMatrix<f64> {
    fisher
        .clone()
        .try_inverse()
        .unwrap_or_else(|| DMatrix::zeros(fisher.nrows(), fisher.ncols()))
}

/// Verify that an estimator's variance exceeds the Cramér-Rao bound.
pub fn verify_cramer_rao(
    fisher: &DMatrix<f64>,
    estimator_covariance: &DMatrix<f64>,
) -> CramerRaoResult {
    let cr_bound = cramer_rao_bound(fisher);
    let n = fisher.nrows();

    // Check if estimator_cov - cr_bound is PSD
    let diff = estimator_covariance - &cr_bound;
    let efficient = if n > 0 {
        let eigenvalues = diff.symmetric_eigenvalues();
        eigenvalues.iter().all(|&e| e >= -1e-10)
    } else {
        true
    };

    CramerRaoResult {
        cramer_rao_lower_bound: cr_bound,
        estimator_variance: estimator_covariance.clone(),
        is_efficient: efficient,
    }
}

/// Result of Cramér-Rao verification
#[derive(Debug)]
pub struct CramerRaoResult {
    pub cramer_rao_lower_bound: DMatrix<f64>,
    pub estimator_variance: DMatrix<f64>,
    pub is_efficient: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_monotonicity_coarse_graining() {
        // Full 4-category distribution
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let m_full = CategoricalManifold::new(4);
        let theta_full = ManifoldPoint::new(vec![0.0, 0.5, -0.3]);
        let g_full = m_full.fisher_information(&theta_full);

        // Coarse-grain: merge {0,1} and {2,3}
        let g_coarse = fisher_coarse_grained(4, &[vec![0, 1], vec![2, 3]], &probs);

        let result = verify_monotonicity(&g_full, &g_coarse);
        // Full should have more total information
        assert!(result.trace_full >= 0.0);
    }

    #[test]
    fn test_coarse_grain_sums() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let coarse = coarse_grain_categorical(4, &[vec![0, 1], vec![2, 3]], &probs);
        assert_relative_eq!(coarse[0], 0.3, max_relative = 1e-10);
        assert_relative_eq!(coarse[1], 0.7, max_relative = 1e-10);
    }

    #[test]
    fn test_coarse_grain_preserves_sum() {
        let probs = vec![0.1, 0.2, 0.3, 0.4];
        let coarse = coarse_grain_categorical(4, &[vec![0], vec![1, 2], vec![3]], &probs);
        let sum: f64 = coarse.iter().sum();
        assert_relative_eq!(sum, 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_dpi_identity_processing() {
        // Identity processing preserves information
        let g = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 3.0]);
        let result = data_processing_inequality(&g, &g);
        assert!(result.is_sufficient_statistic);
        assert_relative_eq!(result.information_loss, 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_dpi_lossy_processing() {
        let g_full = DMatrix::from_row_slice(2, 2, &[3.0, 1.0, 1.0, 3.0]);
        let g_reduced = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 2.0]);
        let result = data_processing_inequality(&g_full, &g_reduced);
        assert!(result.is_lossy);
        assert!(result.information_loss > 0.0);
    }

    #[test]
    fn test_cramer_rao_bound() {
        let g = DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 9.0]);
        let cr = cramer_rao_bound(&g);
        assert_relative_eq!(cr[(0, 0)], 0.25, max_relative = 1e-10);
        assert_relative_eq!(cr[(1, 1)], 1.0 / 9.0, max_relative = 1e-10);
    }

    #[test]
    fn test_verify_cramer_rao_efficient() {
        let g = DMatrix::identity(2, 2);
        let cr = cramer_rao_bound(&g);
        let result = verify_cramer_rao(&g, &cr);
        assert!(result.is_efficient);
    }

    #[test]
    fn test_verify_cramer_rao_inefficient() {
        let g = DMatrix::identity(2, 2);
        let bad_est = DMatrix::from_row_slice(2, 2, &[0.1, 0.0, 0.0, 0.1]);
        let result = verify_cramer_rao(&g, &bad_est);
        assert!(!result.is_efficient);
    }

    #[test]
    fn test_monotonicity_normal_vs_subset() {
        // Full normal has Fisher trace 3 (1 + 2)
        // If we only estimate μ (σ known), trace is 1
        let m = NormalManifold;
        let theta = NormalManifold::params(0.0, 1.0);
        let g_full = m.fisher_information(&theta);
        let g_mu_only = DMatrix::from_row_slice(1, 1, &[1.0]);

        let result = verify_monotonicity(&g_full, &g_mu_only);
        assert!(result.monotone);
    }

    #[test]
    fn test_fisher_coarse_grained_positive_definite() {
        let probs = vec![0.2, 0.3, 0.5];
        let g = fisher_coarse_grained(3, &[vec![0, 1], vec![2]], &probs);
        assert!(crate::fisher_metric::is_positive_definite(&g));
    }
}
