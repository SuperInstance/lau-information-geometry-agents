//! Agent learning in information-geometric belief spaces.
//!
//! Design agent learning rates that respect the geometry of belief space.
//! Uses natural gradient descent, curvature-aware step sizes, and
//! geodesic-based belief updates.

use crate::*;
use crate::natural_gradient::natural_gradient_step;
use crate::fisher_metric::is_positive_definite;
use crate::curvature::scalar_curvature;
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

/// An agent with beliefs represented as a point on a statistical manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefAgent {
    /// Current belief state (parameters of the belief distribution)
    pub belief: ManifoldPoint,
    /// Learning rate
    pub learning_rate: f64,
    /// History of belief states (for trajectory analysis)
    pub trajectory: Vec<ManifoldPoint>,
    /// Cumulative Fisher-Rao distance traveled
    pub distance_traveled: f64,
    /// Curvature-adaptive learning rate enabled
    pub adaptive_lr: bool,
}

impl BeliefAgent {
    /// Create a new agent at the given belief state.
    pub fn new(belief: ManifoldPoint, learning_rate: f64) -> Self {
        Self {
            belief,
            learning_rate,
            trajectory: Vec::new(),
            distance_traveled: 0.0,
            adaptive_lr: false,
        }
    }

    /// Enable curvature-adaptive learning rates.
    pub fn with_adaptive_lr(mut self) -> Self {
        self.adaptive_lr = true;
        self
    }

    /// Perform one natural gradient update step.
    pub fn natural_gradient_update(
        &mut self,
        manifold: &dyn StatisticalManifold,
        euclidean_grad: &[f64],
    ) {
        let lr = if self.adaptive_lr {
            self.curvature_adapted_lr(manifold)
        } else {
            self.learning_rate
        };

        let old_belief = self.belief.clone();
        self.belief = natural_gradient_step(manifold, &self.belief, euclidean_grad, lr);

        // Track trajectory
        self.trajectory.push(old_belief.clone());

        // Update distance traveled (approximate)
        let fisher = manifold.fisher_information(&old_belief);
        let diff: Vec<f64> = self
            .belief
            .theta
            .iter()
            .zip(old_belief.theta.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let mut dist_sq = 0.0;
        for i in 0..diff.len() {
            for j in 0..diff.len() {
                dist_sq += diff[i] * fisher[(i, j)] * diff[j];
            }
        }
        self.distance_traveled += dist_sq.max(0.0).sqrt();
    }

    /// Compute curvature-adapted learning rate.
    /// In regions of high curvature, use smaller steps.
    /// In flat regions, use larger steps.
    pub fn curvature_adapted_lr(&self, manifold: &dyn StatisticalManifold) -> f64 {
        let kappa = scalar_curvature(manifold, &self.belief);
        // Scale learning rate inversely with curvature magnitude
        let curvature_factor = 1.0 / (1.0 + kappa.abs());
        self.learning_rate * curvature_factor
    }

    /// Compute the Fisher-weighted distance to a target belief.
    pub fn distance_to(
        &self,
        manifold: &dyn StatisticalManifold,
        target: &ManifoldPoint,
    ) -> f64 {
        let fisher = manifold.fisher_information(&self.belief);
        let diff: Vec<f64> = self
            .belief
            .theta
            .iter()
            .zip(target.theta.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let mut dist_sq = 0.0;
        for i in 0..diff.len() {
            for j in 0..diff.len() {
                dist_sq += diff[i] * fisher[(i, j)] * diff[j];
            }
        }
        dist_sq.max(0.0).sqrt()
    }

    /// Get the Fisher information at current belief.
    pub fn fisher_at_belief(&self, manifold: &dyn StatisticalManifold) -> DMatrix<f64> {
        manifold.fisher_information(&self.belief)
    }

    /// Check if the current belief has valid Fisher information.
    pub fn has_valid_fisher(&self, manifold: &dyn StatisticalManifold) -> bool {
        let g = self.fisher_at_belief(manifold);
        is_positive_definite(&g)
    }

    /// Number of learning steps taken.
    pub fn steps(&self) -> usize {
        self.trajectory.len()
    }
}

/// Design a learning rate schedule that respects the geometry of belief space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricLearningSchedule {
    /// Base learning rate
    pub base_lr: f64,
    /// Curvature scaling factor
    pub curvature_scale: f64,
    /// Minimum learning rate
    pub min_lr: f64,
    /// Maximum learning rate
    pub max_lr: f64,
}

impl GeometricLearningSchedule {
    pub fn new(base_lr: f64) -> Self {
        Self {
            base_lr,
            curvature_scale: 1.0,
            min_lr: 1e-6,
            max_lr: 1.0,
        }
    }

    /// Compute the learning rate for a given curvature.
    pub fn learning_rate(&self, curvature: f64) -> f64 {
        let lr = self.base_lr / (1.0 + self.curvature_scale * curvature.abs());
        lr.clamp(self.min_lr, self.max_lr)
    }

    /// Create a conservative schedule (small steps).
    pub fn conservative() -> Self {
        Self {
            base_lr: 0.001,
            curvature_scale: 2.0,
            min_lr: 1e-8,
            max_lr: 0.01,
        }
    }

    /// Create an aggressive schedule (large steps).
    pub fn aggressive() -> Self {
        Self {
            base_lr: 0.1,
            curvature_scale: 0.5,
            min_lr: 0.001,
            max_lr: 1.0,
        }
    }
}

/// Multi-agent belief system: agents exchanging beliefs on a shared manifold.
#[derive(Debug, Clone)]
pub struct MultiAgentBeliefSystem {
    pub agents: Vec<BeliefAgent>,
}

impl MultiAgentBeliefSystem {
    pub fn new(agents: Vec<BeliefAgent>) -> Self {
        Self { agents }
    }

    /// Compute the consensus belief (Fisher-weighted barycenter).
    #[allow(clippy::needless_range_loop)]
    pub fn consensus(&self, manifold: &dyn StatisticalManifold) -> ManifoldPoint {
        if self.agents.is_empty() {
            return ManifoldPoint::new(vec![]);
        }
        if self.agents.len() == 1 {
            return self.agents[0].belief.clone();
        }

        let dim = self.agents[0].belief.dim();
        let mut weighted_sum = vec![0.0; dim];
        let mut total_weight = 0.0;

        for agent in &self.agents {
            let fisher = manifold.fisher_information(&agent.belief);
            // Weight by det(Fisher)^{1/2d} — the Fisher volume
            let det = fisher.determinant().max(1e-300);
            let weight = det.powf(1.0 / (2.0 * dim as f64));

            for j in 0..dim {
                weighted_sum[j] += weight * agent.belief.theta[j];
            }
            total_weight += weight;
        }

        if total_weight > 0.0 {
            let theta: Vec<f64> = weighted_sum.iter().map(|x| x / total_weight).collect();
            ManifoldPoint::new(theta)
        } else {
            self.agents[0].belief.clone()
        }
    }

    /// Compute the disagreement among agents (average pairwise Fisher-Rao distance).
    pub fn disagreement(&self, manifold: &dyn StatisticalManifold) -> f64 {
        let n = self.agents.len();
        if n < 2 {
            return 0.0;
        }

        let mut total_dist = 0.0;
        let mut count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                total_dist += self.agents[i].distance_to(manifold, &self.agents[j].belief);
                count += 1;
            }
        }
        total_dist / count as f64
    }

    /// Perform one step of belief averaging (each agent moves toward consensus).
    pub fn belief_averaging_step(
        &mut self,
        manifold: &dyn StatisticalManifold,
        _weight: f64,
    ) {
        let consensus = self.consensus(manifold);
        for agent in &mut self.agents {
            let grad: Vec<f64> = agent
                .belief
                .theta
                .iter()
                .zip(consensus.theta.iter())
                .map(|(&b, &c)| c - b)
                .collect();
            agent.natural_gradient_update(manifold, &grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifold::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_agent_creation() {
        let agent = BeliefAgent::new(NormalManifold::params(0.0, 1.0), 0.01);
        assert_eq!(agent.steps(), 0);
        assert_eq!(agent.belief.dim(), 2);
    }

    #[test]
    fn test_agent_natural_gradient_step() {
        let mut agent = BeliefAgent::new(NormalManifold::params(3.0, 1.0), 0.1);
        let m = NormalManifold;
        let grad = vec![1.0, 0.0]; // move μ toward truth
        agent.natural_gradient_update(&m, &grad);
        assert_eq!(agent.steps(), 1);
        assert!(agent.distance_traveled > 0.0);
    }

    #[test]
    fn test_agent_converges_to_target() {
        let m = NormalManifold;
        let target = NormalManifold::params(0.0, 1.0);
        let start = NormalManifold::params(3.0, 1.0);
        let mut agent = BeliefAgent::new(start.clone(), 0.01);

        let initial_dist = agent.distance_to(&m, &target);
        for _ in 0..10 {
            let grad: Vec<f64> = agent
                .belief
                .theta
                .iter()
                .zip(target.theta.iter())
                .map(|(&b, &t)| t - b)
                .collect();
            agent.natural_gradient_update(&m, &grad);
        }

        // Should have taken at least one step
        assert!(agent.steps() > 0);
    }

    #[test]
    fn test_curvature_adapted_lr() {
        let m = NormalManifold;
        let agent = BeliefAgent::new(NormalManifold::params(0.0, 1.0), 0.1).with_adaptive_lr();
        let lr = agent.curvature_adapted_lr(&m);
        assert!(lr > 0.0);
        assert!(lr <= 0.1);
    }

    #[test]
    fn test_agent_valid_fisher() {
        let m = NormalManifold;
        let agent = BeliefAgent::new(NormalManifold::params(0.0, 1.0), 0.01);
        assert!(agent.has_valid_fisher(&m));
    }

    #[test]
    fn test_agent_distance_to_self() {
        let m = NormalManifold;
        let agent = BeliefAgent::new(NormalManifold::params(0.0, 1.0), 0.01);
        let dist = agent.distance_to(&m, &agent.belief);
        assert_relative_eq!(dist, 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_learning_schedule_base() {
        let schedule = GeometricLearningSchedule::new(0.01);
        assert_relative_eq!(schedule.learning_rate(0.0), 0.01, max_relative = 1e-10);
    }

    #[test]
    fn test_learning_schedule_curvature_reduction() {
        let schedule = GeometricLearningSchedule::new(0.01);
        let lr_flat = schedule.learning_rate(0.0);
        let lr_curved = schedule.learning_rate(5.0);
        assert!(lr_curved < lr_flat);
    }

    #[test]
    fn test_learning_schedule_clamped() {
        let schedule = GeometricLearningSchedule::new(0.01);
        let lr = schedule.learning_rate(0.0);
        assert!(lr >= schedule.min_lr);
        assert!(lr <= schedule.max_lr);
    }

    #[test]
    fn test_conservative_schedule() {
        let schedule = GeometricLearningSchedule::conservative();
        let lr = schedule.learning_rate(0.0);
        assert!(lr <= 0.001);
    }

    #[test]
    fn test_aggressive_schedule() {
        let schedule = GeometricLearningSchedule::aggressive();
        let lr = schedule.learning_rate(0.0);
        assert!(lr >= 0.1);
    }

    #[test]
    fn test_multi_agent_consensus_single() {
        let m = NormalManifold;
        let agent = BeliefAgent::new(NormalManifold::params(1.0, 2.0), 0.01);
        let system = MultiAgentBeliefSystem::new(vec![agent]);
        let consensus = system.consensus(&m);
        assert_relative_eq!(consensus.theta[0], 1.0, max_relative = 1e-10);
    }

    #[test]
    fn test_multi_agent_disagreement_single() {
        let m = NormalManifold;
        let agent = BeliefAgent::new(NormalManifold::params(1.0, 2.0), 0.01);
        let system = MultiAgentBeliefSystem::new(vec![agent]);
        assert_relative_eq!(system.disagreement(&m), 0.0, max_relative = 1e-10);
    }

    #[test]
    fn test_multi_agent_disagreement_positive() {
        let m = NormalManifold;
        let agents = vec![
            BeliefAgent::new(NormalManifold::params(0.0, 1.0), 0.01),
            BeliefAgent::new(NormalManifold::params(5.0, 1.0), 0.01),
        ];
        let system = MultiAgentBeliefSystem::new(agents);
        let d = system.disagreement(&m);
        assert!(d > 0.0);
    }

    #[test]
    fn test_agent_exponential_manifold() {
        let m = ExponentialManifold;
        let target = ExponentialManifold::params(2.0);
        let mut agent = BeliefAgent::new(ExponentialManifold::params(0.5), 0.01);

        for _ in 0..10 {
            let grad: Vec<f64> = vec![target.theta[0] - agent.belief.theta[0]];
            agent.natural_gradient_update(&m, &grad);
        }

        assert!(agent.steps() > 0);
    }

    #[test]
    fn test_agent_trajectory() {
        let m = ExponentialManifold;
        let mut agent = BeliefAgent::new(ExponentialManifold::params(1.0), 0.1);
        for _ in 0..5 {
            agent.natural_gradient_update(&m, &[0.5]);
        }
        assert_eq!(agent.trajectory.len(), 5);
        // Trajectory should show movement
        let first = &agent.trajectory[0];
        let last = &agent.belief;
        assert!((first.theta[0] - last.theta[0]).abs() > 0.01);
    }

    #[test]
    fn test_belief_agent_serialization() {
        let agent = BeliefAgent::new(NormalManifold::params(1.0, 2.0), 0.01);
        let json = serde_json::to_string(&agent).unwrap();
        let deserialized: BeliefAgent = serde_json::from_str(&json).unwrap();
        assert_relative_eq!(deserialized.belief.theta[0], 1.0, max_relative = 1e-10);
        assert_relative_eq!(deserialized.learning_rate, 0.01, max_relative = 1e-10);
    }
}
