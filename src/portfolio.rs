use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use bincode::{Decode, Encode};

#[derive(Debug, Clone, Serialize, Deserialize, Decode, Encode)]
pub struct Portfolio {
    pub id: usize,
    pub rank: Option<usize>,
    pub crowding_distance: Option<f64>,
    pub weights: Vec<f64>,
    pub average_returns: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
}

impl PartialEq for Portfolio {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl PartialOrd for Portfolio {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // if ID is the same return equal
        if self.id == other.id {
            return Some(std::cmp::Ordering::Equal);
        }

        // Compare based on rank
        if self.rank != other.rank {
            return self.rank.partial_cmp(&other.rank);
        }
        // If Rank is the same compare based on Crowding_Distance
        match (self.crowding_distance, other.crowding_distance) {
            (Some(self_distance), Some(other_distance)) => {
                self_distance.partial_cmp(&other_distance)
            }
            _ => panic!("Crowding distance is None for one or both portfolios."), //shouldn't happen after processing
        }
    }
}
impl Eq for Portfolio {}
static PORTFOLIO_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl Portfolio {
    pub fn new(
        weights: Vec<f64>,
        average_returns: f64,
        volatility: f64,
        sharpe_ratio: f64,
    ) -> Self {
        let id = PORTFOLIO_ID_COUNTER.fetch_add(1, Ordering::SeqCst);
        let rank = None;
        let crowding_distance = None;
        Portfolio {
            id,
            rank,
            crowding_distance,
            weights,
            average_returns,
            volatility,
            sharpe_ratio,
        }
    }

    fn to_metrics_vector(&self) -> Vec<f64> {
        // We negate the self.volatility to make maximization the global goal
        vec![self.average_returns, -self.volatility, self.sharpe_ratio]
    }

    pub fn is_dominated_by(&self, other: &Portfolio) -> bool {
        let self_metrics = self.to_metrics_vector();
        let other_metrics = other.to_metrics_vector();

        // Check if 'other' is at least as good as 'self' in all objectives
        let other_is_at_least_as_good_in_all = self_metrics
            .iter()
            .zip(other_metrics.iter())
            .all(|(&self_metric, &other_metric)| other_metric >= self_metric);

        // Check if 'other' is strictly better than 'self' in at least one objective
        let other_is_strictly_better_in_one = self_metrics
            .iter()
            .zip(other_metrics.iter())
            .any(|(&self_metric, &other_metric)| other_metric > self_metric);

        // 'self' is dominated by 'other' if 'other' is at least as good in all
        // objectives AND strictly better in at least one objective
        other_is_at_least_as_good_in_all && other_is_strictly_better_in_one
    }
}
