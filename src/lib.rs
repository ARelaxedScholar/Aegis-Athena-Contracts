// re-export the generated types
pub mod simulation {
    tonic::include_proto!("simulation");
}

// modules
pub mod common_consts;
pub mod common_portfolio_evolution_ds;
pub mod portfolio;
pub mod sampling;
