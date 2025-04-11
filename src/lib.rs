// re-export the generated types
pub mod simulation {
    tonic::include_proto!("simulation");
}

// modules
mod common_consts;
mod common_portfolio_evolution_ds;
mod sampling;
mod portfolio;