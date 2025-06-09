use crate::*;

pub type SimulationResult = ();

pub struct Simulation<'a> {
    world_state: &'a WorldState,
}

impl<'a> Simulation<'a> {
    pub fn new(world_state: &'a WorldState) -> Self {
        Self {
            world_state
        }
    }

    pub fn run(mut self) -> SimulationResult {
        

        todo!()
    }
}
