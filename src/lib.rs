mod mp_analyzer;
mod frame;
mod pymod;

#[doc(inline)]
pub use pymod::*;
pub use frame::*;

#[derive(Debug, Clone)]
pub struct AnalysisError {
    pub msg: String
}
