mod rstft_analyzer;
mod frame;
mod pymod;
mod rfft_analyzer;

#[doc(inline)]
pub use pymod::*;
pub use frame::*;

#[derive(Debug, Clone)]
pub struct AnalysisError {
    pub msg: String
}
