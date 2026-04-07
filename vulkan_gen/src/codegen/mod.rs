pub mod assembler;
pub mod generator_modules;
pub mod logging;
pub mod provisional;
pub mod type_integration;

pub use assembler::AssemblerError;

/// Error type for code generation operations
#[derive(Debug, thiserror::Error)]
pub enum CodegenError {
    #[error("Assembler error: {0}")]
    Assembler(#[from] AssemblerError),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid input: {message}")]
    InvalidInput { message: String },

    #[error("Generation failed: {message}")]
    GenerationFailed { message: String },
}

pub type CodegenResult<T> = Result<T, CodegenError>;
