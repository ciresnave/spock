pub mod assembler;
pub mod generator_modules;
pub mod logging;
pub mod provisional;
pub mod type_integration;

/// Escape a comment line so rustdoc doesn't try to parse markdown / HTML.
///
/// Vulkan's vk.xml comments use prose conventions that collide with rustdoc:
///
/// - Asciidoc cross-references like `<<devsandqueues-lost-device>>` look like
///   invalid HTML tags to rustdoc and produce `rustdoc::invalid_html_tags`
///   warnings.
/// - Bracketed text like `BUFFER[_DYNAMIC]` looks like an intra-doc link
///   with no resolvable target and produces `rustdoc::broken_intra_doc_links`
///   warnings.
///
/// We escape `<`, `>`, `[`, and `]` to their HTML entity equivalents so they
/// render as literal characters in the generated docs without rustdoc trying
/// to interpret them.
pub fn sanitize_doc_line(line: &str) -> String {
    line.trim()
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('[', "&#91;")
        .replace(']', "&#93;")
}

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
