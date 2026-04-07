//! Helpers for converting `VkResult` into Rust's `Result` type.
//!
//! Vulkan functions return `VkResult` codes where `SUCCESS` (0) means success
//! and any other value indicates either a non-fatal status (positive values)
//! or an error (negative values). These helpers make it easy to use Rust's
//! `?` operator with Vulkan calls.
//!
//! # Example
//!
//! ```ignore
//! use spock::raw::bindings::*;
//! use spock::raw::result::VkResultExt;
//!
//! unsafe fn create_instance(create_fn: vkCreateInstance) -> Result<VkInstance, VkResult> {
//!     let info = VkInstanceCreateInfo::default();
//!     let mut instance = std::ptr::null_mut();
//!     create_fn(&info, std::ptr::null(), &mut instance).into_result()?;
//!     Ok(instance)
//! }
//! ```

use crate::raw::bindings::VkResult;

/// Extension trait that converts `VkResult` into a Rust `Result`.
pub trait VkResultExt {
    /// Convert `VkResult::SUCCESS` to `Ok(())` and any other value to `Err(self)`.
    fn into_result(self) -> Result<(), VkResult>;

    /// Returns true if the result indicates success (SUCCESS).
    /// Note that some "non-error" status codes like NOT_READY, TIMEOUT, INCOMPLETE,
    /// and SUBOPTIMAL_KHR are technically positive but indicate work was not completed.
    fn is_success(self) -> bool;

    /// Returns true if the result indicates an error (any negative value).
    fn is_error(self) -> bool;
}

impl VkResultExt for VkResult {
    #[inline]
    fn into_result(self) -> Result<(), VkResult> {
        if self == VkResult::SUCCESS {
            Ok(())
        } else {
            Err(self)
        }
    }

    #[inline]
    fn is_success(self) -> bool {
        self == VkResult::SUCCESS
    }

    #[inline]
    fn is_error(self) -> bool {
        (self as i32) < 0
    }
}

// Implement std::error::Error for VkResult so it works with `?` in functions
// returning Box<dyn std::error::Error>. The generated VkResult already has
// Debug + Display impls from the codegen.
impl std::error::Error for VkResult {}

/// Convenience macro that calls a Vulkan function and propagates errors via `?`.
///
/// # Example
///
/// ```ignore
/// use spock::vk_check;
/// // Inside an unsafe fn that returns Result<_, VkResult>:
/// vk_check!(create_instance(&info, std::ptr::null(), &mut instance))?;
/// ```
#[macro_export]
macro_rules! vk_check {
    ($call:expr) => {{
        use $crate::raw::result::VkResultExt;
        ($call).into_result()
    }};
}
