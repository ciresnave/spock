//! # Spock — Raw Vulkan API Bindings for Rust
//!
//! Spock generates complete, raw Vulkan API bindings directly from the official
//! Vulkan XML specification (`vk.xml`). Every type, constant, struct, enum,
//! function pointer, and dispatch table is derived from the spec — nothing is
//! hardcoded.
//!
//! ## Features
//!
//! - **100% Generated from vk.xml** — swap the XML file and rebuild to target
//!   any Vulkan version from 1.2.175 onward
//! - **Complete API Coverage** — all core types, extensions, and function pointers
//! - **Extension Enum Values** — extension-added enum values are merged into
//!   base enums automatically
//! - **Dispatch Tables** — generated `VkEntryDispatchTable`, `VkInstanceDispatchTable`,
//!   and `VkDeviceDispatchTable` for runtime function loading
//! - **Version Macros** — `vk_make_api_version`, `vk_api_version_major`, etc.
//!   are transpiled from the C macro definitions in vk.xml
//! - **Auto-Download** — with the `fetch-spec` feature, vk.xml is downloaded
//!   from the Khronos GitHub repository automatically
//!
//! ## Supported Vulkan Versions
//!
//! Spock supports Vulkan specification versions **1.2.175** through the latest
//! release. Version 1.2.175 is the minimum because it introduced the
//! `VK_MAKE_API_VERSION` / `VK_API_VERSION_*` macros that replaced the
//! deprecated `VK_MAKE_VERSION` / `VK_VERSION_*` macros.
//!
//! ## Providing vk.xml
//!
//! The build script resolves vk.xml in this order:
//!
//! 1. **`VK_XML_PATH` env var** — point to any local vk.xml file
//! 2. **Local copy** — `spec/registry/Vulkan-Docs/xml/vk.xml` relative to the workspace
//! 3. **Auto-download** — requires the `fetch-spec` feature:
//!    ```bash
//!    cargo build -p spock --features fetch-spec
//!    ```
//!    Set `VK_VERSION` to pin a specific version:
//!    ```bash
//!    VK_VERSION=1.3.250 cargo build -p spock --features fetch-spec
//!    ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use spock::raw::bindings::*;
//!
//! let app_info = VkApplicationInfo {
//!     sType: VkStructureType::STRUCTURE_TYPE_APPLICATION_INFO,
//!     pNext: std::ptr::null(),
//!     pApplicationName: b"My App\0".as_ptr() as *const i8,
//!     applicationVersion: vk_make_api_version(0, 1, 0, 0),
//!     pEngineName: b"My Engine\0".as_ptr() as *const i8,
//!     engineVersion: vk_make_api_version(0, 1, 0, 0),
//!     apiVersion: VK_API_VERSION_1_0,
//! };
//! ```
//!
//! ## Safety
//!
//! All Vulkan functions are `unsafe` as they directly expose the C API.
//! Users are responsible for parameter validation, memory management,
//! thread safety, and Vulkan object lifecycle management.

// Re-export all raw bindings
pub mod raw;

// Re-export commonly used items at crate root for convenience
pub use raw::bindings::*;

/// Version information for these bindings
pub mod version {
    /// The version of these bindings
    pub const BINDINGS_VERSION: &str = env!("CARGO_PKG_VERSION");

    /// Build timestamp (set during build)
    pub const BUILD_TIMESTAMP: &str = env!("BUILD_TIMESTAMP");
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod unit_tests;
