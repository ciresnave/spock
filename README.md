# Spock

Raw Vulkan API bindings for Rust, generated directly from the official Vulkan XML specification.

## What is Spock?

Spock is a Rust crate that generates complete Vulkan API bindings from `vk.xml`, the official machine-readable Vulkan specification maintained by Khronos. Everything is derived from the XML — types, constants, structs, enums (including extension-added values), function pointer typedefs, and dispatch tables. Nothing is hardcoded.

Swap the `vk.xml` file and rebuild to target any Vulkan version from **1.2.175** through the latest release.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
spock = "0.1"
```

If you don't have a local copy of `vk.xml`, enable auto-download:

```toml
[dependencies]
spock = { version = "0.1", features = ["fetch-spec"] }
```

### Basic Usage

```rust
use spock::raw::bindings::*;
use spock::raw::VulkanLibrary;

// All types, constants, and structs are generated from vk.xml
let app_info = VkApplicationInfo {
    sType: VkStructureType::STRUCTURE_TYPE_APPLICATION_INFO,
    pNext: std::ptr::null(),
    pApplicationName: b"My App\0".as_ptr() as *const i8,
    applicationVersion: vk_make_api_version(0, 1, 0, 0),
    pEngineName: std::ptr::null(),
    engineVersion: 0,
    apiVersion: VK_API_VERSION_1_0,
};

// Load the Vulkan library and dispatch tables at runtime
let lib = VulkanLibrary::new().expect("Failed to load Vulkan");
let entry = unsafe { lib.load_entry() };

// Entry, instance, and device dispatch tables are all generated from vk.xml
// with every Vulkan command available as an Option<fn_ptr> field
```

## Providing vk.xml

The build script resolves the Vulkan specification in this order:

1. **`VK_XML_PATH` environment variable** — point to any local `vk.xml` file:
   ```bash
   VK_XML_PATH=/path/to/vk.xml cargo build
   ```

2. **Local copy** — place the Vulkan-Docs repository (or just `vk.xml`) at `spec/registry/Vulkan-Docs/xml/vk.xml` relative to the workspace root.

3. **Auto-download** (requires `fetch-spec` feature) — downloads from the Khronos GitHub repository:
   ```bash
   # Download the latest version
   cargo build --features fetch-spec

   # Download a specific version
   VK_VERSION=1.3.250 cargo build --features fetch-spec
   ```
   Downloaded files are cached locally. Pinned versions are cached permanently; the latest is refreshed after 24 hours.

## Supported Vulkan Versions

Spock supports Vulkan specification versions **1.2.175** through the latest release.

Version 1.2.175 is the minimum because it is the first version that introduced the `VK_MAKE_API_VERSION` / `VK_API_VERSION_*` macro family, which replaced the deprecated `VK_MAKE_VERSION` / `VK_VERSION_*` macros. Spock transpiles these macros from C to Rust `const fn` at build time, so they must be present in the specification.

## What Gets Generated

Everything below is derived entirely from `vk.xml` at build time:

| Category | Count (v1.4, latest) | Description |
|----------|---------------------|-------------|
| Structs | ~1,478 | `#[repr(C)]` with correct pointer/array/const field types |
| Type aliases | ~1,343 | Handles (dispatchable and non-dispatchable), bitmasks, base types |
| Enums | ~148 | Rust enums with extension values merged from all extensions |
| Constants | ~3,064 | Including extension enum values emitted as `pub const` |
| Function pointer typedefs | ~651 | `unsafe extern "system" fn(...)` for every Vulkan command |
| Dispatch tables | 3 | `VkEntryDispatchTable`, `VkInstanceDispatchTable`, `VkDeviceDispatchTable` |
| Version functions | All | `vk_make_api_version`, `vk_api_version_major`, etc. transpiled from C macros |

## Features

| Feature | Description |
|---------|-------------|
| `fetch-spec` | Enable auto-download of `vk.xml` from Khronos GitHub |
| `build-support` | Build support for XML parsing and code generation (enabled by default) |

## Architecture

```
vk.xml
  |
  v
vulkan_gen (tree parser using roxmltree)
  |
  v
11 intermediate JSON files
  |
  v
vulkan-codegen (8 generator modules + assembler)
  |
  v
vulkan_bindings.rs (included via build.rs)
```

The parser uses DOM-based XML parsing (`roxmltree`) to correctly handle nested XML elements like `<member>` and `<param>` that contain mixed text and child elements.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
