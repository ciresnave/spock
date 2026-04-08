# vulkan_gen

Vulkan XML specification parser and Rust binding generator. Used internally
by the [`spock`](https://crates.io/crates/spock) crate, but reusable as a
standalone code generator.

`vulkan_gen` parses [`vk.xml`](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/xml/vk.xml)
into typed Rust structures and emits a complete `vulkan_bindings.rs` file
with every type, constant, struct, enum, function pointer, and dispatch
table the spec defines. The parser uses [`roxmltree`](https://crates.io/crates/roxmltree)
DOM parsing to correctly handle nested elements like `<member>` and
`<param>` that contain mixed text and child element content.

For end users targeting Vulkan from Rust, **install the
[`spock`](https://crates.io/crates/spock) crate instead** — it includes
`vulkan_gen` as a build dependency and exposes both the raw bindings and
a complete safe RAII wrapper for compute and graphics.

## Standalone usage

If you want to generate your own bindings:

```rust
use vulkan_gen::generate_bindings;

let xml_path = std::path::Path::new("vk.xml");
let out_path = std::path::Path::new("vulkan_bindings.rs");
generate_bindings(xml_path, out_path)?;
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
