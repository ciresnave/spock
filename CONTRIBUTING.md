# Contributing to infra-vulkan

Thank you for your interest in contributing to infra-vulkan! This document provides guidelines and instructions for contributing to the project.

## Development Setup

1. Install development dependencies:
   - Vulkan SDK 1.4.316 or later
   - Rust 1.75.0 or later
   - CMake 3.20 or later (for building examples)

2. Clone and build:

   ```bash
   git clone https://github.com/username/infra-vulkan.git
   cd infra-vulkan
   cargo build --all-features
   ```

3. Run tests:

   ```bash
   cargo test
   cargo test --all-features
   cargo test --features validation-layers
   ```

## Code Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `rustfmt` with the project's configuration
- Run `clippy` with all lints enabled
- Maintain comprehensive documentation

## Documentation Requirements

1. Every public API must have:
   - Clear purpose and usage examples
   - Parameter descriptions
   - Safety section for unsafe functions
   - Error conditions and handling

2. Include doc tests demonstrating usage:

```rust
/// Allocate device memory with specific properties
///
/// # Examples
///
/// ```
/// use infra_vulkan::{Device, memory::MemoryPropertyFlags};
/// 
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// # let device: Device = unimplemented!();
/// let memory = device.allocate_memory(
///     1024,
///     MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
/// )?;
/// # Ok(())
/// # }
/// ```
```

## Testing Requirements

1. Unit Tests:
   - Test each public API
   - Cover error cases
   - Mock external dependencies

2. Integration Tests:
   - Test realistic usage patterns
   - Verify component interactions
   - Test platform-specific features

3. Performance Tests:
   - Benchmark critical operations
   - Compare against baselines
   - Document performance characteristics

## Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Make your changes:
   - Follow code style guidelines
   - Add tests
   - Update documentation
4. Submit a pull request:
   - Describe the changes
   - Link related issues
   - Add test results

## Development Workflow

1. Check existing issues and PRs
2. Create an issue for new features
3. Write failing tests first
4. Implement the feature
5. Document thoroughly
6. Submit PR for review

## Safety Guidelines

1. Mark unsafe functions appropriately:

   ```rust
   /// Create a buffer from a raw handle
   ///
   /// # Safety
   ///
   /// The caller must ensure:
   /// - The handle is valid
   /// - The handle was created by the same device
   /// - The handle is not used elsewhere
   pub unsafe fn from_raw(
       device: &Arc<Device>,
       handle: VkBuffer,
   ) -> Buffer {
       // Implementation
   }
   ```

2. Document all safety requirements thoroughly
3. Validate inputs wherever possible
4. Use safe abstractions by default

## Performance Guidelines

1. Profile before optimizing
2. Document performance characteristics
3. Add benchmarks for changes
4. Compare against baselines

## Release Process

1. Update version numbers
2. Update changelog
3. Run full test suite
4. Build documentation
5. Create release tag
6. Publish to crates.io

## Getting Help

- Join our Discord server
- Check project discussions
- Review existing issues
- Ask on the Rust forum

## License

By contributing, you agree to license your code under either:

- Apache License, Version 2.0
- MIT License

at your option.
