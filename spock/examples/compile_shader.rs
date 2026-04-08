//! One-shot helper that compiles every `*.comp` GLSL file under
//! `examples/shaders/` to its matching `*.spv` SPIR-V binary using the
//! optional `naga` feature.
//!
//! Run with: `cargo run -p spock --features naga,fetch-spec --example compile_shader`
//!
//! The generated `.spv` files are checked into the repository, so users
//! running the compute examples don't need `naga`, `glslc`, or any shader
//! toolchain. This helper is only needed if you edit the GLSL sources.

#[cfg(not(feature = "naga"))]
fn main() {
    eprintln!(
        "This example requires the `naga` feature.\n\
         Run with: cargo run -p spock --features naga,fetch-spec --example compile_shader"
    );
    std::process::exit(1);
}

#[cfg(feature = "naga")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use spock::safe::naga::compile_glsl;

    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let shaders_dir = format!("{manifest_dir}/examples/shaders");

    // Compile every .comp file in the shaders directory.
    let mut found = 0usize;
    for entry in std::fs::read_dir(&shaders_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("comp") {
            continue;
        }
        found += 1;
        let glsl_path = path.display().to_string();
        let spv_path = path.with_extension("spv").display().to_string();

        println!("Reading {glsl_path}");
        let source = std::fs::read_to_string(&path)?;

        println!("Compiling GLSL -> SPIR-V via naga");
        let words = compile_glsl(&source, ::naga::ShaderStage::Compute)?;

        // Write as little-endian bytes (the universal SPIR-V on-disk format).
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in &words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }

        println!(
            "Writing {} bytes ({} u32 words) to {spv_path}",
            bytes.len(),
            words.len()
        );
        std::fs::write(&spv_path, &bytes)?;
    }

    if found == 0 {
        eprintln!("No .comp files found under {shaders_dir}");
    } else {
        println!("Done. Compiled {found} shader(s).");
    }
    Ok(())
}
