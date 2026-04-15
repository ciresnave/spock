//! Tests for the optional `slang` Slang -> SPIR-V compilation feature.
//!
//! Only compiled when the `slang` feature is enabled. Fixture files
//! live under `tests/slang_fixtures/`.

#![cfg(feature = "slang")]

use vulkane::safe::slang::{SlangSession, compile_slang_file};

fn fixtures_dir() -> String {
    format!("{}/tests/slang_fixtures", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn test_compile_trivial_compute() {
    let words = compile_slang_file(&fixtures_dir(), "trivial", "main")
        .expect("trivial compute kernel compiles");

    assert!(!words.is_empty(), "compiled SPIR-V should not be empty");
    assert_eq!(
        words[0], 0x07230203,
        "SPIR-V binary should start with the magic number 0x07230203"
    );
}

#[test]
fn test_session_load_compile() {
    let session = SlangSession::with_search_paths(&[&fixtures_dir()]).expect("session created");
    let module = session.load_file("trivial").expect("module compiles");
    let words = module
        .compile_entry_point("main")
        .expect("entry point compiles");

    assert_eq!(words[0], 0x07230203);
}

#[test]
fn test_missing_entry_point_returns_error() {
    let session = SlangSession::with_search_paths(&[&fixtures_dir()]).expect("session created");
    let module = session.load_file("trivial").expect("module compiles");
    let result = module.compile_entry_point("does_not_exist");

    assert!(
        result.is_err(),
        "requesting a nonexistent entry point should fail"
    );
}

#[test]
fn test_multiple_entry_points_from_one_module() {
    // Two compute entry points in the same source. The session-based
    // API compiles the module once and yields both SPIR-V blobs.
    let session = SlangSession::with_search_paths(&[&fixtures_dir()]).expect("session created");
    let module = session.load_file("autodiff").expect("module compiles");

    let fwd = module
        .compile_entry_point("forward")
        .expect("forward entry point compiles");
    let bwd = module
        .compile_entry_point("backward")
        .expect("backward entry point compiles");

    assert_eq!(fwd[0], 0x07230203);
    assert_eq!(bwd[0], 0x07230203);
    // Entry-point names end up in the SPIR-V OpName decorations, so the
    // two blobs must differ even though the kernel bodies are empty.
    assert_ne!(fwd, bwd, "forward and backward SPIR-V should differ");
}

#[test]
fn test_bad_source_returns_error() {
    let result = compile_slang_file(&fixtures_dir(), "bad", "main");
    assert!(
        result.is_err(),
        "invalid Slang source should produce a compile error"
    );
}
