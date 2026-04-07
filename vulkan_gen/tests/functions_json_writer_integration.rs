use std::fs;
use tempfile::tempdir;
use vulkan_gen::parser::parsing_modules::functions::FunctionsHandler;

#[test]
fn test_functions_handler_json_writer_integration() {
    // Create a temporary directory for output
    let temp_dir = tempdir().unwrap();
    let output_dir = temp_dir.path();

    // Create a minimal XML string representing a Vulkan command
    let xml_content = r#"
    <commands>
        <command>
            <proto><type>void</type> <name>vkTestFunction</name></proto>
            <param><type>int</type> <name>param1</name></param>
        </command>
    </commands>
    "#;

    // Initialize the handler and output
    let mut handler = FunctionsHandler::new();
    handler.initialize_output(output_dir).unwrap();

    // Use the hybrid approach to parse the XML
    handler.parse_with_hybrid_approach(xml_content).unwrap();
    handler.finalize_output().unwrap();

    // Read the output file
    let output_path = output_dir.join("functions.json");
    let content = fs::read_to_string(&output_path).expect("Should read functions.json");
    println!("Generated JSON: {}", content);

    // Verify the output contains the function name and parameter
    assert!(content.contains("vkTestFunction"));
    // Verify the parameter parsing worked for this simple case
    assert!(content.contains("param1"));
    // For now, we just verify the function was found and the return type handling
    assert!(content.contains("return_type"));
}
