use serde::Serialize;
use tempfile::tempdir;
use vulkan_gen::parser::parsing_modules::json_writer::JsonWriter;

#[derive(Serialize)]
struct TestConstant {
    name: String,
    value: String,
}

#[derive(Serialize)]
struct TestEnum {
    name: String,
    values: Vec<String>,
}

#[test]
fn test_constants_json_generation() {
    let temp_dir = tempdir().unwrap();

    let mut writer = JsonWriter::array();
    writer
        .initialize(temp_dir.path(), "constants.json")
        .unwrap();

    let constant = TestConstant {
        name: "VK_API_VERSION_1_0".to_string(),
        value: "4194304".to_string(),
    };

    writer.write_item(&constant).unwrap();
    writer.finalize().unwrap();

    let content = std::fs::read_to_string(temp_dir.path().join("constants.json")).unwrap();
    assert!(content.contains("VK_API_VERSION_1_0"));
    assert!(content.contains("4194304"));

    // Verify it's valid JSON
    let _parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
}

#[test]
fn test_enums_json_generation() {
    let temp_dir = tempdir().unwrap();

    let mut writer = JsonWriter::array();
    writer.initialize(temp_dir.path(), "enums.json").unwrap();

    let enum_item = TestEnum {
        name: "VkResult".to_string(),
        values: vec![
            "VK_SUCCESS".to_string(),
            "VK_ERROR_OUT_OF_HOST_MEMORY".to_string(),
        ],
    };

    writer.write_item(&enum_item).unwrap();
    writer.finalize().unwrap();

    let content = std::fs::read_to_string(temp_dir.path().join("enums.json")).unwrap();
    assert!(content.contains("VkResult"));
    assert!(content.contains("VK_SUCCESS"));
    // Verify it's valid JSON and an array
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.is_array());
}
