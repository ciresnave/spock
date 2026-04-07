use tempfile::tempdir;

#[test]
fn test_json_writer_basic() {
    use serde::Serialize;
    use vulkan_gen::parser::parsing_modules::json_writer::JsonWriter;

    #[derive(Serialize)]
    struct TestItem {
        name: String,
        value: i32,
    }

    let temp_dir = tempdir().unwrap();

    // Test array format
    let mut writer = JsonWriter::array();
    writer.initialize(temp_dir.path(), "test.json").unwrap();

    let item1 = TestItem {
        name: "test1".to_string(),
        value: 1,
    };
    let item2 = TestItem {
        name: "test2".to_string(),
        value: 2,
    };

    writer.write_item(&item1).unwrap();
    writer.write_item(&item2).unwrap();
    writer.finalize().unwrap();

    let content = std::fs::read_to_string(temp_dir.path().join("test.json")).unwrap();
    println!("Generated JSON: {}", content);

    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.is_array());
    assert_eq!(parsed.as_array().unwrap().len(), 2);
}

#[test]
fn test_json_writer_object_with_array() {
    use serde::Serialize;
    use vulkan_gen::parser::parsing_modules::json_writer::JsonWriter;

    #[derive(Serialize)]
    struct TestItem {
        name: String,
        value: i32,
    }

    let temp_dir = tempdir().unwrap();

    // Test array format (replacing former object-with-array behavior)
    let mut writer = JsonWriter::array();
    writer
        .initialize(temp_dir.path(), "test_object.json")
        .unwrap();

    let item1 = TestItem {
        name: "test1".to_string(),
        value: 1,
    };
    let item2 = TestItem {
        name: "test2".to_string(),
        value: 2,
    };

    writer.write_item(&item1).unwrap();
    writer.write_item(&item2).unwrap();
    writer.finalize().unwrap();

    let content = std::fs::read_to_string(temp_dir.path().join("test_object.json")).unwrap();
    println!("Generated JSON: {}", content);

    // Verify it's valid JSON and an array
    let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(parsed.is_array());
    assert_eq!(parsed.as_array().unwrap().len(), 2);
}
