use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let vk_xml_path = "spec/registry/Vulkan-Docs/xml/vk.xml";
    let out_dir = "intermediate_tree_test";

    println!("Parsing vk.xml with tree parser...");
    vulkan_gen::vulkan_spec_parser::parse_vulkan_spec(vk_xml_path, out_dir)?;

    // Verify output
    let structs: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(format!("{}/structs.json", out_dir))?)?;
    let functions: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(format!("{}/functions.json", out_dir))?)?;
    let types: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(format!("{}/types.json", out_dir))?)?;
    let enums: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(format!("{}/enums.json", out_dir))?)?;
    let constants: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(format!("{}/constants.json", out_dir))?)?;
    let extensions: Vec<serde_json::Value> =
        serde_json::from_str(&fs::read_to_string(format!("{}/extensions.json", out_dir))?)?;

    println!("\n=== COUNTS ===");
    println!("Structs: {}", structs.len());
    println!("Functions: {}", functions.len());
    println!("Types: {}", types.len());
    println!("Enums: {}", enums.len());
    println!("Constants: {}", constants.len());
    println!("Extensions: {}", extensions.len());

    // Check struct member quality
    let with_members = structs
        .iter()
        .filter(|s| {
            s.get("members")
                .and_then(|m| m.as_array())
                .map_or(false, |a| !a.is_empty())
        })
        .count();
    let alias_structs = structs
        .iter()
        .filter(|s| s.get("is_alias").and_then(|v| v.as_bool()).unwrap_or(false))
        .count();
    println!(
        "\nStructs with members: {} (aliases: {})",
        with_members, alias_structs
    );

    // Check function parameter quality
    let with_return_type = functions
        .iter()
        .filter(|f| {
            f.get("return_type")
                .and_then(|r| r.as_str())
                .map_or(false, |s| !s.is_empty())
        })
        .count();
    let with_params = functions
        .iter()
        .filter(|f| {
            f.get("parameters")
                .and_then(|p| p.as_array())
                .map_or(false, |a| {
                    !a.is_empty()
                        && a[0]
                            .get("name")
                            .and_then(|n| n.as_str())
                            .map_or(false, |s| !s.is_empty())
                })
        })
        .count();
    let alias_funcs = functions
        .iter()
        .filter(|f| f.get("is_alias").and_then(|v| v.as_bool()).unwrap_or(false))
        .count();
    println!(
        "Functions with return_type: {} (aliases: {})",
        with_return_type, alias_funcs
    );
    println!("Functions with named params: {}", with_params);

    // Check type categories
    let mut categories = std::collections::HashMap::new();
    for t in &types {
        let cat = t.get("category").and_then(|c| c.as_str()).unwrap_or("?");
        *categories.entry(cat.to_string()).or_insert(0u32) += 1;
    }
    println!("\nType categories: {:?}", categories);

    // Spot check: VkApplicationInfo
    if let Some(app_info) = structs
        .iter()
        .find(|s| s.get("name").and_then(|n| n.as_str()) == Some("VkApplicationInfo"))
    {
        let members = app_info.get("members").and_then(|m| m.as_array()).unwrap();
        println!("\nVkApplicationInfo members ({}):", members.len());
        for m in members {
            println!(
                "  {} : {} | def: {}",
                m.get("name").and_then(|n| n.as_str()).unwrap_or("?"),
                m.get("type_name").and_then(|n| n.as_str()).unwrap_or("?"),
                m.get("definition").and_then(|n| n.as_str()).unwrap_or("?")
            );
        }
    }

    // Spot check: vkCreateInstance
    if let Some(create_inst) = functions
        .iter()
        .find(|f| f.get("name").and_then(|n| n.as_str()) == Some("vkCreateInstance"))
    {
        println!("\nvkCreateInstance:");
        println!(
            "  return_type: {}",
            create_inst
                .get("return_type")
                .and_then(|r| r.as_str())
                .unwrap_or("?")
        );
        let params = create_inst
            .get("parameters")
            .and_then(|p| p.as_array())
            .unwrap();
        for p in params {
            println!(
                "  param: {} : {} | def: {}",
                p.get("name").and_then(|n| n.as_str()).unwrap_or("?"),
                p.get("type_name").and_then(|n| n.as_str()).unwrap_or("?"),
                p.get("definition").and_then(|n| n.as_str()).unwrap_or("?")
            );
        }
    }

    Ok(())
}
