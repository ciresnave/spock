//! Struct generator module
//!
//! Generates Rust struct definitions from structs.json intermediate file

use std::fs;
use std::path::Path;

use super::{GeneratorError, GeneratorModule, GeneratorResult};

use crate::parser::vk_types::{EnumDefinition, StructDefinition, TypeDefinition};

/// Map from enum name -> the variant identifier to use for `Default::default()`.
/// For enums where 0 is not a valid value, this lets struct Default impls
/// avoid undefined behavior from `mem::zeroed()`.
type EnumDefaultMap = std::collections::HashMap<String, String>;

/// Sanitize a type name to be a valid Rust identifier
fn sanitize_type_name(name: &str) -> String {
    let mut s = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_alphanumeric() || c == '_' {
            s.push(c);
        } else {
            s.push('_');
        }
    }
    // Prevent leading digits
    if s.chars().next().map_or(false, |c| c.is_ascii_digit()) {
        s = format!("_{}", s);
    }
    s
}

/// Generator module for Vulkan structs
pub struct StructGenerator;

impl StructGenerator {
    pub fn new() -> Self {
        Self
    }

    /// Generate Rust code for a single struct
    fn generate_struct(
        &self,
        struct_def: &StructDefinition,
        _all_type_names: &std::collections::HashSet<String>,
        enum_defaults: &EnumDefaultMap,
        _output_dir: &Path,
    ) -> String {
        let mut code = String::new();

        let _is_union = struct_def.category == "union";

        // Determine which traits to derive.
        // We omit Debug because some structs contain union fields which
        // don't implement Debug, and we can't know at generation time
        // which fields are unions vs structs (they're just type names).
        let can_derive_copy = self.can_derive_copy(struct_def);
        let derives = if can_derive_copy {
            "#[derive(Clone, Copy)]"
        } else {
            "#[derive(Clone)]"
        };

        let is_union = struct_def.category == "union";
        let keyword = if is_union { "union" } else { "struct" };

        // Emit doc comment from vk.xml if present
        if let Some(comment) = &struct_def.comment {
            for line in comment.lines() {
                code.push_str(&format!(
                    "/// {}\n",
                    crate::codegen::sanitize_doc_line(line)
                ));
            }
        }

        code.push_str("#[repr(C)]\n");
        code.push_str(&format!("{}\n", derives));
        let sanitized_struct_name = sanitize_type_name(&struct_def.name);
        code.push_str(&format!("pub {} {} {{\n", keyword, sanitized_struct_name));

        // Fields with deduplication
        let mut seen_fields = std::collections::HashSet::new();

        for field in &struct_def.members {
            let field_name = self.escape_rust_keyword(&field.name);

            // Skip duplicate fields
            if seen_fields.contains(&field_name) {
                continue;
            }
            seen_fields.insert(field_name.clone());

            // Emit doc comment from vk.xml if present
            if let Some(comment) = &field.comment {
                for line in comment.lines() {
                    code.push_str(&format!(
                        "    /// {}\n",
                        crate::codegen::sanitize_doc_line(line)
                    ));
                }
            }

            let rust_type = self.map_member_type(field);
            code.push_str(&format!("    pub {}: {},\n", field_name, rust_type));
        }

        code.push_str("}\n\n");

        // Default implementation
        code.push_str(&format!("impl Default for {} {{\n", sanitized_struct_name));
        code.push_str("    fn default() -> Self {\n");

        if is_union {
            // Unions must use zeroed() since only one field can be initialized
            code.push_str("        unsafe { std::mem::zeroed() }\n");
        } else {
            code.push_str("        Self {\n");

            let mut seen_fields = std::collections::HashSet::new();
            for field in &struct_def.members {
                let field_name = self.escape_rust_keyword(&field.name);

                if seen_fields.contains(&field_name) {
                    continue;
                }
                seen_fields.insert(field_name.clone());

                let rust_type = self.map_member_type(field);
                let is_pointer = rust_type.starts_with("*const") || rust_type.starts_with("*mut");
                let is_array = rust_type.starts_with('[');
                // Check if this field's base type is an enum we know how to default.
                // Only applies to scalar fields — array fields keep using zeroed().
                let default_value =
                    if !is_pointer && !is_array && enum_defaults.contains_key(&field.type_name) {
                        let variant = &enum_defaults[&field.type_name];
                        format!("{}::{}", field.type_name, variant)
                    } else {
                        self.get_default_value_for_rust_type(&rust_type, is_pointer)
                    };
                code.push_str(&format!("            {}: {},\n", field_name, default_value));
            }

            code.push_str("        }\n");
        }

        code.push_str("    }\n");
        code.push_str("}\n\n");

        code
    }

    /// Determine if a struct can derive Copy trait
    fn can_derive_copy(&self, struct_def: &StructDefinition) -> bool {
        // Check all fields to see if they can be copied
        for field in &struct_def.members {
            let rust_type = self.simple_map_type(&field.type_name);

            // For simplified version, assume most Vulkan types can be copied
            if !self.type_supports_copy_simple(&rust_type) {
                return false;
            }
        }
        true
    }

    /// Check if a simple type supports Copy trait
    fn type_supports_copy_simple(&self, type_name: &str) -> bool {
        match type_name {
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "f32" | "f64"
            | "bool" | "usize" | "isize" | "c_char" | "c_uchar" | "c_short" | "c_ushort"
            | "c_int" | "c_uint" | "c_long" | "c_ulong" | "c_longlong" | "c_ulonglong"
            | "c_float" | "c_double" | "c_void" => true,

            // Pointers support Copy
            _ if type_name.starts_with("*const") || type_name.starts_with("*mut") => true,

            // Most Vulkan types should support Copy (they're typically enums or simple handles)
            _ if type_name.starts_with("Vk") => true,

            // Be conservative for unknown types
            _ => false,
        }
    }

    /// Map Vulkan types to Rust types with proper array handling
    fn map_type_to_rust(
        &self,
        vulkan_type: &str,
        const_qualified: bool,
        pointer_level: usize,
        is_array: bool,
        array_size: &Option<String>,
    ) -> String {
        // Handle arrays first
        if is_array && pointer_level == 0 {
            if let Some(size) = array_size {
                let base_type = self.map_base_vulkan_to_rust(vulkan_type);
                return format!("[{}; {}]", base_type, size);
            }
        }

        // Handle pointers
        let base_type = self.map_base_vulkan_to_rust(vulkan_type);

        if pointer_level == 0 {
            base_type
        } else {
            let mut result = base_type;
            for level in 0..pointer_level {
                // Apply const qualification to the outer-most pointer when requested.
                // Build from inner to outer; outermost iteration is when level == pointer_level - 1.
                if level == pointer_level - 1 {
                    if const_qualified {
                        result = format!("*const {}", result);
                    } else {
                        result = format!("*mut {}", result);
                    }
                } else {
                    result = format!("*mut {}", result);
                }
            }
            result
        }
    }

    /// Map base Vulkan types to Rust types
    fn map_base_vulkan_to_rust(&self, vulkan_type: &str) -> String {
        match vulkan_type {
            // Use fully-qualified names to avoid relying on a specific import
            // order in the final assembled file.
            "void" => "c_void".to_string(),
            "char" => "c_char".to_string(),
            "uint8_t" => "u8".to_string(),
            "uint16_t" => "u16".to_string(),
            "uint32_t" => "u32".to_string(),
            "uint64_t" => "u64".to_string(),
            "int8_t" => "i8".to_string(),
            "int16_t" => "i16".to_string(),
            "int32_t" => "i32".to_string(),
            "int64_t" => "i64".to_string(),
            "float" => "f32".to_string(),
            "double" => "f64".to_string(),
            "size_t" => "usize".to_string(),
            "int" => "i32".to_string(),
            "unsigned" => "u32".to_string(),
            _ => vulkan_type.to_string(), // Keep Vulkan types as-is
        }
    }

    /// Simple type mapping for simplified intermediate types
    fn simple_map_type(&self, type_name: &str) -> String {
        self.map_base_vulkan_to_rust(type_name)
    }

    /// Parse a struct member definition to produce the full Rust type.
    /// Handles pointers, const, and arrays from the C definition string.
    fn map_member_type(&self, member: &crate::parser::vk_types::StructMember) -> String {
        let def = member.definition.trim();
        let _base = self.map_base_vulkan_to_rust(&member.type_name);

        // Count pointer levels
        let pointer_level = def.chars().filter(|c| *c == '*').count();

        // Check const qualification
        let const_qualified = def.starts_with("const") || def.contains("const ");

        // Check for array: look for [SOMETHING] in the definition
        let array_size = if let Some(bracket_start) = def.find('[') {
            if let Some(bracket_end) = def[bracket_start..].find(']') {
                let size_str = def[bracket_start + 1..bracket_start + bracket_end].trim();
                // If the size is a named constant (not a number), append `as usize`
                if !size_str.is_empty() && !size_str.chars().next().unwrap().is_ascii_digit() {
                    Some(format!("{} as usize", size_str))
                } else {
                    Some(size_str.to_string())
                }
            } else {
                None
            }
        } else {
            None
        };

        self.map_type_to_rust(
            &member.type_name,
            const_qualified,
            pointer_level,
            array_size.is_some(),
            &array_size,
        )
    }

    /// Get default value for a fully mapped Rust type (including arrays)
    fn get_default_value_for_rust_type(&self, rust_type: &str, is_pointer: bool) -> String {
        if is_pointer {
            // Use null_mut for mutable pointers, null for const pointers
            if rust_type.starts_with("*const") {
                return "std::ptr::null()".to_string();
            } else {
                return "std::ptr::null_mut()".to_string();
            }
        }

        // Handle array types like [VkMemoryType; 32]
        if rust_type.starts_with('[') && rust_type.contains(';') && rust_type.ends_with(']') {
            // Extract the array type and size
            if let Some(semicolon_pos) = rust_type.find(';') {
                let _inner_type = rust_type[1..semicolon_pos].trim();
                let _size_part = rust_type[semicolon_pos + 1..rust_type.len() - 1].trim();

                // Use unsafe zeroed for all array defaults since array elements
                // may be complex types (structs, enums) that don't implement Default
                return format!("unsafe {{ std::mem::zeroed() }}");
            }
        }

        // Handle standard types
        match rust_type {
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" => "0".to_string(),
            "f32" | "f64" => "0.0".to_string(),
            "bool" => "false".to_string(),
            "c_char" | "c_uchar" | "c_short" | "c_ushort" | "c_int" | "c_uint" | "c_long"
            | "c_ulong" | "c_longlong" | "c_ulonglong" => "0".to_string(),
            "c_float" => "0.0".to_string(),
            "c_double" => "0.0".to_string(),
            _ => "unsafe { std::mem::zeroed() }".to_string(),
        }
    }

    /// Escape Rust keywords by adding r# prefix
    fn escape_rust_keyword(&self, name: &str) -> String {
        match name {
            "type" | "match" | "impl" | "fn" | "let" | "mut" | "const" | "static" | "if"
            | "else" | "while" | "for" | "loop" | "break" | "continue" | "return" | "struct"
            | "enum" | "trait" | "mod" | "pub" | "use" | "extern" | "crate" | "self" | "Self"
            | "super" | "where" | "async" | "await" | "dyn" | "abstract" | "become" | "box"
            | "do" | "final" | "macro" | "override" | "priv" | "typeof" | "unsized" | "virtual"
            | "yield" | "try" | "union" | "ref" => format!("r#{}", name),
            _ => name.to_string(),
        }
    }

    /// Build a map from enum name to the variant identifier to use for `Default::default()`.
    /// For Rust-enum-style enums (not bitmask), we record the first non-alias variant
    /// only when 0 is not a valid value — meaning `mem::zeroed()` would produce UB.
    fn build_enum_defaults(&self, input_dir: &Path) -> EnumDefaultMap {
        let mut map = EnumDefaultMap::new();
        let enums_path = input_dir.join("enums.json");
        let content = match fs::read_to_string(&enums_path) {
            Ok(c) => c,
            Err(_) => return map,
        };
        let enums: Vec<EnumDefinition> = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(_) => return map,
        };

        for e in &enums {
            // Only handle Rust-enum-style enums (not bitmask, which are emitted as `pub const`).
            // Bitmask enums get u32/u64 default to 0 which is always a valid empty flag set.
            if e.enum_type == "bitmask" {
                continue;
            }

            // Skip if any variant has value 0 — `mem::zeroed()` is safe in that case.
            let mut has_zero = false;
            let mut first_non_alias: Option<&str> = None;
            for v in &e.values {
                if v.is_alias {
                    continue;
                }
                if first_non_alias.is_none() {
                    first_non_alias = Some(&v.name);
                }
                if let Some(val_str) = &v.value {
                    if val_str.trim() == "0" {
                        has_zero = true;
                        break;
                    }
                }
            }
            if has_zero {
                continue;
            }
            if let Some(first_name) = first_non_alias {
                // Format the variant name the same way the enum generator does:
                // strip leading "VK_" prefix.
                let variant = if first_name.starts_with("VK_") {
                    first_name[3..].to_string()
                } else {
                    first_name.to_string()
                };
                map.insert(e.name.clone(), variant);
            }
        }
        map
    }

    /// Generate code for all structs in the input directory
    fn generate_all_structs(
        &self,
        input_dir: &Path,
        output_dir: &Path,
        all_type_names: &std::collections::HashSet<String>,
    ) -> GeneratorResult<()> {
        // Read input file
        let input_path = input_dir.join("structs.json");
        let input_content = fs::read_to_string(&input_path).map_err(|e| GeneratorError::Io(e))?;

        // Parse JSON - try direct array format first, then fallback to object-with-array { "structs": [...] }
        let structs: Vec<StructDefinition> =
            match serde_json::from_str::<Vec<StructDefinition>>(&input_content) {
                Ok(v) => v,
                Err(_) => {
                    #[derive(serde::Deserialize)]
                    struct StructsFile {
                        structs: Vec<StructDefinition>,
                    }

                    let wrapper: StructsFile = serde_json::from_str(&input_content)
                        .map_err(|e| GeneratorError::Json(e))?;
                    wrapper.structs
                }
            };

        // Build the enum-defaults map by reading enums.json. For enums where 0
        // is not a valid variant, we record the first variant name so struct
        // Default impls can use it instead of `mem::zeroed()` (which is UB).
        let enum_defaults = self.build_enum_defaults(input_dir);

        // Generate code
        let mut generated_code = String::new();

        // Don't add imports here - they're handled by the assembler

        // Add allow directives (outer attributes)
        generated_code.push_str("#[allow(non_camel_case_types)]\n");
        generated_code.push_str("#[allow(dead_code)]\n");

        // Generate structs
        for struct_def in &structs {
            generated_code.push_str(&self.generate_struct(
                struct_def,
                all_type_names,
                &enum_defaults,
                output_dir,
            ));
        }

        // Ensure output directory exists
        fs::create_dir_all(output_dir).map_err(|e| GeneratorError::Io(e))?;

        // Write output file
        let output_path = output_dir.join("structs.rs");
        fs::write(output_path, generated_code).map_err(|e| GeneratorError::Io(e))?;

        crate::codegen::logging::log_info(&format!(
            "StructGeneratorModule: Generated {} structs",
            structs.len()
        ));
        Ok(())
    }
}

impl GeneratorModule for StructGenerator {
    fn name(&self) -> &str {
        "StructGenerator"
    }

    fn input_files(&self) -> Vec<String> {
        vec!["structs.json".to_string()]
    }

    fn output_file(&self) -> String {
        "structs.rs".to_string()
    }

    fn dependencies(&self) -> Vec<String> {
        vec![
            "TypeGenerator".to_string(),
            "EnumGenerator".to_string(),
            "ConstantGenerator".to_string(),
        ]
    }

    fn generate(&self, input_dir: &Path, output_dir: &Path) -> GeneratorResult<()> {
        // Collect all type names from types.json and structs.json for reference
        let mut all_type_names = std::collections::HashSet::new();
        // Read types.json
        let types_path = input_dir.join("types.json");
        if let Ok(types_content) = fs::read_to_string(types_path) {
            if let Ok(types) = serde_json::from_str::<Vec<TypeDefinition>>(&types_content) {
                for t in &types {
                    all_type_names.insert(t.name.clone());
                }
            }
        }
        // Read structs.json
        let structs_path = input_dir.join("structs.json");
        if let Ok(structs_content) = fs::read_to_string(structs_path) {
            if let Ok(structs) = serde_json::from_str::<Vec<StructDefinition>>(&structs_content) {
                for s in &structs {
                    all_type_names.insert(s.name.clone());
                }
            }
        }
        self.generate_all_structs(input_dir, output_dir, &all_type_names)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_mapping() {
        let generator = StructGenerator::new();

        assert_eq!(
            generator.map_type_to_rust("uint32_t", false, 0, false, &None),
            "u32"
        );
        assert_eq!(
            generator.map_type_to_rust("uint32_t", true, 1, false, &None),
            "*const u32"
        );
        assert_eq!(
            generator.map_type_to_rust("uint32_t", false, 1, false, &None),
            "*mut u32"
        );
        assert_eq!(
            generator.map_type_to_rust("void", true, 1, false, &None),
            "*const c_void"
        );
        assert_eq!(
            generator.map_type_to_rust("VkDevice", false, 0, false, &None),
            "VkDevice"
        );

        // Test multiple pointer levels
        assert_eq!(
            generator.map_type_to_rust("char", true, 2, false, &None),
            "*const *mut c_char"
        );
        assert_eq!(
            generator.map_type_to_rust("void", false, 3, false, &None),
            "*mut *mut *mut c_void"
        );

        // Test arrays
        assert_eq!(
            generator.map_type_to_rust("char", false, 0, true, &Some("256".to_string())),
            "[c_char; 256]"
        );
    }

    #[test]
    fn test_default_values() {
        let generator = StructGenerator::new();

        // Integer primitives default to 0
        assert_eq!(generator.get_default_value_for_rust_type("u32", false), "0");
        // Float primitives default to 0.0 (NOT 0 — that would be a type error)
        assert_eq!(
            generator.get_default_value_for_rust_type("f32", false),
            "0.0"
        );
        // Const pointers default to null
        assert_eq!(
            generator.get_default_value_for_rust_type("*const u32", true),
            "std::ptr::null()"
        );
        // Mut pointers default to null_mut
        assert_eq!(
            generator.get_default_value_for_rust_type("*mut u32", true),
            "std::ptr::null_mut()"
        );
        // Unknown types (Vulkan handles, structs) fall back to zeroed for FFI safety
        assert_eq!(
            generator.get_default_value_for_rust_type("VkDevice", false),
            "unsafe { std::mem::zeroed() }"
        );
        // Arrays use zeroed because element types may be complex
        assert_eq!(
            generator.get_default_value_for_rust_type("[c_char; 256]", false),
            "unsafe { std::mem::zeroed() }"
        );
    }
}
