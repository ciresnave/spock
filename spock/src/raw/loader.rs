//! Vulkan function loading and management
//!
//! All dispatch tables are generated from vk.xml — no hardcoded function names.

use crate::raw::bindings::*;
use std::ffi::c_void;
use std::sync::Arc;

/// Loads the Vulkan shared library and provides `vkGetInstanceProcAddr`.
pub struct VulkanLibrary {
    _library: Arc<libloading::Library>,
    get_instance_proc_addr: unsafe extern "system" fn(*mut c_void, *const i8) -> *mut c_void,
}

impl VulkanLibrary {
    /// Load the Vulkan runtime library.
    pub fn new() -> Result<Self, libloading::Error> {
        let library = unsafe {
            #[cfg(windows)]
            let lib = libloading::Library::new("vulkan-1.dll")?;
            #[cfg(unix)]
            let lib = libloading::Library::new("libvulkan.so.1")?;
            Arc::new(lib)
        };

        let get_instance_proc_addr = unsafe {
            *library.get::<unsafe extern "system" fn(*mut c_void, *const i8) -> *mut c_void>(
                b"vkGetInstanceProcAddr\0",
            )?
        };

        Ok(Self {
            _library: library,
            get_instance_proc_addr,
        })
    }

    /// Load global (entry-level) functions that don't require a VkInstance.
    pub unsafe fn load_entry(&self) -> VkEntryDispatchTable {
        let gipa = self.get_instance_proc_addr;
        unsafe {
            VkEntryDispatchTable::load(|name| {
                (gipa)(std::ptr::null_mut(), name.as_ptr() as *const i8)
            })
        }
    }

    /// Load instance-level functions for the given VkInstance.
    pub unsafe fn load_instance(&self, instance: VkInstance) -> VkInstanceDispatchTable {
        let gipa = self.get_instance_proc_addr;
        unsafe {
            VkInstanceDispatchTable::load(|name| {
                (gipa)(instance as *mut c_void, name.as_ptr() as *const i8)
            })
        }
    }

    /// Load device-level functions for the given VkDevice.
    ///
    /// `instance` is the VkInstance that owns the device — it is required because
    /// `vkGetDeviceProcAddr` is loaded via `vkGetInstanceProcAddr(instance, ...)`.
    pub unsafe fn load_device(
        &self,
        instance: VkInstance,
        device: VkDevice,
    ) -> VkDeviceDispatchTable {
        let gipa = self.get_instance_proc_addr;

        // First, get vkGetDeviceProcAddr via the instance loader.
        // Per the Vulkan spec, vkGetDeviceProcAddr must be loaded with a
        // valid VkInstance handle (not NULL).
        let gdpa_name = c"vkGetDeviceProcAddr";
        let gdpa_ptr = unsafe { (gipa)(instance as *mut c_void, gdpa_name.as_ptr() as *const i8) };

        if !gdpa_ptr.is_null() {
            // Use vkGetDeviceProcAddr for fastest device-level dispatch.
            let gdpa: unsafe extern "system" fn(*mut c_void, *const i8) -> *mut c_void =
                unsafe { std::mem::transmute(gdpa_ptr) };
            unsafe {
                VkDeviceDispatchTable::load(|name| {
                    (gdpa)(device as *mut c_void, name.as_ptr() as *const i8)
                })
            }
        } else {
            // Fallback: load via instance proc addr (slower, instance-level dispatch).
            unsafe {
                VkDeviceDispatchTable::load(|name| {
                    (gipa)(instance as *mut c_void, name.as_ptr() as *const i8)
                })
            }
        }
    }
}
