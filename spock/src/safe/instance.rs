//! Safe wrapper for `VkInstance`.

use super::{Error, PhysicalDevice, Result, check};
use crate::raw::VulkanLibrary;
use crate::raw::bindings::*;
use std::ffi::{CStr, CString};
use std::sync::Arc;

/// A Vulkan API version, encoded as a `u32` per the Vulkan spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApiVersion(pub u32);

impl ApiVersion {
    /// `VK_API_VERSION_1_0`
    pub const V1_0: Self = Self(VK_API_VERSION_1_0);
    /// `VK_API_VERSION_1_1`
    pub const V1_1: Self = Self(VK_API_VERSION_1_1);
    /// `VK_API_VERSION_1_2`
    pub const V1_2: Self = Self(VK_API_VERSION_1_2);
    /// `VK_API_VERSION_1_3`
    pub const V1_3: Self = Self(VK_API_VERSION_1_3);
    /// `VK_API_VERSION_1_4`
    pub const V1_4: Self = Self(VK_API_VERSION_1_4);

    /// Construct a custom version using `vk_make_api_version`.
    pub const fn new(variant: u32, major: u32, minor: u32, patch: u32) -> Self {
        // Reproduce the bit-packing here so this is a const fn.
        // Same formula as the generated `vk_make_api_version`.
        Self((variant << 29) | (major << 22) | (minor << 12) | patch)
    }

    /// Extract the major version component.
    pub const fn major(self) -> u32 {
        (self.0 >> 22) & 0x7F
    }

    /// Extract the minor version component.
    pub const fn minor(self) -> u32 {
        (self.0 >> 12) & 0x3FF
    }

    /// Extract the patch version component.
    pub const fn patch(self) -> u32 {
        self.0 & 0xFFF
    }
}

impl std::fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major(), self.minor(), self.patch())
    }
}

/// Parameters for [`Instance::new`].
#[derive(Debug, Clone, Default)]
pub struct InstanceCreateInfo<'a> {
    /// Application name (will be NUL-terminated for the C API).
    pub application_name: Option<&'a str>,
    /// Application version.
    pub application_version: ApiVersion,
    /// Engine name.
    pub engine_name: Option<&'a str>,
    /// Engine version.
    pub engine_version: ApiVersion,
    /// Vulkan API version the application targets.
    pub api_version: ApiVersion,
}

impl Default for ApiVersion {
    fn default() -> Self {
        Self::V1_0
    }
}

/// Internal state shared between [`Instance`] and its child handles.
///
/// Lives inside an `Arc` so child handles can keep the instance alive even
/// after the user drops their `Instance` clone.
pub(crate) struct InstanceInner {
    pub(crate) library: VulkanLibrary,
    pub(crate) handle: VkInstance,
    pub(crate) dispatch: VkInstanceDispatchTable,
}

// Safety: VkInstance is documented by the Vulkan spec as safe to share
// between threads. Individual function calls have their own external
// synchronization requirements (which are the user's responsibility), but
// the handle itself is thread-safe to access. The dispatch table contains
// only function pointers which are also Send + Sync.
unsafe impl Send for InstanceInner {}
unsafe impl Sync for InstanceInner {}

impl Drop for InstanceInner {
    fn drop(&mut self) {
        if let Some(destroy) = self.dispatch.vkDestroyInstance {
            // Safety: handle is valid (constructed by Instance::new), and
            // by the Arc invariant we are the last owner.
            unsafe { destroy(self.handle, std::ptr::null()) };
        }
    }
}

/// A safe wrapper around `VkInstance`.
///
/// The instance is destroyed automatically when the last `Instance` clone
/// (and the last child handle that holds an `Arc<InstanceInner>`) is dropped.
#[derive(Clone)]
pub struct Instance {
    pub(crate) inner: Arc<InstanceInner>,
}

impl Instance {
    /// Load the Vulkan library and create a new `VkInstance`.
    pub fn new(info: InstanceCreateInfo<'_>) -> Result<Self> {
        let library = VulkanLibrary::new()?;

        // Convert the optional application/engine name strings to CStrings
        // so we can keep them alive across the call.
        let app_name_c = info.application_name.map(CString::new).transpose()?;
        let engine_name_c = info.engine_name.map(CString::new).transpose()?;

        let app_info = VkApplicationInfo {
            sType: VkStructureType::STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: std::ptr::null(),
            pApplicationName: app_name_c.as_deref().map_or(std::ptr::null(), CStr::as_ptr),
            applicationVersion: info.application_version.0,
            pEngineName: engine_name_c
                .as_deref()
                .map_or(std::ptr::null(), CStr::as_ptr),
            engineVersion: info.engine_version.0,
            apiVersion: info.api_version.0,
        };

        let create_info = VkInstanceCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo: &app_info,
            ..Default::default()
        };

        // Load entry-level functions to get vkCreateInstance.
        // Safety: VulkanLibrary just loaded successfully, the entry table
        // points at valid driver functions.
        let entry = unsafe { library.load_entry() };
        let create = entry
            .vkCreateInstance
            .ok_or(Error::MissingFunction("vkCreateInstance"))?;

        let mut handle: VkInstance = std::ptr::null_mut();
        // Safety: create_info is valid for the duration of the call,
        // app_info lives until end of scope, name CStrings live until end of scope.
        let result = unsafe { create(&create_info, std::ptr::null(), &mut handle) };
        check(result)?;

        // Now that we have a real VkInstance, load the instance dispatch table.
        // Safety: handle is the freshly-created valid instance.
        let dispatch = unsafe { library.load_instance(handle) };

        Ok(Self {
            inner: Arc::new(InstanceInner {
                library,
                handle,
                dispatch,
            }),
        })
    }

    /// Returns the raw `VkInstance` handle.
    ///
    /// # Safety
    ///
    /// The caller must not call `vkDestroyInstance` on the returned handle —
    /// the safe wrapper owns its lifetime and will destroy it on drop.
    pub fn raw(&self) -> VkInstance {
        self.inner.handle
    }

    /// Enumerate the physical devices visible to this instance.
    pub fn enumerate_physical_devices(&self) -> Result<Vec<PhysicalDevice>> {
        let enumerate = self
            .inner
            .dispatch
            .vkEnumeratePhysicalDevices
            .ok_or(Error::MissingFunction("vkEnumeratePhysicalDevices"))?;

        let mut count: u32 = 0;
        // Safety: count query — the device handles array pointer is null.
        check(unsafe { enumerate(self.inner.handle, &mut count, std::ptr::null_mut()) })?;
        let mut handles: Vec<VkPhysicalDevice> = vec![std::ptr::null_mut(); count as usize];
        // Safety: handles has space for `count` elements.
        check(unsafe { enumerate(self.inner.handle, &mut count, handles.as_mut_ptr()) })?;

        Ok(handles
            .into_iter()
            .map(|h| PhysicalDevice::new(Arc::clone(&self.inner), h))
            .collect())
    }
}
