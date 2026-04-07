//! Safe wrappers for Vulkan synchronization primitives.
//!
//! Currently only [`Fence`] is implemented. Semaphores and events are TODO.

use super::device::DeviceInner;
use super::{Device, Error, Result, check};
use crate::raw::bindings::*;
use std::sync::Arc;

/// A safe wrapper around `VkFence`.
///
/// Fences are CPU-GPU synchronization primitives: a queue submission can
/// signal a fence on completion, and the host can wait on the fence.
///
/// The fence is destroyed automatically on drop.
pub struct Fence {
    pub(crate) handle: VkFence,
    pub(crate) device: Arc<DeviceInner>,
}

impl Fence {
    /// Create a new fence in the unsignaled state.
    pub fn new(device: &Device) -> Result<Self> {
        let create = device
            .inner
            .dispatch
            .vkCreateFence
            .ok_or(Error::MissingFunction("vkCreateFence"))?;

        let info = VkFenceCreateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_FENCE_CREATE_INFO,
            ..Default::default()
        };

        let mut handle: VkFence = 0;
        // Safety: info is valid for the call, device is valid.
        check(unsafe { create(device.inner.handle, &info, std::ptr::null(), &mut handle) })?;

        Ok(Self {
            handle,
            device: Arc::clone(&device.inner),
        })
    }

    /// Returns the raw `VkFence` handle.
    pub fn raw(&self) -> VkFence {
        self.handle
    }

    /// Block the calling thread until the fence is signaled, or until the
    /// timeout (in nanoseconds) elapses. Pass `u64::MAX` to wait forever.
    pub fn wait(&self, timeout_nanos: u64) -> Result<()> {
        let wait = self
            .device
            .dispatch
            .vkWaitForFences
            .ok_or(Error::MissingFunction("vkWaitForFences"))?;

        // Safety: handle is valid; we wait on a single fence (count = 1).
        check(unsafe {
            wait(
                self.device.handle,
                1,
                &self.handle,
                1, // wait_all (one fence, doesn't matter)
                timeout_nanos,
            )
        })
    }

    /// Reset the fence back to the unsignaled state.
    pub fn reset(&self) -> Result<()> {
        let reset = self
            .device
            .dispatch
            .vkResetFences
            .ok_or(Error::MissingFunction("vkResetFences"))?;
        // Safety: handle is valid.
        check(unsafe { reset(self.device.handle, 1, &self.handle) })
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        if let Some(destroy) = self.device.dispatch.vkDestroyFence {
            // Safety: handle is valid; we are the sole owner.
            unsafe { destroy(self.device.handle, self.handle, std::ptr::null()) };
        }
    }
}
