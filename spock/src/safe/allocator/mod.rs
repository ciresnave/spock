//! VMA-style sub-allocator for Vulkan device memory.
//!
//! Real Vulkan applications allocate thousands of buffers and images. The
//! "one `vkAllocateMemory` per resource" pattern hits the
//! `maxMemoryAllocationCount` limit (~4096 on most drivers) very quickly,
//! and is wasteful besides — every individual `VkDeviceMemory` carries
//! per-allocation overhead in the driver.
//!
//! [`Allocator`] solves this by sub-allocating from larger
//! [`Block`](Block)s (one `VkDeviceMemory` each) using a
//! [Two-Level Segregated Fit](tlsf::Tlsf) free-list strategy. The API is
//! deliberately modeled after AMD's
//! [VulkanMemoryAllocator](https://gpuopen.com/vulkan-memory-allocator/) so
//! that users coming from C++ Vulkan find the concepts familiar:
//!
//! - **General-purpose pool** (TLSF) — long-lived mixed-size allocations.
//! - **Linear pool** (bump allocator) — stack/ring/single-frame uploads.
//! - **Dedicated allocations** — single-resource `vkAllocateMemory` for
//!   resources that are too large for the pool block size, that the driver
//!   requires to be dedicated, or that the user explicitly opts into.
//! - **Memory-type selection** by `(required, preferred)` property flag
//!   pair.
//! - **Statistics** — per-pool, per-type, and total: bytes used, bytes
//!   free, allocation count, free-region count, peak usage.
//! - **Budget queries** — when `VK_EXT_memory_budget` is enabled and
//!   loadable, [`Allocator::query_budget`] returns the per-heap soft
//!   budget the driver suggests respecting.
//!
//! The default block size is **256 MiB** for "large" heaps (≥ 4 GiB) and
//! **64 MiB** for "small" heaps. Allocations larger than half the block
//! size automatically fall through to a dedicated allocation. These
//! thresholds match VMA's defaults.
//!
//! ## Example
//!
//! ```ignore
//! use spock::safe::{Allocator, AllocationCreateInfo, AllocationUsage,
//!                   Buffer, BufferCreateInfo, BufferUsage};
//!
//! let allocator = Allocator::new(&device, &physical)?;
//!
//! // A device-local storage buffer for compute output.
//! let (buffer, alloc) = allocator.create_buffer(
//!     BufferCreateInfo {
//!         size: 4 * 1024 * 1024,
//!         usage: BufferUsage::STORAGE_BUFFER,
//!     },
//!     AllocationCreateInfo {
//!         usage: AllocationUsage::DeviceLocal,
//!         ..Default::default()
//!     },
//! )?;
//!
//! // ... use the buffer ...
//!
//! // The Drop impl on `alloc` returns the sub-allocation to its TLSF
//! // pool; the buffer's own Drop calls vkDestroyBuffer.
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::raw::bindings::*;

use super::buffer::{Buffer, BufferCreateInfo, MemoryRequirements};
use super::device::DeviceInner;
use super::image::{Image, Image2dCreateInfo};
use super::physical::PhysicalDevice;
use super::{Device, Error, Result, check};

mod linear;
mod tlsf;

use self::linear::Linear;
use self::tlsf::{Tlsf, TlsfAllocation};

/// Default block size for small heaps (< 4 GiB), in bytes.
const SMALL_HEAP_BLOCK_SIZE: u64 = 64 * 1024 * 1024;
/// Default block size for large heaps (>= 4 GiB), in bytes.
const LARGE_HEAP_BLOCK_SIZE: u64 = 256 * 1024 * 1024;
/// Heap-size threshold for picking the larger block size.
const LARGE_HEAP_THRESHOLD: u64 = 4 * 1024 * 1024 * 1024;

/// Hint about how the user intends to access the allocation. This drives
/// memory-type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationUsage {
    /// "I don't care, pick whatever works." Mostly equivalent to
    /// `DeviceLocal` on discrete GPUs.
    #[default]
    Auto,
    /// `DEVICE_LOCAL`. Fastest GPU access; not directly mappable on
    /// discrete GPUs.
    DeviceLocal,
    /// `HOST_VISIBLE | HOST_COHERENT`. Mappable from the host; suitable
    /// for staging buffers and persistent uploads.
    HostVisible,
    /// `HOST_VISIBLE | HOST_COHERENT | DEVICE_LOCAL` (BAR / ReBAR /
    /// integrated GPUs). Falls back to `HostVisible` when not available.
    HostVisibleDeviceLocal,
}

/// Strategy used to lay out allocations within a pool's memory blocks.
///
/// `FreeList` (TLSF) is the default and the right choice for general
/// long-lived mixed-size allocations: O(1) alloc/free, low fragmentation.
///
/// `Linear` is a bump pointer that only frees on
/// [`Allocator::reset_pool`]; suitable for transient per-frame or
/// per-pass allocations where you batch-reset everything at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AllocationStrategy {
    /// Two-Level Segregated Fit free-list (TLSF) — general purpose.
    #[default]
    FreeList,
    /// Linear bump allocator — supports `reset_pool` only.
    Linear,
}

/// Opaque handle to a custom user pool created via
/// [`Allocator::create_pool`]. Pass to
/// [`AllocationCreateInfo::pool`] to allocate from this pool, and to
/// [`Allocator::destroy_pool`] / [`Allocator::reset_pool`] /
/// [`Allocator::pool_statistics`] to manage it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PoolHandle(u64);

/// Parameters for [`Allocator::create_pool`]. Custom pools let users
/// segregate allocations (e.g. all transient buffers from one frame go
/// into a single linear pool that gets reset together) and pick a
/// non-default strategy.
#[derive(Debug, Clone, Copy)]
pub struct PoolCreateInfo {
    /// Memory type the pool will allocate from. Use
    /// [`PhysicalDevice::find_memory_type`] to pick this.
    pub memory_type_index: u32,
    /// Strategy to use within each block.
    pub strategy: AllocationStrategy,
    /// Initial / per-block size in bytes. Set to 0 to use the
    /// allocator's default for the heap (256 MiB on heaps ≥ 4 GiB,
    /// 64 MiB on smaller heaps).
    pub block_size: u64,
    /// Maximum number of blocks the pool may grow to. Set to 0 for
    /// unlimited (the default). Linear pools always have exactly 1 block.
    pub max_block_count: u32,
}

/// Per-allocation knobs the user can set when creating a resource.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocationCreateInfo {
    /// Hint about how the allocation will be used.
    pub usage: AllocationUsage,
    /// If `true`, force a dedicated `vkAllocateMemory` even when the
    /// pool could satisfy the request. Useful for resources whose contents
    /// are owned by an external API (e.g. swapchain images, exportable
    /// memory).
    pub dedicated: bool,
    /// If `true`, the underlying block is mapped persistently and the
    /// allocation exposes a host pointer for the lifetime of the
    /// allocation. Only meaningful for host-visible memory.
    pub mapped: bool,
    /// Allocate from a specific custom pool (created via
    /// [`Allocator::create_pool`]). When `None`, the default per-memory-type
    /// pool is used.
    pub pool: Option<PoolHandle>,
}

/// Aggregate statistics for the [`Allocator`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AllocationStatistics {
    /// Total `VkDeviceMemory` bytes the allocator currently owns
    /// (allocated from Vulkan, regardless of whether they're sub-allocated
    /// to user resources).
    pub block_bytes: u64,
    /// Bytes the allocator has handed out as user allocations.
    pub allocation_bytes: u64,
    /// Number of `VkDeviceMemory` blocks currently held.
    pub block_count: u32,
    /// Number of currently outstanding user allocations.
    pub allocation_count: u32,
    /// Number of separate free regions across all blocks (a measure of
    /// fragmentation: lower is better).
    pub free_region_count: u32,
    /// Peak `allocation_bytes` over the lifetime of this allocator.
    pub peak_allocation_bytes: u64,
    /// Number of allocations that fell through to dedicated
    /// `vkAllocateMemory` rather than being sub-allocated.
    pub dedicated_allocation_count: u32,
}

/// Strategy state stored alongside a block of device memory.
enum BlockStrategy {
    Tlsf(Tlsf),
    Linear(Linear),
}

impl BlockStrategy {
    fn allocation_count(&self) -> u32 {
        match self {
            Self::Tlsf(t) => t.allocation_count(),
            Self::Linear(l) => l.allocation_count(),
        }
    }
    fn free_region_count(&self) -> u32 {
        match self {
            Self::Tlsf(t) => t.free_region_count(),
            // A linear allocator has at most one free region (the
            // unused tail).
            Self::Linear(l) => {
                if l.free_bytes() == 0 {
                    0
                } else {
                    1
                }
            }
        }
    }
}

/// One block of `VkDeviceMemory` plus its allocation strategy and
/// persistent mapping pointer.
#[allow(dead_code)] // capacity / memory_type_index are diagnostic-only fields.
struct Block {
    memory: VkDeviceMemory,
    capacity: u64,
    memory_type_index: u32,
    strategy: BlockStrategy,
    /// Persistent mapped pointer, if the block was created with `mapped`.
    mapped_ptr: *mut std::ffi::c_void,
}

// Safety: Block is owned by Allocator (which is Send) and accessed only
// behind a Mutex.
unsafe impl Send for Block {}
unsafe impl Sync for Block {}

/// One pool of [`Block`]s, all backed by the same memory type. The
/// per-memory-type default pools are TLSF only; user-created custom
/// pools may use either strategy.
#[allow(dead_code)] // memory_type_index / block_size / max_block_count are diagnostic-only.
struct Pool {
    memory_type_index: u32,
    blocks: Vec<Block>,
    block_size: u64,
    max_block_count: u32,
    strategy: AllocationStrategy,
    mapped: bool,
}

impl Pool {
    fn new_default(memory_type_index: u32, block_size: u64) -> Self {
        Self {
            memory_type_index,
            blocks: Vec::new(),
            block_size,
            max_block_count: 0,
            strategy: AllocationStrategy::FreeList,
            mapped: false,
        }
    }

    fn new_custom(info: PoolCreateInfo, default_block_size: u64, mapped: bool) -> Self {
        let block_size = if info.block_size == 0 {
            default_block_size
        } else {
            info.block_size
        };
        // Linear pools are always a single block.
        let max = if info.strategy == AllocationStrategy::Linear {
            1
        } else {
            info.max_block_count
        };
        Self {
            memory_type_index: info.memory_type_index,
            blocks: Vec::new(),
            block_size,
            max_block_count: max,
            strategy: info.strategy,
            mapped,
        }
    }
}

/// Top-level allocator. Owns one pool per memory type and synthesizes
/// dedicated allocations for resources that don't fit the pool model.
///
/// `Allocator` is `Send + Sync`; internal state is protected by a single
/// `Mutex`. For most workloads this is plenty — Vulkan memory operations
/// are infrequent and the critical section is small. Highly contended
/// workloads can use multiple `Allocator` instances per device.
pub struct Allocator {
    inner: Arc<AllocatorInner>,
}

struct AllocatorInner {
    device: Arc<DeviceInner>,
    /// Kept for budget queries via VK_EXT_memory_budget.
    physical: PhysicalDevice,
    /// Cached `VkPhysicalDeviceMemoryProperties` so we don't re-query.
    memory_properties: VkPhysicalDeviceMemoryProperties,
    /// One pool per memory type. None until first use.
    pools: Mutex<PoolState>,
}

struct PoolState {
    /// Per-memory-type default TLSF pools, indexed by memory type.
    pools: Vec<Option<Pool>>,
    /// User-created custom pools, keyed by `PoolHandle`.
    custom_pools: HashMap<u64, Pool>,
    next_pool_id: u64,
    statistics: AllocationStatistics,
    dedicated_blocks: Vec<DedicatedBlock>,
}

/// A `vkAllocateMemory`'d block that holds exactly one user allocation.
#[allow(dead_code)] // memory_type_index is diagnostic-only.
struct DedicatedBlock {
    memory: VkDeviceMemory,
    size: u64,
    memory_type_index: u32,
    mapped_ptr: *mut std::ffi::c_void,
    /// Slot id used to free this dedicated allocation. Stable for the
    /// lifetime of the allocator.
    id: u64,
}

// Safety: DedicatedBlock is only accessed inside the Allocator's Mutex,
// and the raw pointer is only used while the underlying VkDeviceMemory
// is still mapped (which we control).
unsafe impl Send for DedicatedBlock {}
unsafe impl Sync for DedicatedBlock {}

/// A live allocation handed out by the [`Allocator`]. Free it explicitly
/// by passing it to [`Allocator::free`], or by dropping the
/// [`Buffer`]/[`Image`] returned by `create_buffer` / `create_image_2d`.
#[derive(Debug, Clone)]
pub struct Allocation {
    pub(crate) memory: VkDeviceMemory,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) memory_type_index: u32,
    /// Persistent mapped pointer for the *containing block*, if any.
    /// To get a pointer to this allocation specifically, add `offset`.
    pub(crate) mapped_ptr: *mut std::ffi::c_void,
    /// How to free this allocation: default pool, custom pool, or
    /// dedicated.
    pub(crate) kind: AllocationKind,
}

// Safety: the raw pointer inside Allocation is only safe to deref while
// the parent Allocator is alive. Allocation values are owned by exactly
// one Buffer/Image at a time, and the Allocator outlives both via Arc.
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

#[derive(Debug, Clone)]
pub(crate) enum AllocationKind {
    /// Sub-allocation from the default per-memory-type pool.
    DefaultPool {
        memory_type_index: u32,
        block_index: u32,
        /// TLSF side-table id used to free the sub-allocation.
        tlsf_block_id: u32,
    },
    /// Sub-allocation from a custom user pool.
    CustomPool {
        pool_id: u64,
        block_index: u32,
        /// TLSF side-table id (only meaningful for FreeList strategy
        /// pools; ignored for Linear pools whose blocks free in bulk
        /// via reset_pool).
        tlsf_block_id: u32,
    },
    /// Standalone `vkAllocateMemory` allocation.
    Dedicated { id: u64 },
}

impl Allocation {
    /// Returns the raw `VkDeviceMemory` handle this allocation lives in.
    pub fn memory(&self) -> VkDeviceMemory {
        self.memory
    }

    /// Byte offset of this allocation within its `VkDeviceMemory`.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Size of this allocation in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Memory type index this allocation lives in.
    pub fn memory_type_index(&self) -> u32 {
        self.memory_type_index
    }

    /// Returns the host-visible mapped pointer for this allocation, if
    /// the allocation was created `mapped: true` and the underlying
    /// memory type is host-visible. The pointer is into the parent
    /// `VkDeviceMemory` at this allocation's `offset`.
    pub fn mapped_ptr(&self) -> Option<*mut std::ffi::c_void> {
        if self.mapped_ptr.is_null() {
            None
        } else {
            // Safety: the parent block stays mapped for as long as this
            // allocation lives.
            unsafe { Some(self.mapped_ptr.add(self.offset as usize)) }
        }
    }
}

impl Allocator {
    /// Create a new allocator for the given device.
    pub fn new(device: &Device, physical: &PhysicalDevice) -> Result<Self> {
        let get = physical
            .instance()
            .vkGetPhysicalDeviceMemoryProperties
            .ok_or(Error::MissingFunction(
                "vkGetPhysicalDeviceMemoryProperties",
            ))?;

        // Safety: handle is valid; output is fully overwritten.
        let mut props: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
        unsafe { get(physical.raw(), &mut props) };

        let pools: Vec<Option<Pool>> = (0..props.memoryTypeCount).map(|_| None).collect();

        Ok(Self {
            inner: Arc::new(AllocatorInner {
                device: Arc::clone(&device.inner),
                physical: physical.clone(),
                memory_properties: props,
                pools: Mutex::new(PoolState {
                    pools,
                    custom_pools: HashMap::new(),
                    next_pool_id: 1,
                    statistics: AllocationStatistics::default(),
                    dedicated_blocks: Vec::new(),
                }),
            }),
        })
    }

    /// Cheap clone — the underlying state is shared via `Arc`.
    pub fn clone_inner(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Snapshot the current statistics.
    pub fn statistics(&self) -> AllocationStatistics {
        self.inner.pools.lock().unwrap().statistics
    }

    /// Query the per-heap memory budget via `VK_EXT_memory_budget`.
    ///
    /// Returns `None` only when `vkGetPhysicalDeviceMemoryProperties2`
    /// is not loaded (Vulkan 1.0 without
    /// `VK_KHR_get_physical_device_properties2`). The budget/usage values
    /// inside the returned [`crate::safe::MemoryBudget`] are only
    /// meaningful when `VK_EXT_memory_budget` was enabled at instance
    /// creation time, but the heap-count and structural shape are always
    /// usable.
    pub fn query_budget(&self) -> Option<crate::safe::MemoryBudget> {
        self.inner.physical.memory_budget()
    }

    /// Returns the cached `PhysicalDevice` this allocator was created
    /// with. Useful for follow-up queries that the allocator does not
    /// proxy directly.
    pub fn physical_device(&self) -> &PhysicalDevice {
        &self.inner.physical
    }

    /// Create a custom user pool. Returns an opaque [`PoolHandle`] you
    /// can pass to [`AllocationCreateInfo::pool`] to allocate from this
    /// pool, and to [`destroy_pool`](Self::destroy_pool),
    /// [`reset_pool`](Self::reset_pool), and
    /// [`pool_statistics`](Self::pool_statistics) to manage it.
    ///
    /// `Linear` pools always grow to exactly one block. `FreeList`
    /// pools grow on demand up to `info.max_block_count` (or unlimited
    /// when `max_block_count == 0`).
    pub fn create_pool(&self, info: PoolCreateInfo) -> Result<PoolHandle> {
        let mut state = self.inner.pools.lock().unwrap();
        let id = state.next_pool_id;
        state.next_pool_id += 1;
        let block_size = self.heap_block_size_for_type(info.memory_type_index);
        let pool = Pool::new_custom(info, block_size, false);
        state.custom_pools.insert(id, pool);
        Ok(PoolHandle(id))
    }

    /// Destroy a custom pool, freeing all its underlying
    /// `VkDeviceMemory` blocks. The caller must ensure no live
    /// allocations from this pool are still in use.
    pub fn destroy_pool(&self, handle: PoolHandle) {
        let mut state = self.inner.pools.lock().unwrap();
        if let Some(mut pool) = state.custom_pools.remove(&handle.0) {
            for block in pool.blocks.drain(..) {
                self.free_block_memory(&block);
                state.statistics.block_bytes =
                    state.statistics.block_bytes.saturating_sub(block.capacity);
                state.statistics.block_count = state.statistics.block_count.saturating_sub(1);
                // Subtract any still-counted allocation bytes.
                let alloc_count = block.strategy.allocation_count();
                state.statistics.allocation_count = state
                    .statistics
                    .allocation_count
                    .saturating_sub(alloc_count);
            }
            self.refresh_free_region_count(&mut state);
        }
    }

    /// Reset a Linear-strategy pool. All allocations from this pool
    /// become invalid; the user is responsible for ensuring no live
    /// references remain. No-op for `FreeList` pools.
    pub fn reset_pool(&self, handle: PoolHandle) {
        let mut state = self.inner.pools.lock().unwrap();
        if let Some(pool) = state.custom_pools.get_mut(&handle.0)
            && pool.strategy == AllocationStrategy::Linear
        {
            // Count all the allocations we're about to invalidate.
            let mut total_count = 0u32;
            let mut total_bytes = 0u64;
            for block in pool.blocks.iter_mut() {
                total_count += block.strategy.allocation_count();
                if let BlockStrategy::Linear(ref mut l) = block.strategy {
                    total_bytes += l.used_bytes();
                    l.reset();
                }
            }
            state.statistics.allocation_count = state
                .statistics
                .allocation_count
                .saturating_sub(total_count);
            state.statistics.allocation_bytes = state
                .statistics
                .allocation_bytes
                .saturating_sub(total_bytes);
            self.refresh_free_region_count(&mut state);
        }
    }

    /// Snapshot statistics for one custom pool. Returns `None` if the
    /// handle is unknown (or has been destroyed).
    pub fn pool_statistics(&self, handle: PoolHandle) -> Option<AllocationStatistics> {
        let state = self.inner.pools.lock().unwrap();
        let pool = state.custom_pools.get(&handle.0)?;
        let mut stats = AllocationStatistics::default();
        for block in &pool.blocks {
            stats.block_count += 1;
            stats.block_bytes += block.capacity;
            stats.allocation_count += block.strategy.allocation_count();
            stats.free_region_count += block.strategy.free_region_count();
            match &block.strategy {
                BlockStrategy::Tlsf(t) => stats.allocation_bytes += t.used_bytes(),
                BlockStrategy::Linear(l) => stats.allocation_bytes += l.used_bytes(),
            }
        }
        if stats.allocation_bytes > stats.peak_allocation_bytes {
            stats.peak_allocation_bytes = stats.allocation_bytes;
        }
        Some(stats)
    }

    /// Free a previously returned allocation. The user is responsible for
    /// ensuring the resource bound to this allocation has been destroyed
    /// (or has not yet been bound).
    pub fn free(&self, allocation: Allocation) {
        let mut state = self.inner.pools.lock().unwrap();
        match allocation.kind {
            AllocationKind::DefaultPool {
                memory_type_index,
                block_index,
                tlsf_block_id,
            } => {
                if let Some(Some(pool)) = state.pools.get_mut(memory_type_index as usize)
                    && let Some(block) = pool.blocks.get_mut(block_index as usize)
                    && let BlockStrategy::Tlsf(ref mut t) = block.strategy
                {
                    t.free(TlsfAllocation {
                        offset: allocation.offset,
                        size: allocation.size,
                        block_id: tlsf_block_id,
                    });
                }
                state.statistics.allocation_count =
                    state.statistics.allocation_count.saturating_sub(1);
                state.statistics.allocation_bytes = state
                    .statistics
                    .allocation_bytes
                    .saturating_sub(allocation.size);
                self.refresh_free_region_count(&mut state);
            }
            AllocationKind::CustomPool {
                pool_id,
                block_index,
                tlsf_block_id,
            } => {
                if let Some(pool) = state.custom_pools.get_mut(&pool_id)
                    && let Some(block) = pool.blocks.get_mut(block_index as usize)
                    && let BlockStrategy::Tlsf(ref mut t) = block.strategy
                {
                    // Linear pool allocations cannot be freed
                    // individually — caller must use reset_pool. We
                    // silently no-op the per-alloc free in that case.
                    t.free(TlsfAllocation {
                        offset: allocation.offset,
                        size: allocation.size,
                        block_id: tlsf_block_id,
                    });
                }
                state.statistics.allocation_count =
                    state.statistics.allocation_count.saturating_sub(1);
                state.statistics.allocation_bytes = state
                    .statistics
                    .allocation_bytes
                    .saturating_sub(allocation.size);
                self.refresh_free_region_count(&mut state);
            }
            AllocationKind::Dedicated { id } => {
                if let Some(pos) = state.dedicated_blocks.iter().position(|d| d.id == id) {
                    let dedicated = state.dedicated_blocks.swap_remove(pos);
                    // Unmap if needed and free.
                    if !dedicated.mapped_ptr.is_null()
                        && let Some(unmap) = self.inner.device.dispatch.vkUnmapMemory
                    {
                        // Safety: handle is valid.
                        unsafe { unmap(self.inner.device.handle, dedicated.memory) };
                    }
                    if let Some(free) = self.inner.device.dispatch.vkFreeMemory {
                        // Safety: handle is valid; we are the sole owner.
                        unsafe {
                            free(self.inner.device.handle, dedicated.memory, std::ptr::null())
                        };
                    }
                    state.statistics.block_count = state.statistics.block_count.saturating_sub(1);
                    state.statistics.block_bytes =
                        state.statistics.block_bytes.saturating_sub(dedicated.size);
                    state.statistics.dedicated_allocation_count = state
                        .statistics
                        .dedicated_allocation_count
                        .saturating_sub(1);
                    state.statistics.allocation_count =
                        state.statistics.allocation_count.saturating_sub(1);
                    state.statistics.allocation_bytes = state
                        .statistics
                        .allocation_bytes
                        .saturating_sub(allocation.size);
                }
            }
        }
    }

    /// Allocate `size` bytes meeting `requirements` and the `info` hints,
    /// returning an [`Allocation`]. The caller is responsible for binding
    /// the result to a resource via `vkBindBufferMemory` or
    /// `vkBindImageMemory` (use [`create_buffer`](Self::create_buffer) /
    /// [`create_image_2d`](Self::create_image_2d) for the bound case).
    ///
    /// When `info.pool` is `Some`, the allocation comes from the
    /// designated custom pool and `info.usage` is ignored (the pool
    /// already determined the memory type at create time). Otherwise the
    /// per-memory-type default pool is used.
    pub fn allocate(
        &self,
        requirements: MemoryRequirements,
        info: AllocationCreateInfo,
    ) -> Result<Allocation> {
        if let Some(pool_handle) = info.pool {
            return self.allocate_in_custom_pool(pool_handle, &requirements, info);
        }

        let memory_type_index = self
            .pick_memory_type(requirements.memory_type_bits, info.usage)
            .ok_or(Error::Vk(VkResult::ERROR_FEATURE_NOT_PRESENT))?;

        let mut state = self.inner.pools.lock().unwrap();
        let block_size = self.heap_block_size_for_type(memory_type_index);

        // Force dedicated for very large allocations or when explicitly asked.
        let force_dedicated = info.dedicated || requirements.size > block_size / 2;

        if force_dedicated {
            return self.allocate_dedicated(&mut state, memory_type_index, &requirements, info);
        }

        // Lazily create the pool.
        if state.pools[memory_type_index as usize].is_none() {
            state.pools[memory_type_index as usize] =
                Some(Pool::new_default(memory_type_index, block_size));
        }

        // Try existing blocks.
        let pool = state.pools[memory_type_index as usize].as_mut().unwrap();
        for (block_index, block) in pool.blocks.iter_mut().enumerate() {
            let BlockStrategy::Tlsf(ref mut t) = block.strategy else {
                continue;
            };
            if let Some(ta) = t.allocate(requirements.size, requirements.alignment) {
                let alloc = Allocation {
                    memory: block.memory,
                    offset: ta.offset,
                    size: ta.size,
                    memory_type_index,
                    mapped_ptr: block.mapped_ptr,
                    kind: AllocationKind::DefaultPool {
                        memory_type_index,
                        block_index: block_index as u32,
                        tlsf_block_id: ta.block_id,
                    },
                };
                state.statistics.allocation_count += 1;
                state.statistics.allocation_bytes += ta.size;
                if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
                    state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
                }
                self.refresh_free_region_count(&mut state);
                return Ok(alloc);
            }
        }
        // Need to grow: add a new block.
        let new_block_size = block_size.max(
            requirements
                .size
                .next_power_of_two()
                .max(SMALL_HEAP_BLOCK_SIZE),
        );
        let memory = self.raw_allocate(new_block_size, memory_type_index)?;
        let mapped_ptr = if info.mapped && self.is_host_visible(memory_type_index) {
            self.raw_map_persistent(memory)?
        } else {
            std::ptr::null_mut()
        };
        let mut tlsf = Tlsf::new(new_block_size);
        let ta = tlsf
            .allocate(requirements.size, requirements.alignment)
            .ok_or(Error::Vk(VkResult::ERROR_OUT_OF_DEVICE_MEMORY))?;
        let block = Block {
            memory,
            capacity: new_block_size,
            memory_type_index,
            strategy: BlockStrategy::Tlsf(tlsf),
            mapped_ptr,
        };
        let pool = state.pools[memory_type_index as usize].as_mut().unwrap();
        pool.blocks.push(block);
        let block_index = pool.blocks.len() as u32 - 1;

        state.statistics.block_bytes += new_block_size;
        state.statistics.block_count += 1;
        state.statistics.allocation_count += 1;
        state.statistics.allocation_bytes += ta.size;
        if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
            state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
        }
        self.refresh_free_region_count(&mut state);

        Ok(Allocation {
            memory,
            offset: ta.offset,
            size: ta.size,
            memory_type_index,
            mapped_ptr,
            kind: AllocationKind::DefaultPool {
                memory_type_index,
                block_index,
                tlsf_block_id: ta.block_id,
            },
        })
    }

    /// Allocate from a specific custom pool.
    fn allocate_in_custom_pool(
        &self,
        handle: PoolHandle,
        requirements: &MemoryRequirements,
        info: AllocationCreateInfo,
    ) -> Result<Allocation> {
        let mut state = self.inner.pools.lock().unwrap();
        // Snapshot what we need from the pool before we mutate it (to
        // avoid double borrows).
        let (memory_type_index, strategy, block_size, max_blocks) = {
            let pool = state
                .custom_pools
                .get(&handle.0)
                .ok_or(Error::Vk(VkResult::ERROR_OUT_OF_POOL_MEMORY))?;
            (
                pool.memory_type_index,
                pool.strategy,
                pool.block_size,
                pool.max_block_count,
            )
        };

        // First try existing blocks.
        let pool = state.custom_pools.get_mut(&handle.0).unwrap();
        for (block_index, block) in pool.blocks.iter_mut().enumerate() {
            match &mut block.strategy {
                BlockStrategy::Tlsf(t) => {
                    if let Some(ta) = t.allocate(requirements.size, requirements.alignment) {
                        let alloc = Allocation {
                            memory: block.memory,
                            offset: ta.offset,
                            size: ta.size,
                            memory_type_index,
                            mapped_ptr: block.mapped_ptr,
                            kind: AllocationKind::CustomPool {
                                pool_id: handle.0,
                                block_index: block_index as u32,
                                tlsf_block_id: ta.block_id,
                            },
                        };
                        state.statistics.allocation_count += 1;
                        state.statistics.allocation_bytes += ta.size;
                        if state.statistics.allocation_bytes
                            > state.statistics.peak_allocation_bytes
                        {
                            state.statistics.peak_allocation_bytes =
                                state.statistics.allocation_bytes;
                        }
                        self.refresh_free_region_count(&mut state);
                        return Ok(alloc);
                    }
                }
                BlockStrategy::Linear(l) => {
                    if let Some(la) = l.allocate(requirements.size, requirements.alignment) {
                        let alloc = Allocation {
                            memory: block.memory,
                            offset: la.offset,
                            size: la.size,
                            memory_type_index,
                            mapped_ptr: block.mapped_ptr,
                            kind: AllocationKind::CustomPool {
                                pool_id: handle.0,
                                block_index: block_index as u32,
                                tlsf_block_id: 0, // unused for linear
                            },
                        };
                        state.statistics.allocation_count += 1;
                        state.statistics.allocation_bytes += la.size;
                        if state.statistics.allocation_bytes
                            > state.statistics.peak_allocation_bytes
                        {
                            state.statistics.peak_allocation_bytes =
                                state.statistics.allocation_bytes;
                        }
                        self.refresh_free_region_count(&mut state);
                        return Ok(alloc);
                    }
                }
            }
        }

        // Out of room — allocate a new block if we're allowed.
        if max_blocks > 0 && pool.blocks.len() as u32 >= max_blocks {
            return Err(Error::Vk(VkResult::ERROR_OUT_OF_DEVICE_MEMORY));
        }
        let new_block_size = block_size.max(requirements.size.next_power_of_two().max(64));
        let memory = self.raw_allocate(new_block_size, memory_type_index)?;
        let mapped_ptr = if info.mapped && self.is_host_visible(memory_type_index) {
            self.raw_map_persistent(memory)?
        } else {
            std::ptr::null_mut()
        };

        // Build the strategy + first allocation in one shot.
        let (block_strategy, alloc_offset, alloc_size, tlsf_block_id) = match strategy {
            AllocationStrategy::FreeList => {
                let mut t = Tlsf::new(new_block_size);
                let ta = t
                    .allocate(requirements.size, requirements.alignment)
                    .ok_or(Error::Vk(VkResult::ERROR_OUT_OF_DEVICE_MEMORY))?;
                (BlockStrategy::Tlsf(t), ta.offset, ta.size, ta.block_id)
            }
            AllocationStrategy::Linear => {
                let mut l = Linear::new(new_block_size);
                let la = l
                    .allocate(requirements.size, requirements.alignment)
                    .ok_or(Error::Vk(VkResult::ERROR_OUT_OF_DEVICE_MEMORY))?;
                (BlockStrategy::Linear(l), la.offset, la.size, 0)
            }
        };

        let block = Block {
            memory,
            capacity: new_block_size,
            memory_type_index,
            strategy: block_strategy,
            mapped_ptr,
        };
        let pool = state.custom_pools.get_mut(&handle.0).unwrap();
        pool.blocks.push(block);
        let block_index = pool.blocks.len() as u32 - 1;

        state.statistics.block_bytes += new_block_size;
        state.statistics.block_count += 1;
        state.statistics.allocation_count += 1;
        state.statistics.allocation_bytes += alloc_size;
        if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
            state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
        }
        self.refresh_free_region_count(&mut state);

        Ok(Allocation {
            memory,
            offset: alloc_offset,
            size: alloc_size,
            memory_type_index,
            mapped_ptr,
            kind: AllocationKind::CustomPool {
                pool_id: handle.0,
                block_index,
                tlsf_block_id,
            },
        })
    }

    /// Convenience: allocate memory and bind it to a freshly-created
    /// buffer in one call.
    pub fn create_buffer(
        &self,
        info: BufferCreateInfo,
        alloc_info: AllocationCreateInfo,
    ) -> Result<(Buffer, Allocation)> {
        // Build a Device handle from our cached Arc.
        let device_for_buffer = Device {
            inner: Arc::clone(&self.inner.device),
        };
        let buffer = Buffer::new(&device_for_buffer, info)?;
        let req = buffer.memory_requirements();
        let allocation = self.allocate(req, alloc_info)?;
        // Bind via the buffer's bind_memory using a transient DeviceMemory
        // wrapper. We don't actually own the memory through DeviceMemory
        // here — the Allocator owns the lifetime — so we use the raw
        // vkBindBufferMemory directly.
        let bind = self
            .inner
            .device
            .dispatch
            .vkBindBufferMemory
            .ok_or(Error::MissingFunction("vkBindBufferMemory"))?;
        // Safety: device, buffer, and memory handles are all valid; we
        // computed offset to satisfy the buffer's alignment.
        check(unsafe {
            bind(
                self.inner.device.handle,
                buffer.raw(),
                allocation.memory,
                allocation.offset,
            )
        })?;
        Ok((buffer, allocation))
    }

    /// Convenience: allocate memory and bind it to a freshly-created 2D
    /// image in one call.
    pub fn create_image_2d(
        &self,
        info: Image2dCreateInfo,
        alloc_info: AllocationCreateInfo,
    ) -> Result<(Image, Allocation)> {
        let device_for_image = Device {
            inner: Arc::clone(&self.inner.device),
        };
        let image = Image::new_2d(&device_for_image, info)?;
        let req = image.memory_requirements();
        let allocation = self.allocate(req, alloc_info)?;
        let bind = self
            .inner
            .device
            .dispatch
            .vkBindImageMemory
            .ok_or(Error::MissingFunction("vkBindImageMemory"))?;
        // Safety: device, image, and memory handles are all valid.
        check(unsafe {
            bind(
                self.inner.device.handle,
                image.raw(),
                allocation.memory,
                allocation.offset,
            )
        })?;
        Ok((image, allocation))
    }

    // ----- internals -----

    fn allocate_dedicated(
        &self,
        state: &mut PoolState,
        memory_type_index: u32,
        requirements: &MemoryRequirements,
        info: AllocationCreateInfo,
    ) -> Result<Allocation> {
        let memory = self.raw_allocate(requirements.size, memory_type_index)?;
        let mapped_ptr = if info.mapped && self.is_host_visible(memory_type_index) {
            self.raw_map_persistent(memory)?
        } else {
            std::ptr::null_mut()
        };
        let id = state
            .dedicated_blocks
            .iter()
            .map(|d| d.id)
            .max()
            .map_or(0, |m| m + 1);
        state.dedicated_blocks.push(DedicatedBlock {
            memory,
            size: requirements.size,
            memory_type_index,
            mapped_ptr,
            id,
        });
        state.statistics.block_bytes += requirements.size;
        state.statistics.block_count += 1;
        state.statistics.dedicated_allocation_count += 1;
        state.statistics.allocation_count += 1;
        state.statistics.allocation_bytes += requirements.size;
        if state.statistics.allocation_bytes > state.statistics.peak_allocation_bytes {
            state.statistics.peak_allocation_bytes = state.statistics.allocation_bytes;
        }

        Ok(Allocation {
            memory,
            offset: 0,
            size: requirements.size,
            memory_type_index,
            mapped_ptr,
            kind: AllocationKind::Dedicated { id },
        })
    }

    fn raw_allocate(&self, size: u64, memory_type_index: u32) -> Result<VkDeviceMemory> {
        let allocate = self
            .inner
            .device
            .dispatch
            .vkAllocateMemory
            .ok_or(Error::MissingFunction("vkAllocateMemory"))?;

        let info = VkMemoryAllocateInfo {
            sType: VkStructureType::STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize: size,
            memoryTypeIndex: memory_type_index,
            ..Default::default()
        };

        let mut handle: VkDeviceMemory = 0;
        // Safety: info is valid for the call.
        check(unsafe {
            allocate(
                self.inner.device.handle,
                &info,
                std::ptr::null(),
                &mut handle,
            )
        })?;
        Ok(handle)
    }

    fn raw_map_persistent(&self, memory: VkDeviceMemory) -> Result<*mut std::ffi::c_void> {
        let map = self
            .inner
            .device
            .dispatch
            .vkMapMemory
            .ok_or(Error::MissingFunction("vkMapMemory"))?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // Safety: handle is valid; we map the entire range.
        check(unsafe {
            map(
                self.inner.device.handle,
                memory,
                0,
                u64::MAX, // VK_WHOLE_SIZE
                0,
                &mut ptr,
            )
        })?;
        Ok(ptr)
    }

    fn pick_memory_type(&self, type_bits: u32, usage: AllocationUsage) -> Option<u32> {
        // Required + preferred property flag pairs based on usage.
        let (required_a, preferred_a, fallback_required, fallback_preferred) = match usage {
            AllocationUsage::Auto | AllocationUsage::DeviceLocal => (
                0x0001u32, // DEVICE_LOCAL
                0x0001u32, 0u32, 0u32,
            ),
            AllocationUsage::HostVisible => (
                0x0002 | 0x0004, // HOST_VISIBLE | HOST_COHERENT
                0x0002 | 0x0004,
                0x0002,
                0x0002,
            ),
            AllocationUsage::HostVisibleDeviceLocal => (
                0x0001 | 0x0002 | 0x0004,
                0x0001 | 0x0002 | 0x0004,
                0x0002 | 0x0004,
                0x0002 | 0x0004,
            ),
        };
        // First pass: required + preferred.
        if let Some(i) = self.find_type(type_bits, required_a, preferred_a) {
            return Some(i);
        }
        // Second pass: fallback.
        self.find_type(type_bits, fallback_required, fallback_preferred)
    }

    fn find_type(&self, type_bits: u32, required: u32, preferred: u32) -> Option<u32> {
        let mp = &self.inner.memory_properties;
        // First try types that have all `preferred` flags set...
        for i in 0..mp.memoryTypeCount {
            if (type_bits & (1 << i)) == 0 {
                continue;
            }
            let flags = mp.memoryTypes[i as usize].propertyFlags;
            if (flags & required) == required && (flags & preferred) == preferred {
                return Some(i);
            }
        }
        // ...otherwise any type that meets `required`.
        for i in 0..mp.memoryTypeCount {
            if (type_bits & (1 << i)) == 0 {
                continue;
            }
            let flags = mp.memoryTypes[i as usize].propertyFlags;
            if (flags & required) == required {
                return Some(i);
            }
        }
        None
    }

    fn is_host_visible(&self, memory_type_index: u32) -> bool {
        let mp = &self.inner.memory_properties;
        let flags = mp.memoryTypes[memory_type_index as usize].propertyFlags;
        (flags & 0x0002) != 0 // HOST_VISIBLE
    }

    fn heap_block_size_for_type(&self, memory_type_index: u32) -> u64 {
        let mp = &self.inner.memory_properties;
        let heap_index = mp.memoryTypes[memory_type_index as usize].heapIndex;
        let heap_size = mp.memoryHeaps[heap_index as usize].size;
        if heap_size >= LARGE_HEAP_THRESHOLD {
            LARGE_HEAP_BLOCK_SIZE
        } else {
            SMALL_HEAP_BLOCK_SIZE
        }
    }

    fn refresh_free_region_count(&self, state: &mut PoolState) {
        let mut total = 0u32;
        for pool in state.pools.iter().flatten() {
            for block in &pool.blocks {
                total += block.strategy.free_region_count();
            }
        }
        for pool in state.custom_pools.values() {
            for block in &pool.blocks {
                total += block.strategy.free_region_count();
            }
        }
        state.statistics.free_region_count = total;
    }

    /// Unmap (if mapped) and `vkFreeMemory` a block. Used by both
    /// `destroy_pool` and the allocator's own `Drop`.
    fn free_block_memory(&self, block: &Block) {
        if !block.mapped_ptr.is_null()
            && let Some(unmap) = self.inner.device.dispatch.vkUnmapMemory
        {
            // Safety: handle is valid; we are about to free it.
            unsafe { unmap(self.inner.device.handle, block.memory) };
        }
        if let Some(free) = self.inner.device.dispatch.vkFreeMemory {
            // Safety: handle is valid; we are the sole owner.
            unsafe { free(self.inner.device.handle, block.memory, std::ptr::null()) };
        }
    }
}

impl Drop for AllocatorInner {
    fn drop(&mut self) {
        // Free all per-memory-type pool blocks, custom pool blocks, and
        // dedicated allocations. Inlined the unmap+free logic since the
        // helper method on Allocator wants &Allocator, not
        // &AllocatorInner.
        let mut state = self.pools.lock().unwrap();
        let unmap = self.device.dispatch.vkUnmapMemory;
        let free = self.device.dispatch.vkFreeMemory;
        let dh = self.device.handle;

        let free_block = |block: &Block| {
            if !block.mapped_ptr.is_null()
                && let Some(u) = unmap
            {
                // Safety: handle is valid; we are about to free it.
                unsafe { u(dh, block.memory) };
            }
            if let Some(f) = free {
                // Safety: handle is valid; we are the sole owner.
                unsafe { f(dh, block.memory, std::ptr::null()) };
            }
        };

        for pool in state.pools.iter_mut().flatten() {
            for block in pool.blocks.drain(..) {
                free_block(&block);
            }
        }
        let custom_keys: Vec<u64> = state.custom_pools.keys().copied().collect();
        for k in custom_keys {
            if let Some(mut pool) = state.custom_pools.remove(&k) {
                for block in pool.blocks.drain(..) {
                    free_block(&block);
                }
            }
        }
        for dedicated in state.dedicated_blocks.drain(..) {
            if !dedicated.mapped_ptr.is_null()
                && let Some(u) = unmap
            {
                unsafe { u(dh, dedicated.memory) };
            }
            if let Some(f) = free {
                unsafe { f(dh, dedicated.memory, std::ptr::null()) };
            }
        }
    }
}
