//! Integration test for the safe wrapper module.
//!
//! Validates the entire safe API end-to-end against a real Vulkan driver.
//! Skips gracefully on systems without Vulkan installed.

use spock::safe::{
    ApiVersion, Buffer, BufferCreateInfo, BufferUsage, CommandPool, DeviceCreateInfo, DeviceMemory,
    Fence, Instance, InstanceCreateInfo, MemoryPropertyFlags, QueueCreateInfo, QueueFlags,
};

#[test]
fn test_safe_instance_creation_and_enumeration() {
    let instance = match Instance::new(InstanceCreateInfo {
        application_name: Some("spock test"),
        api_version: ApiVersion::V1_0,
        ..Default::default()
    }) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("SKIP: Vulkan not available: {e}");
            return;
        }
    };

    // Enumeration should succeed even if there are no devices.
    let physical_devices = instance.enumerate_physical_devices().unwrap();
    println!("Found {} physical device(s)", physical_devices.len());

    for pd in &physical_devices {
        let props = pd.properties();
        assert!(!props.device_name().is_empty());
        assert!(props.api_version().major() >= 1);

        let queue_families = pd.queue_family_properties();
        assert!(
            !queue_families.is_empty(),
            "every device has at least one queue family"
        );
    }
}

#[test]
fn test_safe_device_creation_and_drop() {
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => {
            eprintln!("SKIP: Vulkan not available");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let physical = match physicals.first() {
        Some(p) => p.clone(),
        None => {
            eprintln!("SKIP: no physical devices");
            return;
        }
    };

    let queue_family = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();

    // Create and drop a device. The Drop impl should call vkDestroyDevice.
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
        })
        .expect("device creation should succeed");

    // Verify we can get a queue handle from it.
    let _queue = device.get_queue(queue_family, 0);

    // Verify wait_idle on a fresh device works.
    device
        .wait_idle()
        .expect("wait_idle on idle device should succeed");

    // Drop happens at end of scope.
}

#[test]
fn test_safe_buffer_with_host_visible_memory() {
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => {
            eprintln!("SKIP: Vulkan not available");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let Some(physical) = physicals.first().cloned() else {
        eprintln!("SKIP: no physical devices");
        return;
    };

    let queue_family = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
        })
        .unwrap();

    // Create a buffer.
    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 256,
            usage: BufferUsage::TRANSFER_DST,
        },
    )
    .unwrap();
    assert_eq!(buffer.size(), 256);

    // Query memory requirements.
    let req = buffer.memory_requirements();
    assert!(req.size >= 256);
    assert!(req.alignment.is_power_of_two());

    // Find a compatible host-visible memory type.
    let mem_type = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("host-visible memory should be available on any platform");

    // Allocate and bind.
    let mut memory = DeviceMemory::allocate(&device, req.size, mem_type).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();

    // Map, write, verify, drop.
    {
        let mut mapped = memory.map().unwrap();
        let slice = mapped.as_slice_mut();
        assert_eq!(slice.len() as u64, req.size);
        for (i, b) in slice.iter_mut().enumerate() {
            *b = (i & 0xFF) as u8;
        }
    }

    // Map again and verify the writes persisted (host-coherent so no flushes needed).
    {
        let mapped = memory.map().unwrap();
        let slice = mapped.as_slice();
        for (i, &b) in slice.iter().enumerate() {
            assert_eq!(b, (i & 0xFF) as u8, "byte {i} did not persist");
        }
    }
}

#[test]
fn test_safe_full_gpu_round_trip() {
    let instance = match Instance::new(InstanceCreateInfo::default()) {
        Ok(i) => i,
        Err(_) => {
            eprintln!("SKIP: Vulkan not available");
            return;
        }
    };

    let physicals = instance.enumerate_physical_devices().unwrap();
    let Some(physical) = physicals.first().cloned() else {
        eprintln!("SKIP: no physical devices");
        return;
    };

    let queue_family = physical.find_queue_family(QueueFlags::TRANSFER).unwrap();
    let device = physical
        .create_device(DeviceCreateInfo {
            queue_create_infos: &[QueueCreateInfo {
                queue_family_index: queue_family,
                queue_priorities: vec![1.0],
            }],
        })
        .unwrap();
    let queue = device.get_queue(queue_family, 0);

    let buffer = Buffer::new(
        &device,
        BufferCreateInfo {
            size: 64,
            usage: BufferUsage::TRANSFER_DST,
        },
    )
    .unwrap();

    let req = buffer.memory_requirements();
    let mem_type = physical
        .find_memory_type(
            req.memory_type_bits,
            MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
    let mut memory = DeviceMemory::allocate(&device, req.size, mem_type).unwrap();
    buffer.bind_memory(&memory, 0).unwrap();

    // Pre-write so we can verify the GPU overwrote.
    {
        let mut m = memory.map().unwrap();
        m.as_slice_mut().fill(0);
    }

    // Record a fill command.
    let pool = CommandPool::new(&device, queue_family).unwrap();
    let mut cmd = pool.allocate_primary().unwrap();
    {
        let mut rec = cmd.begin().unwrap();
        rec.fill_buffer(&buffer, 0, 64, 0xCAFEBABE);
        rec.end().unwrap();
    }

    // Submit with a fence and wait.
    let fence = Fence::new(&device).unwrap();
    queue.submit(&[&cmd], Some(&fence)).unwrap();
    fence.wait(u64::MAX).unwrap();

    // Verify the GPU did the write.
    {
        let mapped = memory.map().unwrap();
        let slice = mapped.as_slice();
        let expected: [u8; 4] = 0xCAFEBABEu32.to_ne_bytes();
        for chunk in slice.chunks_exact(4) {
            assert_eq!(chunk, expected, "GPU did not write expected pattern");
        }
    }

    // Everything drops here in the correct order.
}

#[test]
fn test_api_version_encoding() {
    // ApiVersion bit-packing must match the C macro VK_MAKE_API_VERSION exactly.
    let v = ApiVersion::new(0, 1, 3, 250);
    assert_eq!(v.major(), 1);
    assert_eq!(v.minor(), 3);
    assert_eq!(v.patch(), 250);

    let v0 = ApiVersion::V1_0;
    assert_eq!(v0.major(), 1);
    assert_eq!(v0.minor(), 0);
    assert_eq!(v0.patch(), 0);
}

#[test]
fn test_queue_flags_bitor_and_contains() {
    let combined = QueueFlags::GRAPHICS | QueueFlags::COMPUTE;
    assert!(combined.contains(QueueFlags::GRAPHICS));
    assert!(combined.contains(QueueFlags::COMPUTE));
    assert!(!combined.contains(QueueFlags::TRANSFER));
}

#[test]
fn test_memory_property_flags_bitor() {
    let f = MemoryPropertyFlags::HOST_VISIBLE | MemoryPropertyFlags::HOST_COHERENT;
    assert!(f.contains(MemoryPropertyFlags::HOST_VISIBLE));
    assert!(f.contains(MemoryPropertyFlags::HOST_COHERENT));
    assert!(!f.contains(MemoryPropertyFlags::DEVICE_LOCAL));
}

#[test]
fn test_buffer_usage_bitor() {
    let u = BufferUsage::TRANSFER_DST | BufferUsage::STORAGE_BUFFER;
    assert!(u.contains(BufferUsage::TRANSFER_DST));
    assert!(u.contains(BufferUsage::STORAGE_BUFFER));
    assert!(!u.contains(BufferUsage::TRANSFER_SRC));
}
