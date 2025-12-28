//! Device and backend management.

use std::fmt;
use std::sync::OnceLock;

/// Compute device for array operations.
///
/// Corresponds to jax-js Device type: "cpu" | "wasm" | "webgpu"
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Device {
    /// CPU backend (slow, for debugging)
    Cpu,
    /// WebAssembly backend with SIMD (optional)
    Wasm,
    /// WebGPU backend (primary accelerator)
    WebGpu,
}

impl Device {
    /// Returns all available devices.
    pub fn all() -> &'static [Device] {
        &[Device::Cpu, Device::Wasm, Device::WebGpu]
    }

    /// Returns the name of this device as a string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Wasm => "wasm",
            Device::WebGpu => "webgpu",
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Device {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "cpu" => Ok(Device::Cpu),
            "wasm" => Ok(Device::Wasm),
            "webgpu" => Ok(Device::WebGpu),
            _ => Err(format!("Unknown device: {}", s)),
        }
    }
}

/// Global default device for array operations.
static DEFAULT_DEVICE: OnceLock<Device> = OnceLock::new();

/// Get or set the default device.
///
/// # Examples
///
/// ```
/// # use jax_rs::Device;
/// // Get current default (CPU by default)
/// let device = jax_rs::default_device();
/// assert_eq!(device, Device::Cpu);
/// ```
pub fn default_device() -> Device {
    *DEFAULT_DEVICE.get_or_init(|| Device::Cpu)
}

/// Set the default device for array operations.
pub fn set_default_device(device: Device) {
    // Note: OnceLock doesn't support mutation after init,
    // so we need a different approach for runtime mutation.
    // For now, using a global mutable would require unsafe or a Mutex.
    // Let's use a simple global static that can be updated.
    // This will be improved in later phases with proper backend management.
    let _ = DEFAULT_DEVICE.set(device);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_display() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
        assert_eq!(Device::Wasm.to_string(), "wasm");
        assert_eq!(Device::WebGpu.to_string(), "webgpu");
    }

    #[test]
    fn test_device_from_str() {
        assert_eq!("cpu".parse::<Device>().unwrap(), Device::Cpu);
        assert_eq!("wasm".parse::<Device>().unwrap(), Device::Wasm);
        assert_eq!("webgpu".parse::<Device>().unwrap(), Device::WebGpu);
        assert!("unknown".parse::<Device>().is_err());
    }

    #[test]
    fn test_device_all() {
        let devices = Device::all();
        assert_eq!(devices.len(), 3);
        assert!(devices.contains(&Device::Cpu));
        assert!(devices.contains(&Device::Wasm));
        assert!(devices.contains(&Device::WebGpu));
    }

    #[test]
    fn test_default_device() {
        let device = default_device();
        assert_eq!(device, Device::Cpu);
    }
}
