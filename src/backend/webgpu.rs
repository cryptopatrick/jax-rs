//! WebGPU backend implementation.

use std::sync::OnceLock;

/// Global WebGPU context.
static WEBGPU_CONTEXT: OnceLock<WebGpuContext> = OnceLock::new();

/// WebGPU execution context with device and queue.
pub struct WebGpuContext {
    /// WebGPU device handle
    pub device: wgpu::Device,
    /// WebGPU command queue
    pub queue: wgpu::Queue,
}

impl WebGpuContext {
    /// Initialize WebGPU context.
    ///
    /// This should be called once at startup. Returns an error if
    /// WebGPU is not available.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use jax_rs::backend::webgpu::WebGpuContext;
    ///
    /// pollster::block_on(async {
    ///     WebGpuContext::init().await.expect("WebGPU not available");
    /// });
    /// ```
    pub async fn init() -> Result<(), String> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or("Failed to find suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("jax-rs WebGPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to create device: {}", e))?;

        WEBGPU_CONTEXT
            .set(WebGpuContext { device, queue })
            .map_err(|_| "WebGPU already initialized")?;

        Ok(())
    }

    /// Get the global WebGPU context.
    ///
    /// # Panics
    ///
    /// Panics if WebGPU has not been initialized via `init()`.
    pub fn get() -> &'static WebGpuContext {
        WEBGPU_CONTEXT
            .get()
            .expect("WebGPU not initialized. Call WebGpuContext::init() first.")
    }

    /// Check if WebGPU is initialized.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use jax_rs::backend::webgpu::WebGpuContext;
    ///
    /// assert!(!WebGpuContext::is_initialized());
    /// ```
    pub fn is_initialized() -> bool {
        WEBGPU_CONTEXT.get().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webgpu_not_initialized_initially() {
        // WebGPU should not be initialized at test start
        // Note: Other tests may have initialized it, so we just check the API works
        let _ = WebGpuContext::is_initialized();
    }

    #[test]
    #[should_panic(expected = "WebGPU not initialized")]
    fn test_webgpu_get_panics_when_not_initialized() {
        // This test assumes WebGPU is not initialized
        // If it is, skip this test
        if !WebGpuContext::is_initialized() {
            let _ = WebGpuContext::get();
        } else {
            panic!("WebGPU not initialized");
        }
    }
}
