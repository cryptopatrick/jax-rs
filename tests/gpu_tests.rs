//! GPU backend integration tests.
//!
//! These tests require WebGPU to be available.

#[cfg(feature = "webgpu")]
mod gpu {
    use jax_rs::{Array, Device, DType, Shape};
    use jax_rs::backend::webgpu::WebGpuContext;

    /// Initialize WebGPU context for tests
    async fn init_webgpu() {
        // Try to initialize, but ignore error if already initialized
        if let Err(e) = WebGpuContext::init().await {
            // Only panic if the error is NOT "already initialized"
            if !e.contains("already initialized") {
                panic!("Failed to init WebGPU: {}", e);
            }
        }
    }

    #[test]
    fn test_gpu_init() {
        pollster::block_on(async {
            init_webgpu().await;
            // If we get here, initialization succeeded
        });
    }

    #[test]
    fn test_gpu_buffer_creation() {
        pollster::block_on(async {
            init_webgpu().await;

            let cpu_arr = Array::zeros(Shape::new(vec![10]), DType::Float32);
            let arr = cpu_arr.to_device(Device::WebGpu);
            assert_eq!(arr.device(), Device::WebGpu);
            assert_eq!(arr.size(), 10);

            let data = arr.to_vec();
            assert_eq!(data.len(), 10);
            assert!(data.iter().all(|&x| x == 0.0));
        });
    }

    #[test]
    fn test_gpu_add() {
        pollster::block_on(async {
            init_webgpu().await;

            let a = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);
            let b = Array::from_vec(vec![4.0, 5.0, 6.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);

            let c = a.add(&b);
            let result = c.to_vec();

            assert_eq!(result, vec![5.0, 7.0, 9.0]);
        });
    }

    #[test]
    fn test_gpu_subtract() {
        pollster::block_on(async {
            init_webgpu().await;

            let a = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);
            let b = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);

            let c = a.sub(&b);
            let result = c.to_vec();

            assert_eq!(result, vec![9.0, 18.0, 27.0]);
        });
    }

    #[test]
    fn test_gpu_multiply() {
        pollster::block_on(async {
            init_webgpu().await;

            let a = Array::from_vec(vec![2.0, 3.0, 4.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);
            let b = Array::from_vec(vec![5.0, 6.0, 7.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);

            let c = a.mul(&b);
            let result = c.to_vec();

            assert_eq!(result, vec![10.0, 18.0, 28.0]);
        });
    }

    #[test]
    fn test_gpu_divide() {
        pollster::block_on(async {
            init_webgpu().await;

            let a = Array::from_vec(vec![10.0, 20.0, 30.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);
            let b = Array::from_vec(vec![2.0, 4.0, 5.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);

            let c = a.div(&b);
            let result = c.to_vec();

            assert_eq!(result, vec![5.0, 5.0, 6.0]);
        });
    }

    #[test]
    fn test_gpu_matmul() {
        pollster::block_on(async {
            init_webgpu().await;

            let a = Array::from_vec(
                vec![1.0, 2.0, 3.0, 4.0],
                Shape::new(vec![2, 2]),
            ).to_device(Device::WebGpu);
            let b = Array::from_vec(
                vec![5.0, 6.0, 7.0, 8.0],
                Shape::new(vec![2, 2]),
            ).to_device(Device::WebGpu);

            let c = a.matmul(&b);
            let result = c.to_vec();

            // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
            assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
        });
    }

    #[test]
    fn test_gpu_matmul_larger() {
        pollster::block_on(async {
            init_webgpu().await;

            // 4x4 matrix multiplication
            let a = Array::from_vec(
                vec![
                    1.0, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0,
                ],
                Shape::new(vec![4, 4]),
            ).to_device(Device::WebGpu);
            let b = Array::eye(4, None, DType::Float32).to_device(Device::WebGpu);

            let c = a.matmul(&b);
            let result = c.to_vec();

            // A @ I = A
            assert_eq!(result.len(), 16);
            for i in 0..16 {
                assert!((result[i] - ((i + 1) as f32)).abs() < 1e-5);
            }
        });
    }

    #[test]
    fn test_gpu_reduction_sum() {
        pollster::block_on(async {
            init_webgpu().await;

            let arr = Array::from_vec(
                vec![1.0, 2.0, 3.0, 4.0, 5.0],
                Shape::new(vec![5]),
            ).to_device(Device::WebGpu);

            let sum = arr.sum_all();
            assert_eq!(sum, 15.0);
        });
    }

    #[test]
    fn test_gpu_reduction_max() {
        pollster::block_on(async {
            init_webgpu().await;

            let arr = Array::from_vec(
                vec![3.0, 7.0, 2.0, 9.0, 1.0],
                Shape::new(vec![5]),
            ).to_device(Device::WebGpu);

            let max_val = arr.max_all();
            assert_eq!(max_val, 9.0);
        });
    }

    #[test]
    fn test_gpu_reduction_min() {
        pollster::block_on(async {
            init_webgpu().await;

            let arr = Array::from_vec(
                vec![3.0, 7.0, 2.0, 9.0, 1.0],
                Shape::new(vec![5]),
            ).to_device(Device::WebGpu);

            let min_val = arr.min_all();
            assert_eq!(min_val, 1.0);
        });
    }

    #[test]
    fn test_gpu_reduction_large_array() {
        pollster::block_on(async {
            init_webgpu().await;

            // Test two-pass reduction with 1000 elements
            let data: Vec<f32> = (1..=1000).map(|x| x as f32).collect();
            let arr = Array::from_vec(data, Shape::new(vec![1000]))
                .to_device(Device::WebGpu);

            let sum = arr.sum_all();
            // Sum of 1..1000 = 1000 * 1001 / 2 = 500500
            assert!((sum - 500500.0).abs() < 1e-3);
        });
    }

    #[test]
    fn test_gpu_cpu_transfer() {
        pollster::block_on(async {
            init_webgpu().await;

            // CPU -> GPU
            let cpu_arr = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]));
            let gpu_arr = cpu_arr.to_device(Device::WebGpu);

            assert_eq!(gpu_arr.device(), Device::WebGpu);
            assert_eq!(gpu_arr.to_vec(), vec![1.0, 2.0, 3.0]);

            // GPU -> CPU
            let back_to_cpu = gpu_arr.to_device(Device::Cpu);
            assert_eq!(back_to_cpu.device(), Device::Cpu);
            assert_eq!(back_to_cpu.to_vec(), vec![1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn test_gpu_same_device_transfer() {
        pollster::block_on(async {
            init_webgpu().await;

            let arr = Array::from_vec(vec![1.0, 2.0, 3.0], Shape::new(vec![3]))
                .to_device(Device::WebGpu);

            // Transfer to same device should be no-op (or cheap clone)
            let same = arr.to_device(Device::WebGpu);
            assert_eq!(same.device(), Device::WebGpu);
            assert_eq!(same.to_vec(), vec![1.0, 2.0, 3.0]);
        });
    }

    #[test]
    fn test_gpu_mixed_operations() {
        pollster::block_on(async {
            init_webgpu().await;

            // Test chaining multiple GPU operations
            let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]))
                .to_device(Device::WebGpu);
            let b = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], Shape::new(vec![4]))
                .to_device(Device::WebGpu);

            // (a + b) * b = (1+2, 2+2, 3+2, 4+2) * (2,2,2,2) = (6, 8, 10, 12)
            let result = a.add(&b).mul(&b);
            let data = result.to_vec();

            assert_eq!(data, vec![6.0, 8.0, 10.0, 12.0]);
        });
    }

    // TODO: Broadcasting not yet supported on GPU
    // #[test]
    // fn test_gpu_broadcasting() {
    //     pollster::block_on(async {
    //         init_webgpu().await;

    //         let a = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]))
    //             .to_device(Device::WebGpu);
    //         let b = Array::from_vec(vec![10.0, 20.0], Shape::new(vec![2]))
    //             .to_device(Device::WebGpu);

    //         // Broadcasting should work on GPU
    //         let result = a.add(&b);
    //         let data = result.to_vec();

    //         // [[1,2],[3,4]] + [10,20] = [[11,22],[13,24]]
    //         assert_eq!(data, vec![11.0, 22.0, 13.0, 24.0]);
    //     });
    // }
}

#[cfg(not(feature = "webgpu"))]
#[test]
fn gpu_tests_skipped() {
    // This test exists to ensure the file compiles even without webgpu feature
    println!("GPU tests are only available with --features webgpu");
}
