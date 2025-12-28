//! Cross-backend validation tests for jax-rs.
//!
//! These tests verify that operations produce identical results on CPU and GPU.

#[cfg(feature = "webgpu")]
mod cross_backend {
    use jax_rs::{Array, Device, DType, Shape};
    use jax_rs::backend::webgpu::WebGpuContext;

    /// Initialize WebGPU context for tests
    async fn init_webgpu() {
        if let Err(e) = WebGpuContext::init().await {
            if !e.contains("already initialized") {
                panic!("Failed to init WebGPU: {}", e);
            }
        }
    }

    /// Helper to check if CPU and GPU results are close.
    fn assert_backends_close(cpu_result: &Array, gpu_result: &Array, rtol: f32, atol: f32, op_name: &str) {
        let cpu_vec = cpu_result.to_vec();
        let gpu_vec = gpu_result.to_vec();

        assert_eq!(cpu_vec.len(), gpu_vec.len(), "{}: length mismatch", op_name);

        for (i, (&c, &g)) in cpu_vec.iter().zip(gpu_vec.iter()).enumerate() {
            let diff = (c - g).abs();
            let tolerance = atol + rtol * c.abs();
            assert!(
                diff <= tolerance,
                "{} at index {}: CPU={}, GPU={} (diff={}, tol={})",
                op_name, i, c, g, diff, tolerance
            );
        }
    }

    // =============================================================================
    // BINARY OPERATIONS
    // =============================================================================

    #[test]
    fn test_cross_backend_add() {
        pollster::block_on(async {
            init_webgpu().await;

            let a_cpu = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));
            let b_cpu = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![4]));

            let result_cpu = a_cpu.add(&b_cpu);

            let a_gpu = a_cpu.to_device(Device::WebGpu);
            let b_gpu = b_cpu.to_device(Device::WebGpu);
            let result_gpu = a_gpu.add(&b_gpu);

            assert_backends_close(&result_cpu, &result_gpu, 1e-6, 1e-6, "add");
        });
    }

    #[test]
    fn test_cross_backend_mul() {
        pollster::block_on(async {
            init_webgpu().await;

            let a_cpu = Array::from_vec(vec![2.0, 3.0, 4.0, 5.0], Shape::new(vec![4]));
            let b_cpu = Array::from_vec(vec![3.0, 4.0, 5.0, 6.0], Shape::new(vec![4]));

            let result_cpu = a_cpu.mul(&b_cpu);

            let a_gpu = a_cpu.to_device(Device::WebGpu);
            let b_gpu = b_cpu.to_device(Device::WebGpu);
            let result_gpu = a_gpu.mul(&b_gpu);

            assert_backends_close(&result_cpu, &result_gpu, 1e-6, 1e-6, "mul");
        });
    }

    #[test]
    fn test_cross_backend_sub() {
        pollster::block_on(async {
            init_webgpu().await;

            let a_cpu = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new(vec![4]));
            let b_cpu = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));

            let result_cpu = a_cpu.sub(&b_cpu);

            let a_gpu = a_cpu.to_device(Device::WebGpu);
            let b_gpu = b_cpu.to_device(Device::WebGpu);
            let result_gpu = a_gpu.sub(&b_gpu);

            assert_backends_close(&result_cpu, &result_gpu, 1e-6, 1e-6, "sub");
        });
    }

    #[test]
    fn test_cross_backend_div() {
        pollster::block_on(async {
            init_webgpu().await;

            let a_cpu = Array::from_vec(vec![10.0, 20.0, 30.0, 40.0], Shape::new(vec![4]));
            let b_cpu = Array::from_vec(vec![2.0, 4.0, 5.0, 8.0], Shape::new(vec![4]));

            let result_cpu = a_cpu.div(&b_cpu);

            let a_gpu = a_cpu.to_device(Device::WebGpu);
            let b_gpu = b_cpu.to_device(Device::WebGpu);
            let result_gpu = a_gpu.div(&b_gpu);

            assert_backends_close(&result_cpu, &result_gpu, 1e-5, 1e-5, "div");
        });
    }

    // =============================================================================
    // UNARY OPERATIONS
    // =============================================================================
    // TODO: Unary operations (sin, exp, sqrt) are currently CPU-only
    // Uncomment these tests once GPU support is added

    // #[test]
    // fn test_cross_backend_sin() {
    //     pollster::block_on(async {
    //         init_webgpu().await;
    //
    //         let x_cpu = Array::from_vec(vec![0.0, 1.0, 2.0, 3.0], Shape::new(vec![4]));
    //
    //         let result_cpu = x_cpu.sin();
    //
    //         let x_gpu = x_cpu.to_device(Device::WebGpu);
    //         let result_gpu = x_gpu.sin();
    //
    //         assert_backends_close(&result_cpu, &result_gpu, 1e-6, 1e-6, "sin");
    //     });
    // }
    //
    // #[test]
    // fn test_cross_backend_exp() {
    //     pollster::block_on(async {
    //         init_webgpu().await;
    //
    //         let x_cpu = Array::from_vec(vec![0.0, 0.5, 1.0, 1.5], Shape::new(vec![4]));
    //
    //         let result_cpu = x_cpu.exp();
    //
    //         let x_gpu = x_cpu.to_device(Device::WebGpu);
    //         let result_gpu = x_gpu.exp();
    //
    //         assert_backends_close(&result_cpu, &result_gpu, 1e-5, 1e-5, "exp");
    //     });
    // }
    //
    // #[test]
    // fn test_cross_backend_sqrt() {
    //     pollster::block_on(async {
    //         init_webgpu().await;
    //
    //         let x_cpu = Array::from_vec(vec![1.0, 4.0, 9.0, 16.0], Shape::new(vec![4]));
    //
    //         let result_cpu = x_cpu.sqrt();
    //
    //         let x_gpu = x_cpu.to_device(Device::WebGpu);
    //         let result_gpu = x_gpu.sqrt();
    //
    //         assert_backends_close(&result_cpu, &result_gpu, 1e-6, 1e-6, "sqrt");
    //     });
    // }

    // =============================================================================
    // MATRIX OPERATIONS
    // =============================================================================

    #[test]
    fn test_cross_backend_matmul() {
        pollster::block_on(async {
            init_webgpu().await;

            let a_cpu = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![2, 2]));
            let b_cpu = Array::from_vec(vec![5.0, 6.0, 7.0, 8.0], Shape::new(vec![2, 2]));

            let result_cpu = a_cpu.matmul(&b_cpu);

            let a_gpu = a_cpu.to_device(Device::WebGpu);
            let b_gpu = b_cpu.to_device(Device::WebGpu);
            let result_gpu = a_gpu.matmul(&b_gpu);

            assert_backends_close(&result_cpu, &result_gpu, 1e-5, 1e-5, "matmul");
        });
    }

    #[test]
    fn test_cross_backend_matmul_large() {
        pollster::block_on(async {
            init_webgpu().await;

            // 4x4 matrices
            let a_cpu = Array::from_vec(
                (0..16).map(|x| x as f32).collect(),
                Shape::new(vec![4, 4]),
            );
            let b_cpu = Array::from_vec(
                (0..16).map(|x| (x * 2) as f32).collect(),
                Shape::new(vec![4, 4]),
            );

            let result_cpu = a_cpu.matmul(&b_cpu);

            let a_gpu = a_cpu.to_device(Device::WebGpu);
            let b_gpu = b_cpu.to_device(Device::WebGpu);
            let result_gpu = a_gpu.matmul(&b_gpu);

            assert_backends_close(&result_cpu, &result_gpu, 1e-4, 1e-4, "matmul_large");
        });
    }

    // =============================================================================
    // REDUCTIONS
    // =============================================================================

    #[test]
    fn test_cross_backend_sum_all() {
        pollster::block_on(async {
            init_webgpu().await;

            let x_cpu = Array::from_vec(
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                Shape::new(vec![2, 3]),
            );

            let result_cpu = x_cpu.sum_all();

            let x_gpu = x_cpu.to_device(Device::WebGpu);
            let result_gpu = x_gpu.sum_all();

            assert!((result_cpu - result_gpu).abs() < 1e-5,
                   "sum_all: CPU={}, GPU={}", result_cpu, result_gpu);
        });
    }

    #[test]
    fn test_cross_backend_max_all() {
        pollster::block_on(async {
            init_webgpu().await;

            let x_cpu = Array::from_vec(
                vec![3.0, 7.0, 2.0, 9.0, 1.0, 5.0],
                Shape::new(vec![2, 3]),
            );

            let result_cpu = x_cpu.max_all();

            let x_gpu = x_cpu.to_device(Device::WebGpu);
            let result_gpu = x_gpu.max_all();

            assert!((result_cpu - result_gpu).abs() < 1e-6,
                   "max_all: CPU={}, GPU={}", result_cpu, result_gpu);
        });
    }

    #[test]
    fn test_cross_backend_min_all() {
        pollster::block_on(async {
            init_webgpu().await;

            let x_cpu = Array::from_vec(
                vec![3.0, 7.0, 2.0, 9.0, 1.0, 5.0],
                Shape::new(vec![2, 3]),
            );

            let result_cpu = x_cpu.min_all();

            let x_gpu = x_cpu.to_device(Device::WebGpu);
            let result_gpu = x_gpu.min_all();

            assert!((result_cpu - result_gpu).abs() < 1e-6,
                   "min_all: CPU={}, GPU={}", result_cpu, result_gpu);
        });
    }

    // =============================================================================
    // COMPLEX OPERATIONS
    // =============================================================================

    #[test]
    fn test_cross_backend_chain() {
        pollster::block_on(async {
            init_webgpu().await;

            let x_cpu = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], Shape::new(vec![4]));

            // (x + 1) * 2 - use same shape to avoid broadcasting
            let result_cpu = x_cpu.add(&Array::ones(Shape::new(vec![4]), DType::Float32))
                .mul(&Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], Shape::new(vec![4])));

            let x_gpu = x_cpu.to_device(Device::WebGpu);
            let result_gpu = x_gpu.add(&Array::ones(Shape::new(vec![4]), DType::Float32).to_device(Device::WebGpu))
                .mul(&Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], Shape::new(vec![4])).to_device(Device::WebGpu));

            assert_backends_close(&result_cpu, &result_gpu, 1e-5, 1e-5, "chain");
        });
    }
}

#[cfg(not(feature = "webgpu"))]
#[test]
fn cross_backend_tests_skipped() {
    println!("Cross-backend tests are only available with --features webgpu");
}
