//! wgpu 设备上下文初始化。

use anyhow::{anyhow, Result};

pub struct GpuContext {
    /// 必须保活：window surface 由该 instance 创建，提前 drop 会使 surface 失效。
    #[allow(dead_code)]
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub fn new(surface: Option<&wgpu::Surface>) -> Result<GpuContext> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        GpuContext::with_instance(instance, surface)
    }

    /// 用已有 Instance 初始化（surface 必须由同一 Instance 创建）。
    pub fn with_instance(instance: wgpu::Instance, surface: Option<&wgpu::Surface>) -> Result<GpuContext> {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: surface,
            force_fallback_adapter: false,
        }))
        .map_err(|e| anyhow!("未找到可用的 GPU 适配器({e};尝试安装 vulkan 驱动或设置 WGPU_BACKEND=gl)"))?;
        log::info!(
            "GPU 适配器: {} ({:?})",
            adapter.get_info().name,
            adapter.get_info().backend
        );
        // Anime4K 超分链把 R32F 中间纹理按可过滤采样绑定:适配器支持时
        // 开启 FLOAT32_FILTERABLE(桌面 NVIDIA/AMD/Intel 均支持)。
        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
            features |= wgpu::Features::FLOAT32_FILTERABLE;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("osu-storyboard-render device"),
                required_features: features,
                ..Default::default()
            },
        ))?;
        Ok(GpuContext { instance, adapter, device, queue })
    }

    /// 阻塞等待 GPU 完成（buffer map 回读前使用）。
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait);
    }
}
