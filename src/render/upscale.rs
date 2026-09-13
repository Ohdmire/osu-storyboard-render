//! 实时超分(FSR 1 / Anime4K),用于视频帧与谱面背景:
//!
//! - [`UpscaleMode::Fsr1`]:AMD FidelityFX Super Resolution 1.0 的
//!   EASU(边缘自适应放大)+ RCAS(对比度自适应锐化)两连 pass,WGSL
//!   compute 移植(MIT,见 `shaders/fsr_*.wgsl`)。任意比例直达目标
//!   尺寸,显存开销最小(输入 + 两张目标尺寸纹理)。
//! - [`UpscaleMode::Anime4K`]:bloc97 Anime4K(MIT)的 wgpu 移植
//!   (`anime4k-wgpu` crate),Mode C(Upscale+Denoise CNN)Medium 档,
//!   2× 整数放大后经双线性缩放到目标尺寸。中间纹理为 32F,1080p 输入
//!   显存开销 ~100-150MB,换来动漫内容显著更好的线条重建。
//! - 两条链都输出 alpha=1(仅处理不透明内容:视频帧 / BG)。
//!
//! 逐帧路径:视频通道原帧上传 → [`Upscaler::process`] → 宿主用
//! `set_frame_texture` 换视频纹理;静态一次性路径:[`upscale_image`]
//! (BG 载入期放大后回读成 CPU 图像)。

use anime4k_wgpu::PipelineExecutor;
use anime4k_wgpu::presets::{Anime4KPerformancePreset, Anime4KPreset};

const WORKGROUP: u32 = 8;

/// 超分算法选择。Anime4K 三条链(lively/Anime4K 的 Mode):
/// A 锐利重建(动漫线条)、B 柔和恢复、C 放大+降噪(压缩噪声视频最稳)。
/// 与分辨率无关 —— 源尺寸 = 目标尺寸时同样执行(restore/锐化在 1:1
/// 下正是主要收益),见 [`Upscaler::process`]。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpscaleMode {
    #[default]
    Off,
    Fsr1,
    Anime4K(A4kMode, A4kQuality),
}

/// Anime4K 链变体(bloc97 Mode A/B/C)。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum A4kMode {
    #[default]
    A,
    B,
    C,
}

/// Anime4K 模型档位(权重全部已编译进 anime4k-wgpu,选择零二进制成本;
/// 变化的只是运行时显存/耗时):S 最快 … UL 最强。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum A4kQuality {
    S,
    #[default]
    M,
    L,
    Vl,
    Ul,
}

impl UpscaleMode {
    /// 形如 `anime4k-a-l`;质量段缺省 = M。裸 `anime4k` = A 档 M。
    pub fn parse(s: &str) -> UpscaleMode {
        let (base, qual) = match s.rsplit_once('-') {
            Some((b, q)) => match q {
                "s" | "m" | "l" | "vl" | "ul" => (b, q),
                _ => (s, "m"),
            },
            None => (s, "m"),
        };
        let mode = match base {
            "fsr" | "fsr1" => return UpscaleMode::Fsr1,
            "anime4k" | "anime4k-a" => A4kMode::A,
            "anime4k-b" => A4kMode::B,
            "anime4k-c" => A4kMode::C,
            _ => return UpscaleMode::Off,
        };
        let q = match qual {
            "s" => A4kQuality::S,
            "l" => A4kQuality::L,
            "vl" => A4kQuality::Vl,
            "ul" => A4kQuality::Ul,
            _ => A4kQuality::M,
        };
        UpscaleMode::Anime4K(mode, q)
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            UpscaleMode::Off => "off",
            UpscaleMode::Fsr1 => "fsr",
            UpscaleMode::Anime4K(m, q) => match (m, q) {
                (A4kMode::A, A4kQuality::S) => "anime4k-a-s",
                (A4kMode::A, A4kQuality::M) => "anime4k-a",
                (A4kMode::A, A4kQuality::L) => "anime4k-a-l",
                (A4kMode::A, A4kQuality::Vl) => "anime4k-a-vl",
                (A4kMode::A, A4kQuality::Ul) => "anime4k-a-ul",
                (A4kMode::B, A4kQuality::S) => "anime4k-b-s",
                (A4kMode::B, A4kQuality::M) => "anime4k-b",
                (A4kMode::B, A4kQuality::L) => "anime4k-b-l",
                (A4kMode::B, A4kQuality::Vl) => "anime4k-b-vl",
                (A4kMode::B, A4kQuality::Ul) => "anime4k-b-ul",
                (A4kMode::C, A4kQuality::S) => "anime4k-c-s",
                (A4kMode::C, A4kQuality::M) => "anime4k-c",
                (A4kMode::C, A4kQuality::L) => "anime4k-c-l",
                (A4kMode::C, A4kQuality::Vl) => "anime4k-c-vl",
                (A4kMode::C, A4kQuality::Ul) => "anime4k-c-ul",
            },
        }
    }
}

/// 可复用的目标尺寸纹理(storage 写入 + 可采样 + 可回读)。
struct OutTex {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    size: (u32, u32),
}

impl OutTex {
    fn ensure(device: &wgpu::Device, slot: Option<OutTex>, size: (u32, u32)) -> OutTex {
        if let Some(t) = slot {
            if t.size == size {
                return t;
            }
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("upscale out"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        OutTex { texture, view, size }
    }
}

/// FSR 的常驻 GPU 资源:pipeline / 布局 / 采样器 / 分辨率 uniform。
struct FsrResources {
    easu_pipeline: wgpu::ComputePipeline,
    easu_bgl: wgpu::BindGroupLayout,
    rcas_pipeline: wgpu::ComputePipeline,
    rcas_bgl: wgpu::BindGroupLayout,
    params_bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// EASU 中间产物(目标尺寸),RCAS 的输入。
    tmp: Option<OutTex>,
}

impl FsrResources {
    fn new(device: &wgpu::Device) -> FsrResources {
        let easu_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fsr easu layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
            ],
        });
        let rcas_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fsr rcas layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
            ],
        });
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fsr params layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fsr params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fsr sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let easu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fsr easu"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fsr easu pl"),
                bind_group_layouts: &[&easu_bgl, &params_bgl],
                push_constant_ranges: &[],
            })),
            module: &device.create_shader_module(wgpu::include_wgsl!("shaders/fsr_easu.wgsl")),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let rcas_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fsr rcas"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fsr rcas pl"),
                bind_group_layouts: &[&rcas_bgl],
                push_constant_ranges: &[],
            })),
            module: &device.create_shader_module(wgpu::include_wgsl!("shaders/fsr_rcas.wgsl")),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        FsrResources { easu_pipeline, easu_bgl, rcas_pipeline, rcas_bgl, params_bgl, params_buf, sampler, tmp: None }
    }
}

/// 双线性缩放 pass(`shaders/scale.wgsl`):Anime4K 的 2×→目标 收尾,
/// 以及 Off 模式下源尺寸 ≠ 目标时的直通缩放。
struct ScaleResources {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

impl ScaleResources {
    fn new(device: &wgpu::Device) -> ScaleResources {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scale layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
            ],
        });
        let params_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("scale params layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scale params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("scale sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scale fit"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("scale pl"),
                bind_group_layouts: &[&bgl, &params_bgl],
                push_constant_ranges: &[],
            })),
            module: &device.create_shader_module(wgpu::include_wgsl!("shaders/scale.wgsl")),
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        ScaleResources { pipeline, bgl, params_bgl, params_buf, sampler }
    }
}

/// Anime4K 执行器绑定(源纹理尺寸变化时重建;源纹理必须常驻同一张)。
struct A4kState {
    executor: PipelineExecutor,
    /// CNN 链的 2× 输出(scale pass 的源)。
    out2x: wgpu::Texture,
    src_size: (u32, u32),
    /// 链配置(模式+档位):变化时重建执行器。
    chain: (A4kMode, A4kQuality),
}

fn dispatch_2d(pass: &mut wgpu::ComputePass, w: u32, h: u32) {
    pass.dispatch_workgroups(w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
}

/// 超分执行器:一条按模式选择的 GPU 链 + 可复用输出。
pub struct Upscaler {
    mode: UpscaleMode,
    device: wgpu::Device,
    queue: wgpu::Queue,
    fsr: Option<FsrResources>,
    scale: Option<ScaleResources>,
    a4k: Option<A4kState>,
    out: Option<OutTex>,
    /// 输出纹理重建计数(尺寸变化时 +1):调用方据此判断是否需要
    /// 重新绑定采样点,复用期间逐帧内容原地更新、零重绑。
    gen: u64,
}

impl Upscaler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, mode: UpscaleMode) -> Upscaler {
        Upscaler {
            mode,
            device: device.clone(),
            queue: queue.clone(),
            fsr: None,
            scale: None,
            a4k: None,
            out: None,
            gen: 0,
        }
    }

    /// 输出纹理代数:与上次绑定时不同 = 纹理重建过,需要重绑。
    pub fn generation(&self) -> u64 {
        self.gen
    }

    pub fn mode(&self) -> UpscaleMode {
        self.mode
    }

    /// 跑链:`source`(TEXTURE_BINDING;Anime4K 模式要求常驻同一张纹理,
    /// 尺寸变化时内部重建)→ 超分 → 返回目标尺寸输出纹理(归自身所有
    /// 并复用;wgpu 纹理为廉价句柄克隆,调用方持返回值即可)。
    /// 命令记录进 `encoder`,提交权在调用方。
    pub fn process(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        dst: (u32, u32),
    ) -> wgpu::Texture {
        let prev = self.out.as_ref().map(|o| o.size);
        let out = OutTex::ensure(&self.device, self.out.take(), dst);
        if prev != Some(dst) {
            self.gen += 1;
        }
        let tex = out.texture.clone();
        match self.mode {
            UpscaleMode::Off => {
                if (source.width(), source.height()) == dst {
                    encoder.copy_texture_to_texture(
                        wgpu::ImageCopyTexture { texture: source, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                        wgpu::ImageCopyTexture { texture: &out.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                        wgpu::Extent3d { width: dst.0, height: dst.1, depth_or_array_layers: 1 },
                    );
                } else {
                    self.run_scale(encoder, source, &out.view, dst);
                }
            }
            UpscaleMode::Fsr1 => self.run_fsr(encoder, source, &out, dst),
            UpscaleMode::Anime4K(..) => self.run_a4k(encoder, source, &out, dst),
        }
        self.out = Some(out);
        tex
    }

    fn run_fsr(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        out: &OutTex,
        dst: (u32, u32),
    ) {
        if self.fsr.is_none() {
            self.fsr = Some(FsrResources::new(&self.device));
        }
        let fsr = self.fsr.as_mut().unwrap();
        let src_size = (source.width(), source.height());
        fsr.tmp = Some(OutTex::ensure(&self.device, fsr.tmp.take(), dst));

        let src_view = source.create_view(&Default::default());
        let params_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fsr params bind"),
            layout: &fsr.params_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: fsr.params_buf.as_entire_binding() }],
        });
        self.queue.write_buffer(
            &fsr.params_buf,
            0,
            bytemuck::bytes_of(&[src_size.0 as f32, src_size.1 as f32, dst.0 as f32, dst.1 as f32]),
        );

        let easu_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fsr easu bind"),
            layout: &fsr.easu_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&fsr.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&fsr.tmp.as_ref().unwrap().view) },
            ],
        });
        let rcas_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fsr rcas bind"),
            layout: &fsr.rcas_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&fsr.tmp.as_ref().unwrap().view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&out.view) },
            ],
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fsr easu"), timestamp_writes: None });
            pass.set_pipeline(&fsr.easu_pipeline);
            pass.set_bind_group(0, &easu_bind, &[]);
            pass.set_bind_group(1, &params_bind, &[]);
            dispatch_2d(&mut pass, dst.0, dst.1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("fsr rcas"), timestamp_writes: None });
            pass.set_pipeline(&fsr.rcas_pipeline);
            pass.set_bind_group(0, &rcas_bind, &[]);
            dispatch_2d(&mut pass, dst.0, dst.1);
        }
    }

    fn run_a4k(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        out: &OutTex,
        dst: (u32, u32),
    ) {
        let src_size = (source.width(), source.height());
        let (variant, quality) = match self.mode {
            UpscaleMode::Anime4K(v, q) => (v, q),
            _ => unreachable!(),
        };
        let chain = (variant, quality);
        if self.a4k.as_ref().is_none_or(|s| s.src_size != src_size || s.chain != chain) {
            // 2× 整数链(CNN 重建+放大),scale pass 收尾到目标尺寸。
            // 档位→模型:S/M/L/VL/UL 权重已全部编译进依赖,零二进制成本。
            let preset = match variant {
                A4kMode::A => Anime4KPreset::ModeA,
                A4kMode::B => Anime4KPreset::ModeB,
                A4kMode::C => Anime4KPreset::ModeC,
            };
            let perf = match quality {
                A4kQuality::S => Anime4KPerformancePreset::Light,
                A4kQuality::M => Anime4KPerformancePreset::Medium,
                A4kQuality::L => Anime4KPerformancePreset::High,
                A4kQuality::Vl => Anime4KPerformancePreset::Ultra,
                A4kQuality::Ul => Anime4KPerformancePreset::Extreme,
            };
            let pipelines = preset.create_pipelines(perf, 2.0);
            let (executor, out2x) = PipelineExecutor::new(&pipelines, &self.device, source);
            self.a4k = Some(A4kState { executor, out2x, src_size, chain });
        }
        self.a4k.as_mut().unwrap().executor.pass(encoder);
        let out2x_view = self.a4k.as_ref().unwrap().out2x.create_view(&Default::default());
        let (w2, h2) = {
            let t = &self.a4k.as_ref().unwrap().out2x;
            (t.width(), t.height())
        };
        self.run_scale_view(encoder, &out2x_view, &out.view, (w2, h2), dst);
    }

    fn run_scale(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        out_view: &wgpu::TextureView,
        dst: (u32, u32),
    ) {
        let view = source.create_view(&Default::default());
        self.run_scale_view(encoder, &view, out_view, (source.width(), source.height()), dst);
    }

    fn run_scale_view(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        src_view: &wgpu::TextureView,
        out_view: &wgpu::TextureView,
        src_size: (u32, u32),
        dst: (u32, u32),
    ) {
        if self.scale.is_none() {
            self.scale = Some(ScaleResources::new(&self.device));
        }
        let sc = self.scale.as_mut().unwrap();
        self.queue.write_buffer(
            &sc.params_buf,
            0,
            bytemuck::bytes_of(&[src_size.0 as f32, src_size.1 as f32, dst.0 as f32, dst.1 as f32]),
        );
        let params_bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale params bind"),
            layout: &sc.params_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: sc.params_buf.as_entire_binding() }],
        });
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scale bind"),
            layout: &sc.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sc.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(out_view) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("scale fit"), timestamp_writes: None });
        pass.set_pipeline(&sc.pipeline);
        pass.set_bind_group(0, &bind, &[]);
        pass.set_bind_group(1, &params_bind, &[]);
        dispatch_2d(&mut pass, dst.0, dst.1);
    }
}

/// 静态图像一次性放大(BG 载入期):上传 → 超分 → 回读,返回目标尺寸
/// 的 `(宽, 高, RGBA)`。`Off`、已达目标尺寸或零尺寸时原样返回输入
/// 字节。临时 GPU 资源(Anime4K 的 32F 中间纹理)随调用结束释放。
/// 裸字节接口:调用方无需依赖 `image` crate。
pub fn upscale_image<'a>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: (u32, u32, &'a [u8]),
    mode: UpscaleMode,
    target: (u32, u32),
) -> (u32, u32, std::borrow::Cow<'a, [u8]>) {
    let (w, h, rgba) = src;
    // 严格大于目标才跳过(超大图经上采样链降采样无意义);等尺寸执行
    // —— restore/锐化正是 1:1 下的主要收益
    if mode == UpscaleMode::Off || w == 0 || h == 0 || w > target.0 || h > target.1 || target.0 == 0 || target.1 == 0 {
        return (w, h, std::borrow::Cow::Borrowed(rgba));
    }
    let staging = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("upscale_image staging"),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture { texture: &staging, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        rgba,
        wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let mut up = Upscaler::new(device, queue, mode);
    let out = {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("upscale_image") });
        let out = up.process(&mut encoder, &staging, target);
        let byte_len = (target.0 * target.1 * 4) as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("upscale_image readback"),
            size: byte_len,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture { texture: &out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::ImageCopyBuffer { buffer: &buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(target.0 * 4), rows_per_image: Some(target.1) } },
            wgpu::Extent3d { width: target.0, height: target.1, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        let slice = buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::Wait);
        let data = {
            let mapped = slice.get_mapped_range();
            mapped.to_vec()
        };
        buf.unmap();
        data
    };
    (target.0, target.1, std::borrow::Cow::Owned(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("adapter");
        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::FLOAT32_FILTERABLE) {
            features |= wgpu::Features::FLOAT32_FILTERABLE;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            ..Default::default()
        }))
        .expect("device");
        (device, queue)
    }

    /// 渐变测试图(非平凡内容,验证超分不是黑屏/垃圾):
    /// r = x / w, g = y / h。
    fn gradient(w: u32, h: u32) -> image::RgbaImage {
        image::RgbaImage::from_fn(w, h, |x, y| {
            image::Rgba([(x * 255 / w.max(1)) as u8, (y * 255 / h.max(1)) as u8, 128, 255])
        })
    }

    fn mean_abs_err(a: &image::RgbaImage, b: &image::RgbaImage) -> f64 {
        let (wa, ha) = a.dimensions();
        assert_eq!((wa, ha), b.dimensions());
        let mut sum = 0u64;
        let mut n = 0u64;
        for (p, q) in a.pixels().zip(b.pixels()) {
            for i in 0..3 {
                sum += p[i].abs_diff(q[i]) as u64;
                n += 1;
            }
        }
        sum as f64 / n as f64
    }

    fn reference_downscale(src: &image::RgbaImage, tw: u32, th: u32) -> image::RgbaImage {
        image::imageops::resize(src, tw, th, image::imageops::FilterType::Triangle)
    }

    fn upscale_to_image(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src: &image::RgbaImage,
        mode: UpscaleMode,
        dst: (u32, u32),
    ) -> image::RgbaImage {
        let (w, h, bytes) = upscale_image(device, queue, (src.width(), src.height(), src.as_raw()), mode, dst);
        image::RgbaImage::from_raw(w, h, bytes.into_owned()).expect("size")
    }

    /// FSR 与 Anime4K 都能跑通:输出尺寸正确、alpha=1、内容与双线性
    /// 参考同量级(超分是重建不是复制,均值差在合理带宽内)。
    #[test]
    fn upscale_chains_run_and_produce_sane_output() {
        let (device, queue) = gpu();
        let src = gradient(320, 180);
        let dst = (640, 360);
        let reference = reference_downscale(&src, dst.0, dst.1);
        for mode in [UpscaleMode::Fsr1, UpscaleMode::Anime4K(A4kMode::A, A4kQuality::M), UpscaleMode::Anime4K(A4kMode::C, A4kQuality::L)] {
            let out = upscale_to_image(&device, &queue, &src, mode, dst);
            assert_eq!(out.dimensions(), dst, "{mode:?} 尺寸错误");
            assert!(out.pixels().all(|p| p[3] == 255), "{mode:?} alpha 应恒 1");
            let err = mean_abs_err(&out, &reference);
            // 渐变图上超分与双线性参考的均值差远小于信号摆幅(255)
            assert!(err < 24.0, "{mode:?} 输出与参考偏差过大: {err}");
        }
    }

    /// Off 原样返回(字节一致);严格大于目标跳过;同尺寸执行
    /// (尺寸不变、字节被重建过)。
    #[test]
    fn upscale_image_passthrough_cases() {
        let (device, queue) = gpu();
        let src = gradient(64, 64);
        let (w, h, b) = upscale_image(&device, &queue, (64, 64, src.as_raw()), UpscaleMode::Off, (128, 128));
        assert_eq!((w, h), (64, 64));
        assert!(b.iter().eq(src.as_raw().iter()));
        // 严格大于目标(128 > 64):跳过,原样返回
        let (w, h, b) = upscale_image(&device, &queue, (128, 128, vec![128u8; 128 * 128 * 4].as_slice()), UpscaleMode::Fsr1, (64, 64));
        assert_eq!((w, h), (128, 128));
        // 同尺寸:执行链(尺寸不变、内容被重建 —— 字节应有所变化)
        let (w, h, b) = upscale_image(&device, &queue, (64, 64, src.as_raw()), UpscaleMode::Fsr1, (64, 64));
        assert_eq!((w, h), (64, 64));
        assert!(!b.iter().eq(src.as_raw().iter()), "同尺寸应执行重建(字节变化)");
    }

    /// Upscaler 逐帧复用:同尺寸连续 process 不重建(代数不变),尺寸
    /// 变化后代数递增。
    #[test]
    fn upscaler_generation_tracks_output_rebuilds() {
        let (device, queue) = gpu();
        let mut up = Upscaler::new(&device, &queue, UpscaleMode::Fsr1);
        let staging = gradient_texture(&device, &queue, 320, 180);
        let mut encoder = device.create_command_encoder(&Default::default());
        up.process(&mut encoder, &staging, (640, 360));
        let g0 = up.generation();
        queue.submit([encoder.finish()]);
        let mut encoder = device.create_command_encoder(&Default::default());
        up.process(&mut encoder, &staging, (640, 360));
        assert_eq!(up.generation(), g0, "同尺寸不应重建");
        up.process(&mut encoder, &staging, (800, 450));
        assert_eq!(up.generation(), g0 + 1, "尺寸变化应重建");
        queue.submit([encoder.finish()]);
    }

    fn gradient_texture(device: &wgpu::Device, queue: &wgpu::Queue, w: u32, h: u32) -> wgpu::Texture {
        let img = gradient(w, h);
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test staging"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture { texture: &tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            img.as_raw(),
            wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        tex
    }
}
