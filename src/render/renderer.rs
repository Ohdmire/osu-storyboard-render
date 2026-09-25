//! wgpu 精灵渲染器：每个可见精灵一个实例，按层序/文件序绘制（画家算法），
//! 支持普通 alpha 混合与 `P,A` 加色混合两条管线。

use crate::osb::timeline::CompiledStoryboard;
use crate::osb::{model::Layer, timeline::FailState};
use crate::render::texture::{frame_path, normalize_path, Assets};
use bytemuck::{Pod, Zeroable};
use image::RgbaImage;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// osu! 虚拟分辨率：高固定 480，宽随窗口纵横比扩展（宽屏时超出 0..640）。
const OSU_HEIGHT: f32 = 480.0;
const OSU_CENTRE_X: f32 = 320.0;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuInstance {
    pub pos: [f32; 2],
    pub size: [f32; 2],
    pub anchor: [f32; 2],
    pub rotation: f32,
    pub color: [f32; 4],
    pub flip: [f32; 2],
    pub _pad: [f32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Globals {
    mvp: [f32; 16],
}

pub struct Draw {
    pub texture: String,
    pub additive: bool,
    /// 该精灵的暗度预乘(1.0 = 不衰减)。lazer `UserDimContainer` 把
    /// DimLevel 以 Gray(1-dim) 乘在故事板内容上(含 Overlay 代理),跟随
    /// 背景亮度滑块实时变化。
    pub dim: f32,
    pub instance: GpuInstance,
}

struct Pipelines {
    normal: wgpu::RenderPipeline,
    additive: wgpu::RenderPipeline,
    /// REPLACE 混合(无混合,直写):子矩形模式下用巨型透明四边形做
    /// 区域清屏——整附件 LoadOp::Clear 会清掉图集其余内容,不可用。
    clear: wgpu::RenderPipeline,
}

pub struct TextureSlot {
    /// 原始纹理句柄:视频等逐帧更新的外部纹理需要重复 write_texture。
    pub texture: wgpu::Texture,
    pub bind_group: wgpu::BindGroup,
    pub size: [u32; 2],
    /// GPU 占用字节(w*h*4),LRU 记账用。
    pub bytes: usize,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader: wgpu::ShaderModule,
    globals_layout: wgpu::BindGroupLayout,
    tex_layout: wgpu::BindGroupLayout,
    globals_buf: wgpu::Buffer,
    globals_bg: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    quad_vb: wgpu::Buffer,
    quad_ib: wgpu::Buffer,
    pipelines: HashMap<wgpu::TextureFormat, Pipelines>,
    instance_buf: wgpu::Buffer,
    instance_cap: u64,
    textures: HashMap<String, TextureSlot>,
    /// 每个已上传贴图的字节数合计(w*h*4)。
    texture_bytes: usize,
    /// GPU 贴图预算(字节);超出后按 LRU 淘汰不在本帧使用的槽位。
    /// usize::MAX = 不限制(独立播放器行为)。
    max_gpu_bytes: usize,
    /// 帧计数,驱动 last_used 的 LRU 淘汰。
    frame: u64,
    last_used: HashMap<String, u64>,
    /// 1×1 白纹理的绑定组:清屏四边形需要绑一个合法纹理(颜色 0 把
    /// 采样值整体乘成 0,纹理内容无关紧要)。
    dummy_bg: wgpu::BindGroup,
    /// 视频通道超分(None = 关):原帧先上常驻 staging 纹理,经
    /// [`Upscaler`] 放大后以 `set_frame_texture` 换入视频槽位。
    upscale: Option<crate::render::upscale::Upscaler>,
    /// 超分目标尺寸(SB 合成槽分辨率,视频精灵的最终采样分辨率)。
    upscale_target: (u32, u32),
    /// 视频原帧的常驻 staging 纹理(尺寸随视频变化重建)。
    video_staging: Option<wgpu::Texture>,
    /// 上次换绑视频槽位时的超分代数(输出纹理重建检测)。
    video_upscale_gen_seen: u64,
    /// 视频超分配置日志只打一次(尺寸/模式变化时重置)。
    video_upscale_logged: bool,
}

impl Renderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Renderer {
        let shader = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });

        let tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals bg"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buf.as_entire_binding(),
            }],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sprite sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 单位四边形（角点 {0,1}²），实例属性提供位置/尺寸/锚点
        let quad_vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad"),
            contents: bytemuck::cast_slice(&[0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let quad_ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("quad index"),
            contents: bytemuck::cast_slice(&[0u16, 1, 2, 2, 1, 3]),
            usage: wgpu::BufferUsages::INDEX,
        });

        let instance_cap = 1024;
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("instances"),
            size: instance_cap * std::mem::size_of::<GpuInstance>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 1×1 白纹理:子矩形模式的区域清屏四边形需要绑定一个纹理
        let dummy = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("clear dummy"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            dummy.as_image_copy(),
            &[255, 255, 255, 255],
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4), rows_per_image: Some(1) },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let dummy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear dummy bg"),
            layout: &tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&dummy.create_view(&Default::default())) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        Renderer {
            device: device.clone(),
            queue: queue.clone(),
            shader,
            globals_layout,
            tex_layout,
            globals_buf,
            globals_bg,
            sampler,
            quad_vb,
            quad_ib,
            pipelines: HashMap::new(),
            instance_buf,
            instance_cap,
            textures: HashMap::new(),
            texture_bytes: 0,
            max_gpu_bytes: usize::MAX,
            frame: 0,
            last_used: HashMap::new(),
            dummy_bg,
            upscale: None,
            upscale_target: (0, 0),
            video_staging: None,
            video_upscale_gen_seen: u64::MAX,
            video_upscale_logged: false,
        }
    }

    /// 视频通道超分开关与目标尺寸(`target` = SB 合成槽分辨率:放大后
    /// 的视频纹理以该分辨率进入合成,替代合成器内部的线性拉伸)。
    /// Off 时恢复原帧直传。模式/目标变更即时生效,无需重载。
    pub fn set_video_upscale(
        &mut self,
        mode: crate::render::upscale::UpscaleMode,
        target: (u32, u32),
    ) {
        use crate::render::upscale::{UpscaleMode, Upscaler};
        self.upscale_target = target;
        self.video_upscale_logged = false;
        // 新 Upscaler 的代数从 0 重新计:不复位 seen 的话,旧值恰好相等时
        // 会被误判"输出纹理未变"而跳过重绑 —— 视频槽位继续采样已被
        // 丢弃的旧输出纹理(不再更新),画面冻在最后一帧
        self.video_upscale_gen_seen = u64::MAX;
        match mode {
            UpscaleMode::Off => self.upscale = None,
            m => {
                if self.upscale.as_ref().is_none_or(|u| u.mode() != m) {
                    self.upscale = Some(Upscaler::new(&self.device, &self.queue, m));
                }
            }
        }
        log::info!("[video-upscale] 配置: {:?} → 槽位 {}x{}", mode, target.0, target.1);
    }

    /// GPU 贴图内存预算(字节):超出后 LRU 淘汰未在本帧使用的贴图槽,
    /// 下次用到时重新上传。嵌入式宿主(手机等内存受限环境)应设置;
    /// 独立播放器默认不限。
    pub fn set_gpu_budget(&mut self, bytes: usize) {
        self.max_gpu_bytes = bytes;
    }

    /// GPU 贴图是否受预算约束(存在 LRU 淘汰)。淘汰可能发生时上传后
    /// 必须保留 CPU 解码副本:被淘汰贴图回归时从缓存重传即可,若副本
    /// 已弃则要同步重走整张 PNG 解码,回归集中的一帧会明显卡顿。
    pub fn gpu_budgeted(&self) -> bool {
        self.max_gpu_bytes != usize::MAX
    }

    /// GPU 预算(字节;不限 = `usize::MAX`)。
    pub fn gpu_budget(&self) -> usize {
        self.max_gpu_bytes
    }

    /// 已驻留 GPU 的贴图字节合计(解码 RGBA 口径)。
    pub fn texture_bytes(&self) -> usize {
        self.texture_bytes
    }

    pub fn upload_texture(&mut self, key: &str, img: &RgbaImage) {
        if self.textures.contains_key(key) {
            return;
        }
        // 预乘 alpha:RGB × A 写入纹理。透明像素 RGB=0,双线性插值
        // 不会把黑色混入可见边缘 —— 旋转/缩放精灵的黑边/黑块根因。
        let mut premult = img.clone();
        for px in premult.pixels_mut() {
            let a = px[3] as u32;
            if a == 0 {
                px[0] = 0; px[1] = 0; px[2] = 0;
            } else if a < 255 {
                px[0] = ((px[0] as u32 * a + 127) / 255) as u8;
                px[1] = ((px[1] as u32 * a + 127) / 255) as u8;
                px[2] = ((px[2] as u32 * a + 127) / 255) as u8;
            }
        }
        let (w, h) = premult.dimensions();
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(key),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        write_rgba(&self.queue, &texture, w, h, premult.as_raw());
        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &self.tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let bytes = (w * h * 4) as usize;
        self.texture_bytes += bytes;
        self.textures.insert(
            key.to_string(),
            TextureSlot { texture, bind_group, size: [w, h], bytes },
        );
    }

    /// 写入/更新一个外部逐帧纹理(视频通道):尺寸不变时原地
    /// write_texture,变化时重建纹理与绑定组。内容为 RGBA 行主序。
    /// 返回是否新建了纹理(首帧)。不走 LRU 记账——视频纹理常驻,
    /// 只有一帧的量。开启超分时:原帧先上常驻 staging,跑 FSR/Anime4K
    /// 链放大到 `upscale_target`,再以 GPU 纹理换入视频槽位。
    pub fn write_frame(&mut self, key: &str, w: u32, h: u32, rgba: &[u8]) -> bool {
        debug_assert_eq!(rgba.len(), (w * h * 4) as usize);
        if self.upscale.is_some() {
            let slot = self.upscale_target;
            // 目标保持源宽高比(宽向贴合槽位):非 16:9 视频不被拉伸,
            // 精灵变换按原几何映射,纹理只是更精细
            let target = if w > 0 && h > 0 {
                (slot.0, ((slot.0 as f64 * h as f64 / w as f64).round() as u32).max(1))
            } else {
                slot
            };
            // 源尺寸 = 目标尺寸也执行:Anime4K 的重建/去噪与 FSR 的 RCAS
            // 锐化在 1:1 下正是主要收益(曾经此处跳过同尺寸,表现为
            // "开了超分没效果")
            if target.0 > 0 && target.1 > 0 && w > 0 && h > 0 {
                if !self.video_upscale_logged {
                    self.video_upscale_logged = true;
                    log::info!(
                        "[video-upscale] {:?}: {}x{} → {}x{}",
                        self.upscale.as_ref().map(|u| u.mode()),
                        w, h, target.0, target.1
                    );
                }
                // 常驻 staging(Anime4K 执行器绑定源纹理,必须同一张)
                if self.video_staging.as_ref().is_none_or(|t| t.width() != w || t.height() != h) {
                    self.video_staging = Some(self.device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("video upscale staging"),
                        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    }));
                }
                let staging = self.video_staging.as_ref().unwrap();
                write_rgba(&self.queue, staging, w, h, rgba);
                let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("video upscale"),
                });
                let (out, gen) = {
                    let up = self.upscale.as_mut().unwrap();
                    (up.process(&mut encoder, staging, target), up.generation())
                };
                self.queue.submit([encoder.finish()]);
                // 输出纹理按尺寸复用:代数未变 = 槽位仍绑同一张,逐帧
                // 零重绑(compute 原地更新内容)
                if gen == self.video_upscale_gen_seen && self.textures.contains_key(key) {
                    return false;
                }
                self.video_upscale_gen_seen = gen;
                return self.set_frame_texture(key, out, target.0, target.1);
            }
        }
        if let Some(slot) = self.textures.get(key) {
            if slot.size == [w, h] {
                write_rgba(&self.queue, &slot.texture, w, h, rgba);
                return false;
            }
            let old = self.textures.remove(key).unwrap();
            self.texture_bytes = self.texture_bytes.saturating_sub(old.bytes);
            self.last_used.remove(key);
        }
        let img = image::RgbaImage::from_raw(w, h, rgba.to_vec()).expect("frame size mismatch");
        // 视频逐帧更新:每帧 CPU BC 编码代价过高,保持 RGBA 直传
        self.upload_texture(key, &img);
        true
    }

    /// 注册一张宿主提供的 GPU 纹理为逐帧视频通道(零拷贝路径:宿主经
    /// AHardwareBuffer 导入的解码帧)。替换既有槽位;纹理不进 LRU/字节
    /// 记账,旧槽位随 wgpu 生命周期销毁。返回是否替换了已有槽。
    pub fn set_frame_texture(&mut self, key: &str, texture: wgpu::Texture, w: u32, h: u32) -> bool {
        let bytes = (w * h * 4) as usize;
        let replaced = if let Some(old) = self.textures.remove(key) {
            self.texture_bytes = self.texture_bytes.saturating_sub(old.bytes);
            self.last_used.remove(key);
            true
        } else {
            false
        };
        self.texture_bytes += bytes;
        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &self.tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.textures.insert(
            key.to_string(),
            TextureSlot { texture, bind_group, size: [w, h], bytes: (w * h * 4) as usize },
        );
        replaced
    }


    pub fn texture(&self, key: &str) -> Option<&TextureSlot> {
        self.textures.get(key)
    }

    fn ensure_pipelines(&mut self, format: wgpu::TextureFormat) {
        if self.pipelines.contains_key(&format) {
            return;
        }
        {
            let layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("sprite layout"),
                bind_group_layouts: &[&self.globals_layout, &self.tex_layout],
                push_constant_ranges: &[],
            });
            let vertex_buffers = [
                wgpu::VertexBufferLayout {
                    array_stride: 8,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<GpuInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 1 },  // pos
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 2 },   // size
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 3 },  // anchor
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 24, shader_location: 4 },    // rotation
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 28, shader_location: 5 },  // color
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 44, shader_location: 6 },  // flip
                    ],
                },
            ];
            let make = |blend: wgpu::BlendState| {
                self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("sprite pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &self.shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &vertex_buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &self.shader,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                })
            };
            // 预乘 alpha 混合:贴图上传时已 RGB×A,One/OneMinusSrcAlpha
            // 数学等价于直线 alpha 的 SrcAlpha/OneMinusSrcAlpha,但双线性
            // 插值在透明边界不再产生黑色边沿
            let normal = make(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
            });
            // osu-framework `BlendingParameters.Additive` 是 RGB
            // (SrcAlpha, One)。精灵输出已预乘,RGB 用 One/One 等价。
            // Alpha 不累加(Zero/One):槽位从透明清屏,之后用预乘 over
            // 贴回屏幕。Alpha 也 One/One 时,重叠的加色精灵会把槽位
            // 撑成不透明,贴回时盖住背景而不是往背景上加光,背景亮度
            // 就再也透不过来(MariannE 4:19 的一排 damnaestar)。
            let additive = make(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            });
            // REPLACE(直写,无混合):区域清屏四边形用
            let clear = make(wgpu::BlendState::REPLACE);
            self.pipelines.insert(format, Pipelines { normal, additive, clear });
        }
    }

    /// 渲染一帧。draws 顺序即绘制顺序（画家算法）。widescreen=false 时固定 4:3（黑边）。
    /// `clear` 为 RGBA 清屏色(合成到宿主场景时传透明 `[0,0,0,0]`,
    /// 4:3 黑边区域保持透明而不是黑)。
    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        widescreen: bool,
        draws: &[Draw],
        clear: [f64; 4],
    ) {
        self.render_subrect(target, format, width, height, widescreen, draws, clear, None);
    }

    /// [`render`] 的子矩形版本:`subrect = Some((x,y,w,h))` 时把整个
    /// 640×480 场景映射到目标纹理的该矩形(嵌入宿主直接渲进图集槽位,
    /// 替代"独立纹理 + copy_into_atlas"的双份显存与每帧拷贝)。
    ///
    /// 子矩形模式不能整附件 Clear——LoadOp::Clear 会清掉图集其余内容
    /// (字体/皮肤/背景),改用 Load + REPLACE 混合的巨型透明四边形做
    /// 区域清屏;viewport 把 NDC 裁进矩形,投影矩阵与全幅模式一致,
    /// 渲染结果与"全幅渲染后拷入矩形"逐像素等价。
    pub fn render_subrect(
        &mut self,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        widescreen: bool,
        draws: &[Draw],
        clear: [f64; 4],
        subrect: Option<(u32, u32, u32, u32)>,
    ) {
        self.frame += 1;
        // LRU 记账仅在真有预算限制时进行:无限制(独立播放器)时,
        // 每个 draw 一次 String clone + HashMap insert 是纯浪费
        // (world.execute(me) 峰值 3.5K draw/帧)。
        if self.max_gpu_bytes != usize::MAX {
            for d in draws {
                self.last_used.insert(d.texture.clone(), self.frame);
            }
        }
        let globals = Globals { mvp: ortho_640x480(width, height) };
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        // 只保留贴图已就绪的 draw，保证实例索引对齐
        let mut instances: Vec<GpuInstance> = draws
            .iter()
            .filter(|d| self.textures.contains_key(&d.texture))
            .map(|d| d.instance)
            .collect();
        if std::env::var("SB_DEBUG_PASS").is_ok() {
            eprintln!(
                "[sb-pass] subrect={subrect:?} draws={} included={} tex_bytes={} textures={}",
                draws.len(),
                instances.len(),
                self.texture_bytes,
                self.textures.len()
            );
        }

        // 子矩形模式:实例 0 为区域清屏四边形(覆盖全视口的透明 REPLACE)
        if subrect.is_some() {
            instances.insert(
                0,
                GpuInstance {
                    pos: [320.0, 240.0],
                    size: [100_000.0, 100_000.0], // 远超任何视口,出界部分被裁剪
                    anchor: [0.5, 0.5],
                    rotation: 0.0,
                    color: [0.0, 0.0, 0.0, 0.0], // 颜色 0 → 输出恒为 (0,0,0,0)
                    flip: [0.0, 0.0],
                    _pad: [0.0; 3],
                },
            );
        }

        let needed = instances.len() as u64;
        if needed > self.instance_cap {
            self.instance_cap = needed.next_power_of_two();
            self.instance_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("instances"),
                size: self.instance_cap * std::mem::size_of::<GpuInstance>() as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        if !instances.is_empty() {
            self.queue.write_buffer(&self.instance_buf, 0, bytemuck::cast_slice(&instances));
        }

        self.ensure_pipelines(format);
        let pipelines = self.pipelines.get(&format).expect("管线已创建");
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("storyboard pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // 子矩形模式必须保留图集其余内容,清屏交给 REPLACE 四边形
                        load: if subrect.is_some() { wgpu::LoadOp::Load } else { wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0],
                            g: clear[1],
                            b: clear[2],
                            a: clear[3],
                        }) },
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            if let Some((x, y, w, h)) = subrect {
                pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
            }
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_vertex_buffer(0, self.quad_vb.slice(..));
            // 绑定整个缓冲而非按实例数截断：实例数为 0 时空切片会让 wgpu panic
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));
            pass.set_index_buffer(self.quad_ib.slice(..), wgpu::IndexFormat::Uint16);

            let mut cur_additive: Option<bool> = None;
            let mut i: u32 = 0;
            if subrect.is_some() {
                pass.set_pipeline(&pipelines.clear);
                pass.set_bind_group(1, &self.dummy_bg, &[]);
                pass.draw_indexed(0..6, 0, 0..1);
                i = 1;
            }
            // 非宽屏故事板容器遮罩(lazer StoryboardLayer.Masking):精灵
            // 一律裁剪到居中的 640×480 容器,容器外的屏幕区域保持透明,
            // 由宿主透出谱面背景。清除四边形在遮罩之前执行(整槽清透)。
            if !widescreen {
                let (vx, vy, vw, vh) =
                    subrect.map_or((0, 0, width, height), |(x, y, w, h)| (x, y, w, h));
                let (mx, my, mw, mh) = container_mask_rect(vx, vy, vw, vh);
                pass.set_scissor_rect(mx, my, mw, mh);
            }
            // 连续同纹理 + 同混合模式的 draw 合并为一次实例化绘制:
            // 绘制顺序、实例数据、管线切换时机与逐精灵绘制完全一致,
            // 只是省掉重复的 set_bind_group/draw_indexed(wgpu 每次调用
            // 都有验证与驱动开销;world.execute(me) 类生成式 SB 峰值
            // 3505 个精灵仅 6 个 run)。
            let mut idx = 0;
            while idx < draws.len() {
                let d = &draws[idx];
                let Some(slot) = self.textures.get(&d.texture) else {
                    idx += 1;
                    continue;
                };
                let additive = d.additive;
                if cur_additive != Some(additive) {
                    let pipe = if additive { &pipelines.additive } else { &pipelines.normal };
                    pass.set_pipeline(pipe);
                    cur_additive = Some(additive);
                }
                pass.set_bind_group(1, &slot.bind_group, &[]);
                let run_start = i;
                while idx < draws.len()
                    && draws[idx].additive == additive
                    && draws[idx].texture == d.texture
                {
                    idx += 1;
                    i += 1;
                }
                pass.draw_indexed(0..6, 0, run_start..i);
            }
        }
        self.queue.submit(Some(encoder.finish()));

        // GPU 贴图超预算:淘汰最久未用(且不在本帧 draw 里)的槽位。
        // 提交后再删,保证本帧已绑定的 bind group 不会失效。
        if self.texture_bytes > self.max_gpu_bytes {
            let mut candidates: Vec<(String, u64, usize)> = self
                .textures
                .iter()
                .filter(|(k, _)| self.last_used.get(*k).copied().unwrap_or(0) < self.frame)
                .map(|(k, slot)| (k.clone(), self.last_used.get(k).copied().unwrap_or(0), slot.bytes))
                .collect();
            candidates.sort_by_key(|(_, used, _)| *used);
            for (key, _, bytes) in candidates {
                if self.texture_bytes <= self.max_gpu_bytes {
                    break;
                }
                self.textures.remove(&key);
                self.last_used.remove(&key);
                self.texture_bytes -= bytes;
            }
        }
    }
}

/// Queue 一个 RGBA 行主序上传。wgpu 要求 bytes_per_row 对齐 256:
/// 宽度非 64 倍数时按对齐行距重排(视频与精灵宽度都可能任意)。
fn write_rgba(queue: &wgpu::Queue, texture: &wgpu::Texture, w: u32, h: u32, rgba: &[u8]) {
    let tight = w * 4;
    let dst = wgpu::TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: wgpu::Origin3d::ZERO,
        aspect: wgpu::TextureAspect::All,
    };
    let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
    if tight % wgpu::COPY_BYTES_PER_ROW_ALIGNMENT == 0 {
        queue.write_texture(
            dst,
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(tight), rows_per_image: Some(h) },
            size,
        );
        return;
    }
    let bpr = (tight + wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1) & !(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT - 1);
    let mut padded = vec![0u8; (bpr * h) as usize];
    for row in 0..h as usize {
        let src = row * tight as usize;
        let d = row * bpr as usize;
        padded[d..d + tight as usize].copy_from_slice(&rgba[src..src + tight as usize]);
    }
    queue.write_texture(
        dst,
        &padded,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(bpr), rows_per_image: Some(h) },
        size,
    );
}

/// 640x480 正交投影：高固定 480,宽按**目标纵横比**向两侧扩展(始终以
/// [320±240×aspect] 为可视世界窗)。宽屏与否不再改变投影——lazer
/// `DrawableStoryboard` 的容器两种都是同一缩放(ScreenH/480)居中,宽屏
/// 标志只决定容器宽度(853 或 640 单位)进而决定 Masking 裁剪范围,
/// 由 render_subrect 的 scissor 实现。此前非宽屏把 640 单位拉满整屏宽,
/// 造成 4:3 故事板被水平拉宽。
fn ortho_640x480(width: u32, height: u32) -> [f32; 16] {
    let aspect = if height == 0 { 4.0 / 3.0 } else { width as f32 / height as f32 };
    let view_w = OSU_HEIGHT * aspect;
    let x0 = OSU_CENTRE_X - view_w / 2.0;
    let x1 = OSU_CENTRE_X + view_w / 2.0;
    let sx = 2.0 / (x1 - x0);
    let sy = -2.0 / OSU_HEIGHT;
    let tx = -(x1 + x0) / (x1 - x0);
    let ty = 1.0;
    // 列主序
    [
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, -1.0, 0.0,
        tx, ty, 0.0, 1.0,
    ]
}

/// 非宽屏故事板的容器遮罩(中央 4:3)在视口内的像素矩形。lazer
/// `StoryboardLayer.Masking = true`:精灵一律裁剪到容器内(宽屏容器
/// 853×480、非宽屏 640×480,均居中;前者在 16:9 视口下恰好全覆盖,
/// 故只需对非宽屏裁剪)。返回 (x, y, w, h)。
fn container_mask_rect(vx: u32, vy: u32, vw: u32, vh: u32) -> (u32, u32, u32, u32) {
    let aspect = vw as f32 / vh.max(1) as f32;
    let frac = (4.0 / 3.0 / aspect).min(1.0);
    let mw = (vw as f32 * frac).floor() as u32;
    let mx = vx + (vw - mw) / 2;
    (mx, vy, mw, vh)
}

/// 求值时刻 t 的全部可见精灵并填充贴图，返回按绘制顺序排列的 Draw 列表。
pub fn build_draws(
    renderer: &mut Renderer,
    assets: &mut Assets,
    sb: &CompiledStoryboard,
    t: f32,
    fail: FailState,
) -> Vec<Draw> {
    build_draws_filtered(renderer, assets, sb, t, fail, 1.0, |_| true)
}

/// [`build_draws`] 的层过滤版本:嵌入宿主把 storyboard 拆到游戏画面上下
/// 两侧时用(osu! 层序:Background/Fail/Pass 在游戏区之下,Foreground/
/// Overlay 在其上)。`include` 对每个元素的层返回是否参与本次求值。
/// 预取 storyboard 贴图:按元素起播时刻排序,把引用的贴图(动画展开
/// 全部帧)解码并上传,直到 GPU 预算或 `deadline`。惰性加载下"整批
/// 贴图首次可见"的那一帧要同步走完数百次解码+上传——帧动画式 SB
/// (单拍激活几百张新贴图)首播卡一下、回看不卡的根因;预取后首播与
/// 回看一致。超预算/超时未取到的贴图保持运行期惰性加载,行为不变。
/// 返回本次实际上传的张数。
pub fn prefetch_textures(
    renderer: &mut Renderer,
    assets: &mut Assets,
    sb: &CompiledStoryboard,
    deadline: Option<std::time::Instant>,
) -> usize {
    // (起播时刻, 逻辑路径):时间序保证时限内先备好最早登场的内容
    let mut wanted: Vec<(f32, String)> = Vec::new();
    for el in &sb.elements {
        match &el.animation {
            None => wanted.push((el.start, normalize_path(&el.path))),
            Some(a) => {
                for i in 0..a.frame_count.max(1) {
                    wanted.push((el.start, normalize_path(&frame_path(&el.path, i as usize))));
                }
            }
        }
    }
    wanted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let budget = renderer.gpu_budget();
    let mut uploaded = 0;
    for (_, key) in &wanted {
        if renderer.texture(&key).is_some() {
            continue;
        }
        if renderer.texture_bytes() >= budget {
            break;
        }
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            break;
        }
        if let Some(img) = assets.get(key) {
            renderer.upload_texture(key, img);
            // 与 build_draws_filtered 同策略:无淘汰可能时 CPU 副本纯浪费
            if !renderer.gpu_budgeted() {
                assets.discard(key);
            }
            uploaded += 1;
        }
    }
    uploaded
}

/// [`build_draws`] 的层过滤版本:嵌入宿主把 storyboard 拆到游戏画面上下
/// 两侧时用(osu! 层序:Background/Fail/Pass 在游戏区之下,Foreground/
/// Overlay 在其上)。`include` 对每个元素的层返回是否参与本次求值。
pub fn build_draws_filtered(
    renderer: &mut Renderer,
    assets: &mut Assets,
    sb: &CompiledStoryboard,
    t: f32,
    fail: FailState,
    dim: f32,
    include: impl Fn(&Layer) -> bool,
) -> Vec<Draw> {
    let mut out = Vec::new();
    for el in &sb.elements {
        if !include(&el.layer) {
            continue;
        }
        if el.layer == Layer::Fail && fail == FailState::Pass {
            continue;
        }
        if el.layer == Layer::Pass && fail == FailState::Fail {
            continue;
        }
        let Some(st) = el.state_at(t) else { continue };
        if st.alpha <= 0.002 {
            continue;
        }
        let logical = match &el.animation {
            None => el.path.clone(),
            Some(_) => frame_path(&el.path, el.frame_at(t)),
        };
        let key = normalize_path(&logical);
        if renderer.texture(&key).is_none() {
            if let Some(img) = assets.get(&key) {
                renderer.upload_texture(&key, img);
                // 无预算限制时 GPU 永不淘汰,CPU 副本是纯双驻留,上传后
                // 即弃;有预算时保留副本,LRU 淘汰后的回归从缓存重传,
                // 避免整批同步重解码造成的单帧卡顿(CPU 侧另有
                // SB_CACHE_MB 预算管内存)。
                if !renderer.gpu_budgeted() {
                    assets.discard(&key);
                }
            } else {
                continue; // 缺贴图：跳过该精灵
            }
        }
        let Some(slot) = renderer.texture(&key) else { continue };
        let [tw, th] = slot.size;
        // 暗度统一生效:lazer 的 Overlay 层虽被代理到物件上方,但其
        // 绘制继承 dimContent 的 FadeColour(Gray(1-dim)),同样衰减。
        let dim = dim;
        if std::env::var("SB_DEBUG_DRAWS").is_ok() {
            eprintln!(
                "[sb-draw] {key} pos=({:.0},{:.0}) size=({:.0}x{:.0}) rot={:.2} color=({:.2},{:.2},{:.2},{:.2}) add={} dim={:.2}",
                st.x, st.y, tw as f32 * st.scale_x, th as f32 * st.scale_y, st.rotation,
                st.colour[0], st.colour[1], st.colour[2], st.alpha, st.additive, dim
            );
        }
        let mut inst = GpuInstance {
            pos: [st.x, st.y],
            size: [tw as f32 * st.scale_x, th as f32 * st.scale_y],
            anchor: el.origin.anchor(),
            rotation: st.rotation,
            color: [st.colour[0], st.colour[1], st.colour[2], st.alpha],
            flip: [st.flip_h as u32 as f32, st.flip_v as u32 as f32],
            _pad: [0.0; 3],
        };
        inst.color[0] *= dim;
        inst.color[1] *= dim;
        inst.color[2] *= dim;
        out.push(Draw {
            texture: key,
            additive: st.additive,
            dim,
            instance: inst,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ortho_maps_corners() {
        // 4:3 → x 覆盖 0..640，y 覆盖 0..480
        let m = ortho_640x480(800, 600);
        let apply = |m: &[f32; 16], x: f32, y: f32| {
            // 列主序矩阵乘法
            let cx = m[0] * x + m[4] * y + m[12];
            let cy = m[1] * x + m[5] * y + m[13];
            [cx, cy]
        };
        let [x, y] = apply(&m, 0.0, 0.0);
        assert!((x + 1.0).abs() < 1e-4 && (y - 1.0).abs() < 1e-4);
        let [x, y] = apply(&m, 640.0, 480.0);
        assert!((x - 1.0).abs() < 1e-4 && (y + 1.0).abs() < 1e-4);
        let [x, _] = apply(&m, 320.0, 0.0);
        assert!(x.abs() < 1e-4);

        // 16:9 → 可见宽度 853.33px，640px 位于 (640-(-106.67))/853.33*2-1 = 0.75
        let m = ortho_640x480(1600, 900);
        let [x, _] = apply(&m, 640.0, 240.0);
        assert!((x - 0.75).abs() < 1e-3, "16:9 下 640px 应映射到 0.75: {x}");
        let [x, _] = apply(&m, 320.0, 240.0);
        assert!(x.abs() < 1e-3, "中心仍在原点: {x}");
        // 投影不再区分宽屏:0..640 不再铺满 16:9 全宽,而是居中的 75%
        let [x, _] = apply(&m, 0.0, 240.0);
        assert!((x + 0.75).abs() < 1e-3, "x=0 应映射到 -0.75(中央 4:3 左缘): {x}");
    }

    #[test]
    fn mask_is_central_43_on_widescreen_viewport() {
        // 16:9 视口:容器占中央 75% 宽
        let (mx, my, mw, mh) = container_mask_rect(0, 0, 1920, 1080);
        assert_eq!((mx, my, mw, mh), (240, 0, 1440, 1080));
        // 4:3 视口:容器铺满,遮罩为全视口
        assert_eq!(container_mask_rect(0, 0, 1440, 1080), (0, 0, 1440, 1080));
        // 带偏移的子矩形
        let (mx, my, mw, mh) = container_mask_rect(100, 50, 1920, 1080);
        assert_eq!((mx, my, mw, mh), (340, 50, 1440, 1080));
    }
}
