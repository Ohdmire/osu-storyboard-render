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
    pub instance: GpuInstance,
}

struct Pipelines {
    normal: wgpu::RenderPipeline,
    additive: wgpu::RenderPipeline,
}

pub struct TextureSlot {
    /// 原始纹理句柄:视频等逐帧更新的外部纹理需要重复 write_texture。
    pub texture: wgpu::Texture,
    pub bind_group: wgpu::BindGroup,
    pub size: [u32; 2],
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
        }
    }

    /// GPU 贴图内存预算(字节):超出后 LRU 淘汰未在本帧使用的贴图槽,
    /// 下次用到时重新上传。嵌入式宿主(手机等内存受限环境)应设置;
    /// 独立播放器默认不限。
    pub fn set_gpu_budget(&mut self, bytes: usize) {
        self.max_gpu_bytes = bytes;
    }

    pub fn upload_texture(&mut self, key: &str, img: &RgbaImage) {
        if self.textures.contains_key(key) {
            return;
        }
        let (w, h) = img.dimensions();
        let size = wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(key),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        write_rgba(&self.queue, &texture, w, h, img.as_raw());
        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &self.tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.texture_bytes += (w * h * 4) as usize;
        self.textures.insert(key.to_string(), TextureSlot { texture, bind_group, size: [w, h] });
    }

    /// 写入/更新一个外部逐帧纹理(视频通道):尺寸不变时原地
    /// write_texture,变化时重建纹理与绑定组。内容为 RGBA 行主序。
    /// 返回是否新建了纹理(首帧)。不走 LRU 记账——视频纹理常驻,
    /// 只有一帧的量。
    pub fn write_frame(&mut self, key: &str, w: u32, h: u32, rgba: &[u8]) -> bool {
        debug_assert_eq!(rgba.len(), (w * h * 4) as usize);
        if let Some(slot) = self.textures.get(key) {
            if slot.size == [w, h] {
                write_rgba(&self.queue, &slot.texture, w, h, rgba);
                return false;
            }
            let old = self.textures.remove(key).unwrap();
            self.texture_bytes -= (old.size[0] * old.size[1] * 4) as usize;
            self.last_used.remove(key);
        }
        let img = image::RgbaImage::from_raw(w, h, rgba.to_vec()).expect("frame size mismatch");
        self.upload_texture(key, &img);
        true
    }

    /// 注册一张宿主提供的 GPU 纹理为逐帧视频通道(零拷贝路径:宿主经
    /// AHardwareBuffer 导入的解码帧)。替换既有槽位;纹理不进 LRU/字节
    /// 记账,旧槽位随 wgpu 生命周期销毁。返回是否替换了已有槽。
    pub fn set_frame_texture(&mut self, key: &str, texture: wgpu::Texture, w: u32, h: u32) -> bool {
        let replaced = self.textures.remove(key).is_some();
        let view = texture.create_view(&Default::default());
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &self.tex_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.textures.insert(key.to_string(), TextureSlot { texture, bind_group, size: [w, h] });
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
            let normal = make(wgpu::BlendState::ALPHA_BLENDING);
            let additive = make(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            });
            self.pipelines.insert(format, Pipelines { normal, additive });
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
        self.frame += 1;
        for d in draws {
            self.last_used.insert(d.texture.clone(), self.frame);
        }
        let globals = Globals { mvp: ortho_640x480(width, height, widescreen) };
        self.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        // 只保留贴图已就绪的 draw，保证实例索引对齐
        let instances: Vec<GpuInstance> = draws
            .iter()
            .filter(|d| self.textures.contains_key(&d.texture))
            .map(|d| d.instance)
            .collect();

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
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: clear[0],
                            g: clear[1],
                            b: clear[2],
                            a: clear[3],
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.globals_bg, &[]);
            pass.set_vertex_buffer(0, self.quad_vb.slice(..));
            // 绑定整个缓冲而非按实例数截断：实例数为 0 时空切片会让 wgpu panic
            pass.set_vertex_buffer(1, self.instance_buf.slice(..));
            pass.set_index_buffer(self.quad_ib.slice(..), wgpu::IndexFormat::Uint16);

            let mut cur_additive: Option<bool> = None;
            let mut i: u32 = 0;
            for d in draws {
                let Some(slot) = self.textures.get(&d.texture) else { continue };
                if cur_additive != Some(d.additive) {
                    let pipe = if d.additive { &pipelines.additive } else { &pipelines.normal };
                    pass.set_pipeline(pipe);
                    cur_additive = Some(d.additive);
                }
                pass.set_bind_group(1, &slot.bind_group, &[]);
                pass.draw_indexed(0..6, 0, i..i + 1);
                i += 1;
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
                .map(|(k, slot)| (k.clone(), self.last_used.get(k).copied().unwrap_or(0), (slot.size[0] * slot.size[1] * 4) as usize))
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

/// 640x480 正交投影：高固定 480；widescreen 时宽按目标纵横比向两侧扩展，
/// 否则固定 4:3 宽度（宽窗口两侧黑边，与 osu! 非宽屏谱面行为一致）。
fn ortho_640x480(width: u32, height: u32, widescreen: bool) -> [f32; 16] {
    let aspect = if height == 0 { 4.0 / 3.0 } else { width as f32 / height as f32 };
    let view_w = if widescreen { OSU_HEIGHT * aspect } else { OSU_HEIGHT * 4.0 / 3.0 };
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

/// 求值时刻 t 的全部可见精灵并填充贴图，返回按绘制顺序排列的 Draw 列表。
pub fn build_draws(
    renderer: &mut Renderer,
    assets: &mut Assets,
    sb: &CompiledStoryboard,
    t: f32,
    fail: FailState,
) -> Vec<Draw> {
    build_draws_filtered(renderer, assets, sb, t, fail, |_| true)
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
            } else {
                continue; // 缺贴图：跳过该精灵
            }
        }
        let Some(slot) = renderer.texture(&key) else { continue };
        let [tw, th] = slot.size;
        out.push(Draw {
            texture: key,
            additive: st.additive,
            instance: GpuInstance {
                pos: [st.x, st.y],
                size: [tw as f32 * st.scale_x, th as f32 * st.scale_y],
                anchor: el.origin.anchor(),
                rotation: st.rotation,
                color: [st.colour[0], st.colour[1], st.colour[2], st.alpha],
                flip: [st.flip_h as u32 as f32, st.flip_v as u32 as f32],
                _pad: [0.0; 3],
            },
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
        let m = ortho_640x480(800, 600, true);
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
        let m = ortho_640x480(1600, 900, true);
        let [x, _] = apply(&m, 640.0, 240.0);
        assert!((x - 0.75).abs() < 1e-3, "widescreen 下 640px 应映射到 0.75: {x}");
        let [x, _] = apply(&m, 320.0, 240.0);
        assert!(x.abs() < 1e-3, "中心仍在原点: {x}");

        // 非宽屏：16:9 窗口下 0..640 仍铺满整个宽度（4:3 区域居中，两侧黑边）
        let m = ortho_640x480(1600, 900, false);
        let [x, _] = apply(&m, 0.0, 240.0);
        assert!((x + 1.0).abs() < 1e-3, "x=0 仍在左缘: {x}");
        let [x, _] = apply(&m, 640.0, 240.0);
        assert!((x - 1.0).abs() < 1e-3, "x=640 仍在右缘: {x}");
    }
}
