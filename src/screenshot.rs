//! 无窗口渲染：离屏渲染一帧并保存为 PNG（不依赖显示服务器）。

use osu_storyboard_render::osb::timeline::{CompiledStoryboard, FailState};
use osu_storyboard_render::render::gpu::GpuContext;
use osu_storyboard_render::render::renderer::{build_draws, Renderer};
use osu_storyboard_render::render::texture::Assets;
use anyhow::Result;
use image::RgbaImage;
use std::sync::mpsc;

pub fn render_screenshot(
    sb: &CompiledStoryboard,
    assets: &mut Assets,
    time: f32,
    width: u32,
    height: u32,
    fail: FailState,
    out: &std::path::Path,
) -> Result<()> {
    let ctx = GpuContext::new(None)?;
    let mut renderer = Renderer::new(&ctx.device, &ctx.queue);
    let draws = build_draws(&mut renderer, assets, sb, time, fail);
    log::info!("时刻 {time}ms 可见精灵: {}", draws.len());

    let extent = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    renderer.render(&view, wgpu::TextureFormat::Rgba8Unorm, width, height, sb.widescreen, &draws, [0.0, 0.0, 0.0, 1.0]);

    // 回读到 CPU
    let bytes_per_row = (width * 4).next_multiple_of(256);
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot readback"),
        size: bytes_per_row as u64 * height as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture: &target, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        extent,
    );
    ctx.queue.submit(Some(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    ctx.wait();
    rx.recv()
        .map_err(|_| anyhow::anyhow!("GPU 回读失败"))?
        .map_err(|e| anyhow::anyhow!("映射缓冲区失败: {e:?}"))?;

    let data = slice.get_mapped_range();
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height as usize {
        let src = &data[y * bytes_per_row as usize..y * bytes_per_row as usize + width as usize * 4];
        pixels.extend_from_slice(src);
    }
    drop(data);
    buffer.unmap();
    let img = RgbaImage::from_raw(width, height, pixels).expect("回读尺寸不匹配");

    img.save(out)?;
    log::info!("已保存截图 {} ({}x{})", out.display(), width, height);
    Ok(())
}
