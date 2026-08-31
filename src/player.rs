//! winit 窗口播放器：实时回放 storyboard，支持暂停/快进/调速。

use crate::osb::timeline::{CompiledStoryboard, FailState};
use crate::render::gpu::GpuContext;
use crate::render::renderer::{build_draws, Renderer};
use crate::render::texture::Assets;
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, Modifiers, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes};

pub struct PlayerConfig {
    pub width: u32,
    pub height: u32,
    pub start: f32,
    pub speed: f32,
    pub fail: FailState,
}

struct Clock {
    t: f32,
    playing: bool,
    speed: f32,
    last: Option<Instant>,
}

impl Clock {
    fn step(&mut self, end: f32) {
        let now = Instant::now();
        if let Some(prev) = self.last.replace(now) {
            let dt = now.duration_since(prev).as_millis().min(250) as f32;
            if self.playing {
                self.t += dt * self.speed;
                if self.t > end {
                    self.t = end;
                    self.playing = false;
                    log::info!("播放结束（ {:.1}s ），按 R 重新播放", end / 1000.0);
                }
            }
        }
    }
}

pub fn run(sb: CompiledStoryboard, assets: Assets, cfg: PlayerConfig) -> Result<()> {
    println!(
        "控制: Space 暂停/继续 · ←/→ 快退/快进 1s (Shift: 5s) · R 重播 · -/= 调速 · Esc 退出"
    );
    let event_loop = EventLoop::new()?;
    let mut app = App {
        sb: Some(sb),
        assets: Some(assets),
        cfg,
        window: None,
        surface: None,
        ctx: None,
        renderer: None,
        size: (0, 0),
        format: wgpu::TextureFormat::Bgra8Unorm,
        clock: Clock { t: 0.0, playing: true, speed: 1.0, last: None },
        fail: FailState::Pass,
        mods: Modifiers::default(),
        title_at: Instant::now(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    sb: Option<CompiledStoryboard>,
    assets: Option<Assets>,
    cfg: PlayerConfig,
    window: Option<Arc<Window>>,
    surface: Option<wgpu::Surface<'static>>,
    ctx: Option<GpuContext>,
    renderer: Option<Renderer>,
    size: (u32, u32),
    format: wgpu::TextureFormat,
    clock: Clock,
    fail: FailState,
    mods: Modifiers,
    title_at: Instant,
}

impl App {
    fn configure_surface(&mut self) {
        let Some(surface) = &self.surface else { return };
        let Some(ctx) = &self.ctx else { return };
        let (w, h) = self.size;
        if w == 0 || h == 0 {
            return;
        }
        let caps = surface.get_capabilities(&ctx.adapter);
        // 优先非 sRGB 格式，保证与 osu! 稳定版相同的伽马空间混合
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| matches!(f, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Rgba8Unorm))
            .or_else(|| caps.formats.first().copied())
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);
        self.format = format;
        surface.configure(
            &ctx.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: w,
                height: h,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
    }

    fn handle_key(&mut self, key: &Key, shift: bool, event_loop: &ActiveEventLoop) {
        let step = if shift { 5000.0 } else { 1000.0 };
        match key {
            Key::Named(NamedKey::Space) => {
                self.clock.playing = !self.clock.playing;
            }
            Key::Named(NamedKey::ArrowLeft) => {
                self.clock.t = (self.clock.t - step).max(0.0);
            }
            Key::Named(NamedKey::ArrowRight) => {
                self.clock.t = (self.clock.t + step).min(self.cfg_limit());
            }
            Key::Named(NamedKey::Escape) => event_loop.exit(),
            Key::Character(c) => match c.to_lowercase().as_str() {
                "r" => {
                    self.clock.t = 0.0;
                    self.clock.playing = true;
                }
                "-" => self.clock.speed = (self.clock.speed / 1.25).max(0.05),
                "=" | "+" => self.clock.speed = (self.clock.speed * 1.25).min(16.0),
                _ => {}
            },
            _ => {}
        }
    }

    fn cfg_limit(&self) -> f32 {
        self.sb.as_ref().map(|s| s.duration + 1500.0).unwrap_or(f32::MAX)
    }

    fn render_frame(&mut self) {
        let Some(surface) = &self.surface else { return };
        let Some(renderer) = &mut self.renderer else { return };
        let Some(sb) = &self.sb else { return };
        let Some(assets) = &mut self.assets else { return };

        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.configure_surface();
                return;
            }
            Err(e) => {
                log::warn!("获取帧失败: {e:?}");
                return;
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let draws = build_draws(renderer, assets, sb, self.clock.t, self.fail);
        renderer.render(&view, self.format, self.size.0, self.size.1, sb.widescreen, &draws);
        frame.present();

        // 每 300ms 更新一次标题
        if self.title_at.elapsed().as_millis() > 300 {
            if let Some(w) = &self.window {
                w.set_title(&format!(
                    "osu-storyboard-render — {:>6.1}s ×{:.2} {}",
                    self.clock.t / 1000.0,
                    self.clock.speed,
                    if self.clock.playing { "▶" } else { "⏸" }
                ));
            }
            self.title_at = Instant::now();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("osu-storyboard-render")
            .with_inner_size(LogicalSize::new(self.cfg.width as f32, self.cfg.height as f32));
        let window = Arc::new(event_loop.create_window(attrs).expect("创建窗口失败"));
        let inner = window.inner_size();
        self.size = (inner.width.max(1), inner.height.max(1));

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("创建 surface 失败");
        let ctx = GpuContext::with_instance(instance, Some(&surface)).expect("初始化 wgpu 失败");
        self.window = Some(window);
        self.surface = Some(surface);
        self.ctx = Some(ctx);
        self.configure_surface();
        let ctx = self.ctx.as_ref().unwrap();
        self.renderer = Some(Renderer::new(&ctx.device, &ctx.queue));

        self.clock.t = self.cfg.start;
        self.clock.speed = self.cfg.speed;
        self.fail = self.cfg.fail;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _window_id: winit::window::WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.size = (size.width.max(1), size.height.max(1));
                self.configure_surface();
            }
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key, state: ElementState::Pressed, .. },
                ..
            } => {
                let shift = self.mods.state().shift_key();
                self.handle_key(&logical_key, shift, event_loop);
            }
            WindowEvent::ModifiersChanged(m) => self.mods = m,
            WindowEvent::RedrawRequested => {
                let limit = self.cfg_limit();
                self.clock.step(limit);
                self.render_frame();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}
