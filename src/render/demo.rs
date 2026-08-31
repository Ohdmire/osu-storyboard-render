//! 内置 demo：一份展示常用命令的 storyboard 源码 + 程序生成的贴图。
//! 走与真实 .osb 完全相同的解析/编译/渲染管线，便于无素材验证。

use image::{Rgba, RgbaImage};
use std::collections::HashMap;

pub fn storyboard() -> &'static str {
    "\
[Events]
//Background and Video events
//Storyboard Layer 0 (Background)
Sprite,BottomCentre,demo/bg.png,320,480
 F,0,0,,0,1
 C,17,0,16000,255,255,255,110,80,170
//Storyboard Layer 2 (Pass)
Animation,Centre,demo/pulse.png,320,240,8,90,LoopForever
 F,0,800,,0,0.9
 S,5,800,9500,0.7,1.7
 M,11,800,9500,320,90,320,390
 P,0,6000,7000,H
//Storyboard Layer 3 (Foreground)
Sprite,Centre,demo/star.png,320,240
 F,0,400,,0,1
 F,0,9500,,1,0
 M,1,400,4800,320,50,320,410
 M,2,4800,9500,320,410,320,50
 R,0,400,9500,0,12.56637
 V,6,400,9500,0.4,0.4,1.3,1.3
Sprite,Centre,demo/star.png,320,240
 F,0,6000,,0,0.6
 P,0,6000,9500,A
 S,7,6000,9500,1.5,0.5
Sprite,TopLeft,demo/box.png,80,80
 F,0,0,,0,1
 L,200,10
  M,25,0,1500,80,80,560,400
  M,25,1500,3000,560,400,80,80
  R,0,0,3000,0,6.28318
Sample,5000,0,\"demo/none.wav\"
"
}

pub fn textures() -> HashMap<String, RgbaImage> {
    let mut m = HashMap::new();
    m.insert("demo/bg.png".to_string(), background(1280, 960));
    m.insert("demo/star.png".to_string(), star(256));
    m.insert("demo/box.png".to_string(), box_texture(200));
    for i in 0..8 {
        m.insert(format!("demo/pulse{}.png", i), pulse(256, i));
    }
    m
}

fn background(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        let t = y as f32 / h as f32;
        let top = [16.0, 10.0, 38.0];
        let bot = [88.0, 48.0, 118.0];
        let mut px = [
            top[0] + (bot[0] - top[0]) * t,
            top[1] + (bot[1] - top[1]) * t,
            top[2] + (bot[2] - top[2]) * t,
        ];
        // 伪随机星点
        let hash = (x as u64).wrapping_mul(73856093) ^ (y as u64).wrapping_mul(19349663);
        if hash % 1201 < 4 {
            px = [200.0 + (hash % 55) as f32, 200.0, 235.0];
        }
        Rgba([px[0] as u8, px[1] as u8, px[2] as u8, 255])
    })
}

fn star(size: u32) -> RgbaImage {
    let c = size as f32 / 2.0 - 0.5;
    RgbaImage::from_fn(size, size, |x, y| {
        let dx = (x as f32 - c) / c;
        let dy = (y as f32 - c) / c;
        let d = (dx * dx + dy * dy).sqrt();
        let glow = (1.0 - d).clamp(0.0, 1.0).powf(2.5);
        // 十字光芒
        let rays = ((1.0 - dx.abs()) * (1.0 - dy.abs())).clamp(0.0, 1.0).powf(3.0);
        let a = (glow * (0.25 + 0.75 * rays) * 1.6).clamp(0.0, 1.0);
        let core = (1.0 - d * 3.5).clamp(0.0, 1.0);
        let r = 255.0 * (0.45 + 0.55 * core);
        let g = 220.0 * (0.5 + 0.5 * core) + 35.0 * core;
        let b = 140.0 + 115.0 * core;
        Rgba([r as u8, g.min(255.0) as u8, b.min(255.0) as u8, (a * 255.0) as u8])
    })
}

fn box_texture(size: u32) -> RgbaImage {
    let border = size / 20;
    RgbaImage::from_fn(size, size, |x, y| {
        let edge = x < border || y < border || x >= size - border || y >= size - border;
        if edge {
            Rgba([140, 55, 12, 255])
        } else {
            Rgba([232, 122, 42, 255])
        }
    })
}

/// 第 i 帧：半径随 i 增大的圆环。
fn pulse(size: u32, i: u32) -> RgbaImage {
    let c = size as f32 / 2.0 - 0.5;
    let radius = 18.0 + i as f32 * 24.0;
    RgbaImage::from_fn(size, size, |x, y| {
        let d = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt();
        let dist = (d - radius).abs();
        if dist < 7.0 {
            let a = (1.0 - dist / 7.0) * (1.0 - i as f32 / 10.0);
            Rgba([110, 225, 255, (a * 255.0) as u8])
        } else {
            Rgba([0, 0, 0, 0])
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{storyboard, textures};
    use crate::osb::parser::parse;
    use crate::osb::timeline::CompiledStoryboard;
    use crate::render::texture::normalize_path;

    #[test]
    fn demo_storyboard_compiles() {
        let sb = parse(storyboard()).expect("demo storyboard 应能解析");
        assert!(sb.warnings.is_empty(), "{:?}", sb.warnings);
        let cs = CompiledStoryboard::compile(sb);
        assert!(cs.elements.len() >= 5);
        assert!(cs.duration > 25_000.0, "duration={}", cs.duration);
        assert!(cs.loop_iterations > 0, "demo 应包含循环");
    }

    #[test]
    fn demo_textures_cover_all_references() {
        // storyboard 里引用的每一帧都能在生成贴图中找到
        let sb = parse(storyboard()).unwrap();
        let tex = textures();
        let mut keys: Vec<String> = tex.keys().cloned().collect();
        for e in sb.elements.iter() {
            let s = e.sprite();
            if let crate::osb::model::Element::Animation(a) = e {
                for i in 0..a.frame_count {
                    keys.push(crate::render::texture::frame_path(&s.path, i as usize));
                }
            } else {
                keys.push(s.path.clone());
            }
        }
        for k in keys {
            assert!(tex.contains_key(&normalize_path(&k).to_lowercase()), "缺少 {k}");
        }
    }
}
