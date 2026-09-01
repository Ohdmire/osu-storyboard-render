//! osu-storyboard-render 库接口:storyboard 解析与 wgpu 精灵渲染。
//!
//! 解析(.osb / .osu Events、缓动、时间轴编译、.osu + 共享 .osb 合并)
//! 已迁至共享的 `osu-parse` crate 并在此 re-export——既有的
//! `osu_storyboard_render::osb` / `::loader` 路径保持可用。本仓库保留
//! wgpu 渲染与播放器,供 CLI 和外部嵌入器(osu-replay-render 等)复用。
//!
//! 嵌入入口:
//! - [`loader::load_beatmap`] —— 从谱面路径加载合并后的 storyboard;
//! - [`render::renderer`] —— 把 storyboard 渲到任意纹理视图(可指定清屏色,
//!   便于合成到宿主的场景里);
//! - [`render::texture::Assets`] —— 素材加载(磁盘/内存),可设缓存上限。

pub use osu_parse::storyboard as osb;
pub use osu_parse::storyboard::loader;

pub mod render;
