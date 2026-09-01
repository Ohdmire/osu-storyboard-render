//! osu-storyboard-render 库接口:storyboard 解析(`osb`)与 wgpu 精灵渲染
//! (`render`),供本仓库的 CLI 与外部嵌入器(osu-replay-render 等)复用。
//!
//! 嵌入入口:
//! - [`loader::load_beatmap`] —— 从谱面路径加载合并后的 storyboard;
//! - [`render::renderer`] —— 把 storyboard 渲到任意纹理视图(可指定清屏色,
//!   便于合成到宿主的场景里);
//! - [`render::texture::Assets`] —— 素材加载(磁盘/内存),可设缓存上限。

pub mod loader;
pub mod osb;
pub mod render;
