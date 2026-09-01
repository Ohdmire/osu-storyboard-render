//! 从谱面目录加载 storyboard:所选难度 `.osu` 的 `[Events]` 在前、谱组共用
//! `.osb` 在后合并(与 osu! 稳定版一致),素材根目录为谱面所在目录。

use crate::osb::{model::Storyboard, parser};
use std::path::{Path, PathBuf};

/// 加载结果:合并后的 storyboard + 素材根目录。
pub struct LoadedStoryboard {
    pub story: Storyboard,
    pub root: PathBuf,
}

/// 解析谱面的 storyboard。无 `.osb` 时只用 `.osu` 自身的 Events;
/// 两者都空(或文件不可读)返回 `None`。
///
/// `strip_background`:剔除旧版背景行生成的常驻精灵(`always_visible`)。
/// 嵌入宿主(osu-replay-render)自己绘制谱面背景图,置 true 避免同一张
/// 背景被宿主和 storyboard 各画一遍;独立播放器置 false 保留背景。
pub fn load_beatmap(map_path: &Path, strip_background: bool) -> Option<LoadedStoryboard> {
    let root = map_path
        .parent()
        .map(|p| p.to_path_buf())
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new(".").to_path_buf());

    let mut story = match std::fs::read_to_string(map_path) {
        Ok(text) => parser::parse(&text).ok()?,
        Err(_) => return None,
    };

    // 谱组共用 .osb(目录下第一个;同组多份属于异常打包,取稳定序第一个)。
    let shared = std::fs::read_dir(&root)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case("osb"))
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>()
        })
        .and_then(|paths| paths.into_iter().min_by(|a, b| a.file_name().cmp(&b.file_name())));

    if let Some(shared) = shared {
        // 稳定版中谱面背景图由 .osb 接管:编辑器把背景写成 .osb 首个精灵并用
        // F,0,0,,0 隐藏,合并时跳过 .osu 的旧版背景行。
        if strip_background {
            story.elements.retain(|e| !e.sprite().always_visible);
        }
        if let Ok(text) = std::fs::read_to_string(&shared) {
            if let Ok(shared_story) = parser::parse(&text) {
                story.elements.extend(shared_story.elements);
                story.videos.extend(shared_story.videos);
                story.samples.extend(shared_story.samples);
                if story.widescreen.is_none() {
                    story.widescreen = shared_story.widescreen;
                }
            }
        }
    }

    // 背景行剔除后可能再无元素(纯背景图谱面)——但视频元素仍值得返回
    // (lazer 的 storyboard 含 Video 层;谱面可以只有视频没有精灵)。
    if strip_background {
        story.elements.retain(|e| !e.sprite().always_visible);
    }
    if story.elements.is_empty() && story.videos.is_empty() {
        return None;
    }
    if story.widescreen.is_none() {
        story.widescreen = Some(true);
    }
    Some(LoadedStoryboard { story, root })
}
