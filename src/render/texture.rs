//! 素材（贴图）加载与路径解析。
//!
//! storyboard 里的路径是相对谱面目录的；osu! 生态里路径大小写经常不匹配、
//! 也常把素材放在 `sb/` 子目录，这里做三级回退：原样 → `sb/` 前缀 → 目录大小写索引。

use image::RgbaImage;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

pub struct Assets {
    root: Option<PathBuf>,
    memory: HashMap<String, RgbaImage>,
    /// root 下全部文件的小写相对路径索引（懒构建）。
    index: Option<HashMap<String, PathBuf>>,
    cache: HashMap<String, Option<RgbaImage>>,
    warned: HashSet<String>,
}

/// 把 storyboard 路径规范化为 `/` 分隔。
pub fn normalize_path(p: &str) -> String {
    p.trim().trim_matches('"').replace('\\', "/")
}

/// 动画第 i 帧的路径：`sb/a.png` → `sb/a3.png`。
pub fn frame_path(path: &str, index: usize) -> String {
    let dir_end = path.rfind('/').map(|i| i + 1).unwrap_or(0);
    match path[dir_end..].rfind('.') {
        Some(dot) => {
            let at = dir_end + dot;
            format!("{}{}.{}", &path[..at], index, &path[at + 1..])
        }
        None => format!("{}{}", path, index),
    }
}

impl Assets {
    /// 磁盘素材：`root` 为谱面目录。
    pub fn disk(root: impl Into<PathBuf>) -> Assets {
        Assets {
            root: Some(root.into()),
            memory: HashMap::new(),
            index: None,
            cache: HashMap::new(),
            warned: HashSet::new(),
        }
    }

    /// 内存素材（内置 demo 用）：key 为 normalize 后的小写路径。
    pub fn memory(map: HashMap<String, RgbaImage>) -> Assets {
        Assets {
            root: None,
            memory: map,
            index: None,
            cache: HashMap::new(),
            warned: HashSet::new(),
        }
    }

    pub fn get(&mut self, logical: &str) -> Option<&RgbaImage> {
        let norm = normalize_path(logical);
        if !self.cache.contains_key(&norm) {
            let loaded = self.load(&norm);
            if loaded.is_none() && self.warned.insert(norm.clone()) {
                log::warn!("缺少贴图: {norm}");
            }
            self.cache.insert(norm.clone(), loaded);
        }
        self.cache.get(&norm).and_then(|o| o.as_ref())
    }

    fn load(&mut self, norm: &str) -> Option<RgbaImage> {
        if !self.memory.is_empty() {
            let lower = norm.to_lowercase();
            if let Some(img) = self.memory.get(&lower) {
                return Some(img.clone());
            }
            // 无扩展名时补常见扩展（osu! 稳定版行为）
            for ext in [".png", ".jpg", ".jpeg"] {
                if let Some(img) = self.memory.get(&format!("{lower}{ext}")) {
                    return Some(img.clone());
                }
            }
            return None;
        }
        let root = self.root.as_ref()?;

        // 路径变体：原样；文件名无扩展名时补 .png/.jpg/.jpeg
        let mut variants = vec![norm.to_string()];
        let file_name = norm.rsplit('/').next().unwrap_or(norm);
        if !file_name.contains('.') {
            for ext in [".png", ".jpg", ".jpeg"] {
                variants.push(format!("{norm}{ext}"));
            }
        }

        for prefix in ["", "sb/"] {
            for v in &variants {
                let cand = root.join(format!("{prefix}{v}"));
                if cand.is_file() {
                    return decode(&cand);
                }
            }
        }

        if self.index.is_none() {
            let mut idx = HashMap::new();
            walk(root, root, 0, &mut idx);
            self.index = Some(idx);
        }
        for prefix in ["", "sb/"] {
            for v in &variants {
                if let Some(p) = self.index.as_ref().unwrap().get(&format!("{prefix}{}", v.to_lowercase())) {
                    return decode(p);
                }
            }
        }
        None
    }
}

fn decode(p: &Path) -> Option<RgbaImage> {
    image::open(p).ok().map(|d| d.to_rgba8())
}

fn walk(root: &Path, dir: &Path, depth: u32, out: &mut HashMap<String, PathBuf>) {
    if depth > 6 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(root, &p, depth + 1, out);
        } else if let Ok(rel) = p.strip_prefix(root) {
            out.insert(rel.to_string_lossy().replace('\\', "/").to_lowercase(), p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_paths() {
        assert_eq!(frame_path("sb/a.png", 3), "sb/a3.png");
        assert_eq!(frame_path("anim.jpg", 12), "anim12.jpg");
        assert_eq!(frame_path("noext", 2), "noext2");
    }

    #[test]
    fn normalize() {
        assert_eq!(normalize_path("sb\\a.png"), "sb/a.png");
        assert_eq!(normalize_path(" \"a,b.png\" "), "a,b.png");
    }
}
