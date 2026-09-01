//! osu-storyboard-render —— 从 storyboard 源文件（.osb / .osu）渲染动画。
//!
//! 解析 Events 节的 Sprite/Animation/命令/循环/触发器，用 wgpu 实时播放或离屏截图。
//! 解析/渲染核心在 `osu_storyboard_render` 库中,本文件只是 CLI 壳。

mod player;
mod screenshot;

use osu_storyboard_render::loader;
use osu_storyboard_render::osb::parser;
use osu_storyboard_render::osb::timeline::{CompiledStoryboard, FailState};
use osu_storyboard_render::render::demo;
use osu_storyboard_render::render::texture::{frame_path, Assets};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str = "\
osu-storyboard-render —— 从源文件渲染 osu! storyboard

用法:
  osu-storyboard-render [选项] [storyboard.osb | beatmap.osu | beatmap.osz]

选项:
  --demo                    使用内置 demo storyboard（未指定文件时的默认值）
  --assets <DIR>            素材目录（默认: storyboard 文件所在目录）
  --diff <NAME>             打开 .osz 时选择难度（文件名子串匹配，默认包内第一个）
  --width <N> --height <N>  窗口/截图尺寸（默认 1600x900；非宽屏谱面自动 4:3 黑边）
  --start <MS>              起始时间（毫秒）
  --speed <X>               播放倍速（默认 1.0）
  --fail                    显示 Fail 层（默认 Pass 层）
  --screenshot <MS> <FILE>  无窗口渲染一帧保存为 PNG 后退出
  --list                    解析并打印 storyboard 摘要后退出
  -h, --help                显示本帮助

播放控制: Space 暂停/继续 · ←/→ 快退/快进 1s (Shift: 5s) · R 重播 · -/= 调速 · Esc 退出

说明: 视频（Video）与音效（Sample）元素只解析不渲染；T 触发器组需要游戏事件，
本渲染器不自动激活（可正常渲染其余全部内容）。
打开 .osz 时渲染 <所选难度 .osu 的 Events> + <共用 .osb> 的合并结果，与 osu! 一致。
";

struct Args {
    path: Option<PathBuf>,
    assets_dir: Option<PathBuf>,
    demo: bool,
    diff: Option<String>,
    width: u32,
    height: u32,
    start: f32,
    speed: f32,
    fail: bool,
    list: bool,
    screenshot: Option<(f32, PathBuf)>,
    dump_visible: Option<f32>,
}

fn parse_args(argv: &[String]) -> Result<Args> {
    let mut args = Args {
        path: None,
        assets_dir: None,
        demo: false,
        diff: None,
        width: 1600,
        height: 900,
        start: 0.0,
        speed: 1.0,
        fail: false,
        list: false,
        screenshot: None,
        dump_visible: None,
    };
    let value = |argv: &[String], i: &mut usize, what: &str| -> Result<String> {
        *i += 1;
        argv.get(*i)
            .cloned()
            .with_context(|| format!("缺少 {what} 的参数值"))
    };
    let mut i = 0;
    while i < argv.len() {
        let a = &argv[i];
        match a.as_str() {
            "-h" | "--help" => bail!("{}", USAGE.trim_matches('\n').to_string() + "\n"),
            "--demo" => args.demo = true,
            "--fail" => args.fail = true,
            "--list" => args.list = true,
            "--diff" => args.diff = Some(value(argv, &mut i, "--diff")?),
            "--assets" => args.assets_dir = Some(PathBuf::from(value(argv, &mut i, "--assets")?)),
            "--width" => {
                args.width = value(argv, &mut i, "--width")?
                    .parse()
                    .context("--width 需为正整数")?
            }
            "--height" => {
                args.height = value(argv, &mut i, "--height")?
                    .parse()
                    .context("--height 需为正整数")?
            }
            "--start" => {
                args.start = value(argv, &mut i, "--start")?
                    .parse()
                    .context("--start 需为毫秒数")?
            }
            "--speed" => {
                args.speed = value(argv, &mut i, "--speed")?
                    .parse()
                    .context("--speed 需为数值")?
            }
            "--screenshot" => {
                let ms: f32 = value(argv, &mut i, "--screenshot")?
                    .parse()
                    .context("--screenshot 需为毫秒数")?;
                let out = PathBuf::from(value(argv, &mut i, "--screenshot <MS>")?);
                args.screenshot = Some((ms, out));
            }
            "--dump-visible" => {
                let ms: f32 = value(argv, &mut i, "--dump-visible")?
                    .parse()
                    .context("--dump-visible 需为毫秒数")?;
                args.dump_visible = Some(ms);
            }
            other if other.starts_with('-') => bail!("未知选项 {other}\n\n{USAGE}"),
            other => {
                if args.path.is_some() {
                    bail!("只能指定一个 storyboard 文件");
                }
                args.path = Some(PathBuf::from(other));
            }
        }
        i += 1;
    }
    Ok(args)
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let args = match parse_args(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e:#}");
            return ExitCode::FAILURE;
        }
    };

    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("错误: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<()> {
    let fail = if args.fail { FailState::Fail } else { FailState::Pass };

    let (mut sb, mut assets) = if args.demo || args.path.is_none() {
        if args.path.is_none() && !args.demo {
            println!("未指定文件，使用内置 demo（--demo 可显式指定；-h 查看帮助）");
        }
        (parser::parse(demo::storyboard())?, Assets::memory(demo::textures()))
    } else {
        let loaded = load_storyboard(&args.path.clone().unwrap(), args.diff.as_deref())?;
        let root = args.assets_dir.clone().unwrap_or(loaded.root);
        let assets = Assets::disk(root);

        // 独立播放器保留旧版背景行(常驻精灵);.osu 输入经 loader 与
        // 同目录共用 .osb 合并,纯 .osb 输入直接解析。
        let sb = match loaded.source {
            Source::Osu => loader::load_beatmap(&loaded.osu, false)
                .map(|l| l.story)
                .unwrap_or_default(),
            Source::Osb => std::fs::read_to_string(&loaded.osu)
                .ok()
                .and_then(|t| parser::parse(&t).ok())
                .unwrap_or_default(),
        };
        if sb.elements.is_empty() {
            bail!("storyboard 中没有可渲染的元素");
        }
        (sb, assets)
    };
    if sb.widescreen.is_none() {
        sb.widescreen = Some(true);
    }
    if let Some(ws) = sb.widescreen {
        println!("宽屏 storyboard: {}", if ws { "是" } else { "否（4:3 加黑边）" });
    }
    for w in sb.warnings.iter().take(5) {
        log::warn!("{w}");
    }
    if sb.warnings.len() > 5 {
        log::warn!("… 共 {} 条解析警告", sb.warnings.len());
    }
    let compiled = CompiledStoryboard::compile(sb);

    if args.list {
        print_summary(&compiled, &mut assets);
        return Ok(());
    }

    if let Some(ms) = args.dump_visible {
        // 诊断：按纹理聚合打印 t 时刻可见精灵（x y 缩放 透明度 加色）
        use std::collections::BTreeMap;
        let mut by_tex: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for e in &compiled.elements {
            if let Some(st) = e.state_at(ms) {
                if st.alpha <= 0.002 {
                    continue;
                }
                by_tex.entry(e.path.clone()).or_default().push(format!(
                    "  ({:>7.1},{:>7.1}) s={:.3} a={:.2}{}",
                    st.x, st.y, st.scale_x, st.alpha, if st.additive { " A" } else { "" }
                ));
            }
        }
        for (tex, list) in &by_tex {
            println!("{tex} × {}", list.len());
            for l in list.iter().take(3) {
                println!("{l}");
            }
        }
        println!(
            "总计 {} 个纹理 / {} 个可见精灵 @ {ms}ms",
            by_tex.len(),
            by_tex.values().map(|v| v.len()).sum::<usize>()
        );
        return Ok(());
    }

    if let Some((ms, out)) = &args.screenshot {
        return screenshot::render_screenshot(
            &compiled,
            &mut assets,
            *ms,
            args.width.max(1),
            args.height.max(1),
            fail,
            out,
        );
    }

    let cfg = player::PlayerConfig {
        width: args.width,
        height: args.height,
        start: args.start,
        speed: args.speed,
        fail,
    };
    player::run(compiled, assets, cfg)
}

/// .osz 解包后的 storyboard 组成：难度 .osu（或纯 .osb）+ 素材根目录。
/// （共用 .osb 由 `loader::load_beatmap` 在 .osu 所在目录自行发现并合并。）
struct Loaded {
    /// 输入文件按 .osu 还是 .osb 解释。
    source: Source,
    osu: PathBuf,
    root: PathBuf,
}

enum Source {
    Osu,
    Osb,
}

/// 解析输入路径：.osz 自动解包到临时目录，按 --diff 选择难度；.osb/.osu 直接读取。
fn load_storyboard(path: &Path, diff: Option<&str>) -> Result<Loaded> {
    let is_osz = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("osz"))
        .unwrap_or(false);
    if !is_osz {
        let root = path
            .parent()
            .map(|p| p.to_path_buf())
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new(".").to_path_buf());
        let lower = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let source = if lower == "osb" { Source::Osb } else { Source::Osu };
        return Ok(Loaded { source, osu: path.to_path_buf(), root });
    }

    let file = std::fs::File::open(path).with_context(|| format!("打开 {}", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("读取 .osz 压缩包 {}", path.display()))?;

    let root = std::env::temp_dir().join(format!(
        "osu-storyboard-render-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis()
    ));
    std::fs::create_dir_all(&root)?;

    let mut osb: Option<PathBuf> = None;
    let mut osus: Vec<PathBuf> = Vec::new();
    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if entry.is_dir() {
            continue;
        }
        // zip-slip 防护
        let name = entry.name().to_string();
        if name.split('/').any(|seg| seg == "..") {
            continue;
        }
        let out_path = root.join(&name);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut out = std::fs::File::create(&out_path)?;
        std::io::copy(&mut entry, &mut out)?;
        let lower = name.to_lowercase();
        if lower.ends_with(".osb") {
            osb.get_or_insert(out_path);
        } else if lower.ends_with(".osu") {
            osus.push(out_path);
        }
    }
    // 难度 .osu：--diff 子串匹配（不区分大小写），否则取包内第一个
    let osu = match diff {
        Some(name) => {
            let matches: Vec<&PathBuf> = osus
                .iter()
                .filter(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.to_lowercase().contains(&name.to_lowercase()))
                        .unwrap_or(false)
                })
                .collect();
            match matches.len() {
                0 => bail!(
                    "未找到难度 “{}”，可用难度: {}",
                    name,
                    osus.iter().filter_map(|p| p.file_name()).map(|n| n.to_string_lossy()).collect::<Vec<_>>().join(" / ")
                ),
                1 => Some(matches[0].clone()),
                _ => bail!("难度 “{}” 匹配到多个文件，请用更长的名称", name),
            }
        }
        None => osus.first().cloned(),
    };
    // 只有 .osb 的包:把它当输入文件直接解析。
    let source = if osu.is_some() { Source::Osu } else { Source::Osb };
    let input = osu.clone().or(osb.clone());
    let Some(input) = input else {
        bail!("{} 中未找到 .osb/.osu 文件", path.display());
    };
    log::info!(
        "难度 storyboard: {} + {}",
        osu.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "（无）".into()),
        osb.as_deref().map(|p| p.display().to_string()).unwrap_or_else(|| "（无）".into())
    );
    Ok(Loaded { source, osu: input, root })
}

fn print_summary(cs: &CompiledStoryboard, assets: &mut Assets) {
    println!("元素: {} 个（循环展开后命令 {} 条，展开迭代 {} 次）", cs.elements.len(), cs.total_commands, cs.loop_iterations);
    let mut layer_counts = std::collections::BTreeMap::new();
    let mut anim_count = 0;
    let mut triggers = 0;
    for e in &cs.elements {
        *layer_counts.entry(e.layer.name()).or_insert(0usize) += 1;
        triggers += e.trigger_count;
        if e.animation.is_some() {
            anim_count += 1;
        }
    }
    for (name, n) in &layer_counts {
        println!("  {name:<12} {n} 个元素");
    }
    if anim_count > 0 {
        println!("动画元素: {anim_count} 个");
    }
    if triggers > 0 {
        println!("触发器命令: {triggers} 条（需要游戏事件，本渲染器不激活）");
    }
    if cs.videos > 0 || cs.samples > 0 {
        println!("视频 {} 个 / 音效 {} 个（不渲染）", cs.videos, cs.samples);
    }
    println!("时长: {:.2}s", cs.duration / 1000.0);

    let mut missing = Vec::new();
    for e in &cs.elements {
        let mut check = |p: String, missing: &mut Vec<String>| {
            if assets.get(&p).is_none() {
                missing.push(p);
            }
        };
        match &e.animation {
            Some(a) => {
                for i in 0..a.frame_count {
                    check(frame_path(&e.path, i as usize), &mut missing);
                }
            }
            None => check(e.path.clone(), &mut missing),
        }
    }
    if missing.is_empty() {
        println!("贴图: 全部可解析");
    } else {
        println!("贴图缺失 {} 个:", missing.len());
        for m in missing.iter().take(10) {
            println!("  - {m}");
        }
    }
}
