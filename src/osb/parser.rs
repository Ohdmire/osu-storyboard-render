//! 解析 `.osb` / `.osu` 的 `[Events]` 节为 `Storyboard`。
//!
//! 兼容的写法：
//! - 层由 `//Storyboard Layer N (...)` 注释区块或元素行内首个参数（数字/名称）共同决定；
//! - 命令行以空白缩进区分，`L`/`T` 组行从属于最近的元素；
//! - 旧版数字元素行（`0,0,"bg.jpg",x,y` 背景、`1,0,"video.mp4"` 视频）；
//! - 带引号的路径（可包含逗号）、缺失的起止时间与起止值。

use crate::osb::easing::Easing;
use crate::osb::model::*;

#[derive(Debug)]
pub struct ParseError(pub String);
impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "storyboard 解析失败: {}", self.0)
    }
}
impl std::error::Error for ParseError {}

enum Group {
    Loop(CommandLoop),
    Trigger(CommandTrigger),
}

struct Cursor {
    sprite: Option<Sprite>,
    is_animation: bool,
    anim: Option<(u32, f32, LoopType)>,
    group: Option<Group>,
}

impl Cursor {
    fn new() -> Cursor {
        Cursor { sprite: None, is_animation: false, anim: None, group: None }
    }

    fn flush_group(&mut self) {
        if let (Some(g), Some(s)) = (self.group.take(), self.sprite.as_mut()) {
            match g {
                Group::Loop(l) => s.loops.push(l),
                Group::Trigger(t) => s.triggers.push(t),
            }
        }
    }

    fn push_command(&mut self, cmd: Command, indent: usize) {
        // 与 lazer 一致按缩进深度分组：深度 >= 2 的命令属于当前 L/T 组，
        // 更浅的命令结束该组、回到元素顶层（否则后置命令会被吞进循环，
        // 迭代时长与命令时间全部错乱）。
        if indent >= 2 {
            match &mut self.group {
                Some(Group::Loop(l)) => l.commands.push(cmd),
                Some(Group::Trigger(t)) => t.commands.push(cmd),
                None => {
                    if let Some(s) = self.sprite.as_mut() {
                        s.commands.push(cmd);
                    }
                }
            }
            return;
        }
        self.flush_group();
        if let Some(s) = self.sprite.as_mut() {
            s.commands.push(cmd);
        }
    }

    fn flush_into(&mut self, sb: &mut Storyboard) {
        self.flush_group();
        if let Some(s) = self.sprite.take() {
            if self.is_animation {
                if let Some((frame_count, frame_delay, loop_type)) = self.anim {
                    sb.elements.push(Element::Animation(Animation {
                        base: s,
                        frame_count,
                        frame_delay,
                        loop_type,
                    }));
                } else {
                    sb.warnings.push("Animation 元素缺少帧数/帧间隔，按 Sprite 处理".into());
                    sb.elements.push(Element::Sprite(s));
                }
            } else {
                sb.elements.push(Element::Sprite(s));
            }
        }
        self.is_animation = false;
        self.anim = None;
    }
}

pub fn parse(input: &str) -> Result<Storyboard, ParseError> {
    let mut sb = Storyboard::default();
    let mut cur = Cursor::new();
    let mut in_events = true;
    let mut current_layer = Layer::Background;

    // .osu/.osb 常带 UTF-8 BOM 与 "osu file format vNN" 头行
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);

    for raw in input.lines() {
        let line = raw.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("osu file format") {
            continue;
        }

        if let Some(rest) = trimmed.strip_prefix("//") {
            // "//Storyboard Layer 3 (Foreground)" 决定后续元素的默认层
            let low = rest.trim().to_ascii_lowercase();
            if let Some(after) = low.strip_prefix("storyboard layer ") {
                if let Some(d) = after.chars().next().and_then(|c| c.to_digit(10)) {
                    if let Some(l) = Layer::from_token(&d.to_string()) {
                        current_layer = l;
                    }
                }
            }
            continue;
        }

        if trimmed.starts_with('[') {
            let header = trimmed.to_ascii_lowercase();
            if header.starts_with("[events") {
                in_events = true;
                cur.flush_into(&mut sb);
            } else if in_events {
                // Events 节结束
                cur.flush_into(&mut sb);
                in_events = false;
            }
            continue;
        }
        if !in_events {
            // [General] 的宽屏标志（.osu），在任意节都可能出现
            if let Some(v) = trimmed.strip_prefix("WidescreenStoryboard:") {
                sb.widescreen = Some(v.trim() == "1");
            }
            continue;
        }

        // 按首个 token 分发：元素/组行与命令行都可能带缩进（真实 .osb 中 L/T 组行
        // 通常缩进一格、组内命令两格），因此不做缩进判断，只看关键字。
        let first_token = trimmed.split(',').next().unwrap_or("").trim().to_ascii_lowercase();
        if matches!(first_token.as_str(), "l" | "t") || !is_command_letter(&first_token) {
            let f = split_csv(trimmed);
            match first_token.as_str() {
                "sprite" | "animation" => {
                    cur.flush_into(&mut sb);
                    match parse_element_line(&f, first_token == "animation", current_layer) {
                        Ok((sprite, anim)) => {
                            cur.sprite = Some(sprite);
                            cur.is_animation = first_token == "animation";
                            cur.anim = anim;
                        }
                        Err(e) => sb.warnings.push(format!("跳过元素行 “{trimmed}”: {e}")),
                    }
                }
                "sample" => {
                    cur.flush_into(&mut sb);
                    if f.len() >= 4 {
                        sb.samples.push(Sample {
                            time: parse_f32(&f[1]).unwrap_or(0.0),
                            layer: parse_f32(&f[2]).unwrap_or(0.0) as i32,
                            path: f[3].clone(),
                        });
                    } else {
                        sb.warnings.push(format!("Sample 行参数不足: {trimmed}"));
                    }
                }
                "video" => {
                    cur.flush_into(&mut sb);
                    if f.len() >= 3 {
                        sb.videos.push(Video { start_time: parse_f32(&f[1]).unwrap_or(0.0), path: f[2].clone() });
                    }
                }
                "l" => {
                    if cur.sprite.is_some() {
                        cur.flush_group();
                        let start = f.get(1).and_then(|s| parse_f32(s)).unwrap_or(0.0);
                        let count = f.get(2).and_then(|s| parse_f32(s)).unwrap_or(1.0).max(0.0) as u32;
                        cur.group = Some(Group::Loop(CommandLoop {
                            start_time: start.max(0.0),
                            total_iterations: count,
                            commands: Vec::new(),
                        }));
                    }
                }
                "t" => {
                    if cur.sprite.is_some() {
                        cur.flush_group();
                        cur.group = Some(Group::Trigger(CommandTrigger {
                            trigger_name: f.get(1).cloned().unwrap_or_default(),
                            start_time: f.get(2).and_then(|s| parse_f32(s)).unwrap_or(0.0),
                            end_time: f.get(3).and_then(|s| parse_f32(s)).unwrap_or(f32::INFINITY),
                            total_iterations: f
                                .get(4)
                                .and_then(|s| parse_f32(s))
                                .map(|v| v.max(0.0) as u32)
                                .unwrap_or(1),
                            commands: Vec::new(),
                        }));
                    }
                }
                "0" => {
                    // 旧版背景: 0,offset,"bg.jpg",x,y
                    if f.len() >= 5 && cur.sprite.is_none() {
                        sb.elements.push(Element::Sprite(Sprite {
                            layer: Layer::Background,
                            origin: Origin::TopLeft,
                            path: f[2].clone(),
                            x: parse_f32(&f[3]).unwrap_or(0.0),
                            y: parse_f32(&f[4]).unwrap_or(0.0),
                            always_visible: true,
                            commands: Vec::new(),
                            loops: Vec::new(),
                            triggers: Vec::new(),
                        }));
                    }
                }
                "1" => {
                    // 旧版视频: 1,offset,"video.mp4"
                    if f.len() >= 3 {
                        sb.videos.push(Video { start_time: parse_f32(&f[1]).unwrap_or(0.0), path: f[2].clone() });
                    }
                }
                "2" | "break" => { /* Break Periods，忽略 */ }
                _ => {
                    sb.warnings.push(format!("忽略无法识别的行: {trimmed}"));
                }
            }
            continue;
        }

        if let Some(cmd) = parse_command(trimmed, &mut sb.warnings) {
            let indent = line.len() - line.trim_start().len();
            cur.push_command(cmd, indent);
        }
    }

    cur.flush_into(&mut sb);
    Ok(sb)
}

fn is_command_letter(tok: &str) -> bool {
    matches!(tok, "f" | "m" | "mx" | "my" | "s" | "v" | "r" | "c" | "p")
}

/// 解析 Sprite/Animation 元素行。返回 (精灵, 动画参数)。
fn parse_element_line(
    f: &[String],
    is_animation: bool,
    default_layer: Layer,
) -> Result<(Sprite, Option<(u32, f32, LoopType)>), String> {
    // 兼容两种形式：
    //   Sprite,layer,origin,path,x,y            （显式层）
    //   Sprite,origin,path,x,y                  （层由注释区块决定）
    let needed = if is_animation { 6 } else { 4 }; // 无层形式的最少参数量
    let mut i = 1;
    let mut layer = default_layer;
    if f.len() > i + 1 {
        if let (Some(l), true) = (
            Layer::from_token(&f[i]),
            Origin::from_token(&f[i + 1]).is_some(),
        ) {
            // 消费层参数后，剩余 token 需仍够 origin,path,x,y(,count,delay[,loop])
            if f.len() - (i + 1) >= needed {
                layer = l;
                i += 1;
            }
        }
    }
    if f.len() < i + needed {
        return Err(format!("参数数量 {} 不足", f.len() - 1));
    }
    let origin = Origin::from_token(&f[i]).unwrap_or(Origin::Centre);
    let path = f[i + 1].clone();
    let x = parse_f32(&f[i + 2]).unwrap_or(320.0);
    let y = parse_f32(&f[i + 3]).unwrap_or(240.0);

    let mut anim = None;
    if is_animation {
        let frame_count = parse_f32(&f[i + 4]).unwrap_or(1.0).max(1.0) as u32;
        let frame_delay = parse_f32(&f[i + 5]).unwrap_or(0.0).max(0.0);
        let loop_type = f.get(i + 6).map(|s| LoopType::from_token(s)).unwrap_or(LoopType::LoopForever);
        anim = Some((frame_count, frame_delay, loop_type));
    }

    Ok((
        Sprite {
            layer,
            origin,
            path,
            x,
            y,
            always_visible: false,
            commands: Vec::new(),
            loops: Vec::new(),
            triggers: Vec::new(),
        },
        anim,
    ))
}

/// 解析缩进的命令行，如 `F,0,0,,1` 或 `M,3,500,1500,320,240,520,240`。
fn parse_command(line: &str, warnings: &mut Vec<String>) -> Option<Command> {
    let f = split_csv(line);
    let ctype = f[0].trim().to_ascii_uppercase();
    if !matches!(ctype.as_str(), "F" | "M" | "MX" | "MY" | "S" | "V" | "R" | "C" | "P") {
        warnings.push(format!("忽略未知命令: {line}"));
        return None;
    }

    let num = |i: usize| -> Option<f32> { f.get(i).and_then(|s| parse_f32(s)) };
    let easing = num(1).map(|v| Easing::from_id(v as i32)).unwrap_or(Easing::LINEAR);
    let start = num(2).unwrap_or(0.0).max(0.0);
    let end = num(3).unwrap_or(start).max(0.0);
    let (start, end) = if end < start { (end, start) } else { (start, end) };

    // 数值成对（from/to），缺失一侧时用另一侧补齐
    let pair = |a: Option<f32>, b: Option<f32>| -> Option<(f32, f32)> {
        match (a, b) {
            (Some(x), Some(y)) => Some((x, y)),
            (Some(x), None) | (None, Some(x)) => Some((x, x)),
            (None, None) => None,
        }
    };
    let vec2 = |a: [Option<f32>; 2], b: [Option<f32>; 2]| -> Option<([f32; 2], [f32; 2])> {
        let x = pair(a[0], b[0])?;
        let y = pair(a[1], b[1])?;
        Some(([x.0, y.0], [x.1, y.1]))
    };

    let effect = match ctype.as_str() {
        "F" => {
            let (from, to) = pair(num(4), num(5))?;
            Effect::Fade { from, to }
        }
        "M" => {
            let (from, to) = vec2([num(4), num(5)], [num(6), num(7)])?;
            Effect::Move { from, to }
        }
        "MX" => {
            let (from, to) = pair(num(4), num(5))?;
            Effect::MoveX { from, to }
        }
        "MY" => {
            let (from, to) = pair(num(4), num(5))?;
            Effect::MoveY { from, to }
        }
        "S" => {
            let (from, to) = pair(num(4), num(5))?;
            Effect::Scale { from, to }
        }
        "V" => {
            let (from, to) = vec2([num(4), num(5)], [num(6), num(7)])?;
            Effect::VecScale { from, to }
        }
        "R" => {
            let (from, to) = pair(num(4), num(5))?;
            Effect::Rotate { from, to }
        }
        "C" => {
            let norm = |v: f32| (v / 255.0).clamp(0.0, 1.0);
            let (r, g, b) = (num(4), num(5), num(6));
            let (r2, g2, b2) = (num(7), num(8), num(9));
            let to = [norm(pair(r, r2)?.1), norm(pair(g, g2)?.1), norm(pair(b, b2)?.1)];
            let from = [norm(pair(r, r2)?.0), norm(pair(g, g2)?.0), norm(pair(b, b2)?.0)];
            Effect::Colour { from, to }
        }
        "P" => {
            let p = f.get(4)?.trim().to_ascii_uppercase();
            let param = match p.as_str() {
                "H" => Parameter::FlipHorizontal,
                "V" => Parameter::FlipVertical,
                "A" => Parameter::AdditiveBlending,
                _ => {
                    warnings.push(format!("忽略未知参数命令 P,{p}"));
                    return None;
                }
            };
            Effect::Parameter(param)
        }
        _ => return None,
    };

    Some(Command { easing, effect, start_time: start, end_time: end })
}

fn parse_f32(s: &str) -> Option<f32> {
    s.trim().parse::<f32>().ok()
}

/// 按逗号切分，支持双引号包裹的字段（引号内容不切分、保留逗号）。
fn split_csv(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut quoted = false;
    for ch in line.chars() {
        match ch {
            '"' => quoted = !quoted,
            ',' if !quoted => {
                out.push(cur.trim().to_string());
                cur.clear();
            }
            c => cur.push(c),
        }
    }
    out.push(cur.trim().to_string());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_elements_and_commands() {
        let sb = parse(
            "[Events]\n\
             //Storyboard Layer 0 (Background)\n\
             Sprite,Centre,\"bg.jpg\",320,240\n\
             F,0,0,1000,0,1\n\
             M,1,0,1000,320,240,520,240\n\
             //Storyboard Layer 3 (Foreground)\n\
             Animation,Centre,\"sb/a.png\",320,240,4,20,LoopOnce\n\
             S,2,0,500,1,2\n",
        )
        .unwrap();
        assert_eq!(sb.elements.len(), 2);
        let Element::Sprite(s) = &sb.elements[0] else { panic!() };
        assert_eq!(s.layer, Layer::Background);
        assert_eq!(s.path, "bg.jpg");
        assert_eq!(s.commands.len(), 2);
        assert_eq!(s.commands[1].effect, Effect::Move { from: [320.0, 240.0], to: [520.0, 240.0] });
        let Element::Animation(a) = &sb.elements[1] else { panic!() };
        assert_eq!(a.base.layer, Layer::Foreground);
        assert_eq!((a.frame_count, a.frame_delay, a.loop_type), (4, 20.0, LoopType::LoopOnce));
    }

    #[test]
    fn quoted_path_with_comma() {
        let sb = parse("Sprite,Centre,\"a,b.png\",10,20\n F,0,0,,1\n").unwrap();
        assert_eq!(sb.elements[0].sprite().path, "a,b.png");
    }

    #[test]
    fn explicit_layer_token() {
        let sb = parse("Sprite,Foreground,Centre,\"x.png\",0,0\n F,0,0,,1\n").unwrap();
        assert_eq!(sb.elements[0].sprite().layer, Layer::Foreground);
    }

    #[test]
    fn loops_and_triggers_attach_to_element() {
        let sb = parse(
            "Sprite,Centre,\"x.png\",0,0\n\
             L,1000,3\n  F,0,0,100,0,1\n \
             T,Passing,0,1000\n  S,0,0,50,1,2\n",
        )
        .unwrap();
        let s = sb.elements[0].sprite();
        assert_eq!(s.loops.len(), 1);
        assert_eq!(s.loops[0].total_iterations, 3);
        assert_eq!(s.loops[0].commands.len(), 1);
        assert_eq!(s.triggers.len(), 1);
        assert_eq!(s.triggers[0].trigger_name, "Passing");
        assert_eq!(s.triggers[0].commands.len(), 1);
    }

    #[test]
    fn command_after_loop_ends_the_group() {
        // world.execute(me); 的实际写法：循环之后还有顶层 S/C/P 命令，
        // 它们不得被吞进循环（否则迭代时长与命令时间全部错乱）
        let sb = parse(
            "Sprite,Centre,\"x.png\",320,240\n \
             L,12002,6\n  M,2,0,3600,320,480,197.067,-107.271\n  F,0,0,3600,1,0\n \
             S,0,15011,,0.2\n \
             C,0,15011,15511,0,0,0,255,255,255\n",
        )
        .unwrap();
        let s = sb.elements[0].sprite();
        assert_eq!(s.loops.len(), 1);
        assert_eq!(s.loops[0].commands.len(), 2, "循环组只有 M/F 两条");
        assert_eq!(s.commands.len(), 2, "S/C 是顶层命令");
        assert_eq!(s.commands[0].effect, Effect::Scale { from: 0.2, to: 0.2 });
        assert_eq!(s.commands[0].start_time, 15011.0);
    }

    #[test]
    fn missing_values_inherit() {
        let sb = parse("Sprite,Centre,\"x.png\",0,0\n F,0,500,,0.5\n M,0,0,,320,240\n").unwrap();
        let s = sb.elements[0].sprite();
        assert_eq!(s.commands[0].effect, Effect::Fade { from: 0.5, to: 0.5 });
        assert_eq!(s.commands[0].start_time, 500.0);
        assert_eq!(s.commands[0].end_time, 500.0);
        assert_eq!(s.commands[1].effect, Effect::Move { from: [320.0, 240.0], to: [320.0, 240.0] });
    }

    #[test]
    fn colour_normalizes() {
        let sb = parse("Sprite,Centre,\"x.png\",0,0\n C,0,0,100,255,0,0,0,255,0\n").unwrap();
        match sb.elements[0].sprite().commands[0].effect {
            Effect::Colour { from, to } => {
                assert!((from[0] - 1.0).abs() < 1e-6 && from[1].abs() < 1e-6);
                assert!(to[1] > 0.99 && to[0].abs() < 1e-6);
            }
            ref e => panic!("{e:?}"),
        }
    }

    #[test]
    fn legacy_background_and_video() {
        let sb = parse("0,0,\"bg.jpg\",0,0\n1,0,\"video.mp4\"\n").unwrap();
        assert_eq!(sb.elements.len(), 1);
        assert!(sb.elements[0].sprite().always_visible);
        assert_eq!(sb.videos.len(), 1);
        assert_eq!(sb.videos[0].path, "video.mp4");
    }

    #[test]
    fn centre_right_origin_and_bom() {
        // BOM + osu file format 头行 + CentreRight origin（Mr HeliX 谱面实际用法）
        let sb = parse(
            "\u{feff}osu file format v14\r\n\
             [Events]\r\n\
             Sprite,Overlay,CentreRight,\"sb/square.png\",800,0\r\n\
             F,0,0,100,0,1\r\n",
        )
        .unwrap();
        assert!(sb.warnings.is_empty(), "{:?}", sb.warnings);
        let s = sb.elements[0].sprite();
        assert_eq!(s.origin, Origin::CentreRight);
        assert_eq!(s.path, "sb/square.png");
        assert_eq!((s.x, s.y), (800.0, 0.0));
    }

    #[test]
    fn parameters_parse() {
        let sb = parse("Sprite,Centre,\"x.png\",0,0\n P,0,0,100,A\n P,0,0,100,H\n").unwrap();
        let s = sb.elements[0].sprite();
        assert_eq!(s.commands[0].effect, Effect::Parameter(Parameter::AdditiveBlending));
        assert_eq!(s.commands[1].effect, Effect::Parameter(Parameter::FlipHorizontal));
    }
}
