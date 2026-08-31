//! 把 `Storyboard` 编译为可直接求值的时间轴：
//! - 展开 `L` 命令循环为绝对时间命令；
//! - 按通道（透明度、x、y、缩放、旋转、颜色、参数）分桶并按开始时间排序；
//! - 提供 `state_at(t)` 对任意时刻求值。
//!
//! `T` 触发器组只解析不自动激活（渲染器无法产生游戏事件），保留计数用于摘要。

use crate::osb::model::*;

/// 游戏状态：显示 Fail 层还是 Pass 层。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailState {
    Pass,
    Fail,
}

/// 某个精灵在时刻 t 的完整状态。
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteState {
    pub x: f32,
    pub y: f32,
    pub scale_x: f32,
    pub scale_y: f32,
    pub rotation: f32, // 弧度，正值为顺时针
    pub colour: [f32; 3],
    pub alpha: f32,
    pub flip_h: bool,
    pub flip_v: bool,
    pub additive: bool,
}

#[derive(Default)]
struct Channels {
    alpha: Vec<Command>,
    pos_x: Vec<Command>,
    pos_y: Vec<Command>,
    scale_x: Vec<Command>,
    scale_y: Vec<Command>,
    rotation: Vec<Command>,
    colour: Vec<Command>,
    flip_h: Vec<Command>,
    flip_v: Vec<Command>,
    additive: Vec<Command>,
}

impl Channels {
    fn push(&mut self, cmd: &Command) {
        match cmd.effect {
            Effect::Fade { .. } => self.alpha.push(cmd.clone()),
            Effect::Move { .. } => {
                self.pos_x.push(cmd.clone());
                self.pos_y.push(cmd.clone());
            }
            Effect::MoveX { .. } => self.pos_x.push(cmd.clone()),
            Effect::MoveY { .. } => self.pos_y.push(cmd.clone()),
            Effect::Scale { .. } => {
                self.scale_x.push(cmd.clone());
                self.scale_y.push(cmd.clone());
            }
            Effect::VecScale { .. } => {
                self.scale_x.push(cmd.clone());
                self.scale_y.push(cmd.clone());
            }
            Effect::Rotate { .. } => self.rotation.push(cmd.clone()),
            Effect::Colour { .. } => self.colour.push(cmd.clone()),
            Effect::Parameter(p) => match p {
                Parameter::FlipHorizontal => self.flip_h.push(cmd.clone()),
                Parameter::FlipVertical => self.flip_v.push(cmd.clone()),
                Parameter::AdditiveBlending => self.additive.push(cmd.clone()),
            },
        }
    }

    fn finish(&mut self) {
        let by_start = |a: &Command, b: &Command| {
            a.start_time.total_cmp(&b.start_time).then(b.end_time.total_cmp(&a.end_time))
        };
        self.alpha.sort_by(by_start);
        self.pos_x.sort_by(by_start);
        self.pos_y.sort_by(by_start);
        self.scale_x.sort_by(by_start);
        self.scale_y.sort_by(by_start);
        self.rotation.sort_by(by_start);
        self.colour.sort_by(by_start);
        self.flip_h.sort_by(by_start);
        self.flip_v.sort_by(by_start);
        self.additive.sort_by(by_start);
    }
}

pub struct AnimationInfo {
    pub frame_count: u32,
    pub frame_delay: f32,
    pub loop_type: LoopType,
}

/// 编译后的单个元素。
pub struct CompiledElement {
    pub layer: Layer,
    pub origin: Origin,
    pub path: String,
    pub start_pos: [f32; 2],
    pub animation: Option<AnimationInfo>,
    /// 旧版背景行生成的常驻精灵：不受命令时间窗约束。
    pub always_visible: bool,
    pub start: f32,
    pub end: f32,
    pub trigger_count: usize,
    ch: Channels,
}

impl CompiledElement {
    /// 时刻 t 不在元素生命周期内返回 None。
    pub fn state_at(&self, t: f32) -> Option<SpriteState> {
        if !self.always_visible && !(self.start..=self.end + 1.0).contains(&t) {
            return None;
        }
        // osu! 语义（lazer `StoryboardCommand.ApplyInitialValue`）：命令在其开始时间
        // 之前钳制为起始值，因此某通道首条命令开始前，属性 = 首条命令的起始值；
        // Sprite 行声明的 x/y 仅在该通道没有任何命令时生效。
        let [dx, dy] = self.start_pos;
        Some(SpriteState {
            x: sample_f32(&self.ch.pos_x, t, initial(&self.ch.pos_x, |c| c.x(), dx), |c| c.x()),
            y: sample_f32(&self.ch.pos_y, t, initial(&self.ch.pos_y, |c| c.y(), dy), |c| c.y()),
            scale_x: sample_f32(&self.ch.scale_x, t, initial(&self.ch.scale_x, |c| c.scale_x(), 1.0), |c| c.scale_x()),
            scale_y: sample_f32(&self.ch.scale_y, t, initial(&self.ch.scale_y, |c| c.scale_y(), 1.0), |c| c.scale_y()),
            rotation: sample_f32(&self.ch.rotation, t, initial(&self.ch.rotation, |c| c.rotation(), 0.0), |c| c.rotation()),
            colour: sample_colour(&self.ch.colour, t, initial_colour(&self.ch.colour)),
            alpha: sample_f32(&self.ch.alpha, t, initial(&self.ch.alpha, |c| c.alpha(), 1.0), |c| c.alpha()),
            flip_h: active(&self.ch.flip_h, t),
            flip_v: active(&self.ch.flip_v, t),
            additive: active(&self.ch.additive, t),
        })
    }

    /// 动画当前帧索引（帧图片路径 = 路径去扩展名 + 索引 + 扩展名）。
    pub fn frame_at(&self, t: f32) -> usize {
        let Some(a) = &self.animation else { return 0 };
        let count = a.frame_count.max(1) as i64;
        let idx = if a.frame_delay <= 0.0 {
            0
        } else {
            (t / a.frame_delay).floor() as i64
        };
        let idx = match a.loop_type {
            LoopType::LoopOnce => idx.clamp(0, count - 1),
            LoopType::LoopForever => idx.rem_euclid(count),
        };
        idx as usize
    }
}

pub struct CompiledStoryboard {
    /// 按层序（Background→Fail/Pass→Foreground→Overlay）稳定排序后的元素。
    pub elements: Vec<CompiledElement>,
    /// 最后一条命令的结束时间（毫秒）。
    pub duration: f32,
    pub total_commands: usize,
    pub loop_iterations: usize,
    pub videos: usize,
    pub samples: usize,
    /// WidescreenStoryboard：true = 640×480 向两侧扩展；false = 固定 4:3（宽窗口加黑边）。
    pub widescreen: bool,
}

impl CompiledStoryboard {
    pub fn compile(sb: Storyboard) -> CompiledStoryboard {
        let mut elements = Vec::new();
        let mut total_commands = 0usize;
        let mut loop_iterations = 0usize;
        let mut duration = 0.0f32;

        for element in sb.elements {
            let sprite = element.sprite();
            let mut ch = Channels::default();
            let mut count = 0usize;

            for c in &sprite.commands {
                ch.push(&clamp_positive(c.clone()));
                count += 1;
            }
            for l in &sprite.loops {
                // 循环单次迭代时长 = 组内最长命令结束时间（相对值）
                let iter_dur = l.commands.iter().map(|c| c.end_time).fold(0.0f32, f32::max);
                for i in 0..l.total_iterations {
                    let offset = l.start_time + iter_dur * i as f32;
                    for c in &l.commands {
                        let mut a = c.clone();
                        a.start_time = (a.start_time + offset).max(0.0);
                        a.end_time = (a.end_time + offset).max(0.0);
                        if a.end_time < a.start_time {
                            std::mem::swap(&mut a.start_time, &mut a.end_time);
                        }
                        ch.push(&a);
                        count += 1;
                    }
                }
                loop_iterations += l.total_iterations as usize;
            }

            let trigger_count = sprite.triggers.iter().map(|t| t.commands.len()).sum::<usize>();

            let mut start = f32::MAX;
            let mut end = f32::MIN;
            let all: [&[Command]; 10] = [
                &ch.alpha, &ch.pos_x, &ch.pos_y, &ch.scale_x, &ch.scale_y,
                &ch.rotation, &ch.colour, &ch.flip_h, &ch.flip_v, &ch.additive,
            ];
            for cmds in all {
                for c in cmds {
                    start = start.min(c.start_time);
                    end = end.max(c.end_time);
                }
            }
            if sprite.always_visible {
                start = 0.0;
                end = f32::INFINITY;
            }
            if end.is_finite() {
                duration = duration.max(end);
            }
            total_commands += count;
            ch.finish();

            let animation = match &element {
                Element::Animation(a) => Some(AnimationInfo {
                    frame_count: a.frame_count,
                    frame_delay: a.frame_delay,
                    loop_type: a.loop_type,
                }),
                Element::Sprite(_) => None,
            };

            elements.push(CompiledElement {
                layer: sprite.layer,
                origin: sprite.origin,
                path: sprite.path.clone(),
                start_pos: [sprite.x, sprite.y],
                animation,
                always_visible: sprite.always_visible,
                start,
                end,
                trigger_count,
                ch,
            });
        }

        // 稳定排序：层内保持文件顺序，层间按渲染顺序
        elements.sort_by_key(|e| e.layer as u8);

        CompiledStoryboard {
            elements,
            duration,
            total_commands,
            loop_iterations,
            videos: sb.videos.len(),
            samples: sb.samples.len(),
            widescreen: sb.widescreen.unwrap_or(true),
        }
    }
}

fn clamp_positive(mut c: Command) -> Command {
    c.start_time = c.start_time.max(0.0);
    c.end_time = c.end_time.max(0.0);
    if c.end_time < c.start_time {
        std::mem::swap(&mut c.start_time, &mut c.end_time);
    }
    c
}

/// 通道首条命令的起始值（`finish()` 已按开始时间升序，故即首个）；通道为空时回退默认值。
fn initial(cmds: &[Command], pick: impl Fn(&Command) -> Option<(f32, f32)>, fallback: f32) -> f32 {
    cmds.first().and_then(pick).map_or(fallback, |(from, _)| from)
}

fn initial_colour(cmds: &[Command]) -> [f32; 3] {
    cmds.first().and_then(|c| c.colour()).map_or([1.0, 1.0, 1.0], |(from, _)| from)
}

/// 对单通道命令序列采样。命令按开始时间升序，后开始的命令覆盖先开始的；
/// 命令结束后值保持，重叠时后启动者生效。
fn sample_f32(
    cmds: &[Command],
    t: f32,
    default: f32,
    pick: impl Fn(&Command) -> Option<(f32, f32)>,
) -> f32 {
    let mut value = default;
    for c in cmds {
        if c.start_time > t {
            break;
        }
        let Some((from, to)) = pick(c) else { continue };
        if t >= c.end_time {
            value = to;
        } else if t <= c.start_time {
            value = from;
        } else {
            let span = (c.end_time - c.start_time).max(1e-6);
            let p = c.easing.apply((t - c.start_time) / span);
            value = from + (to - from) * p;
        }
    }
    value
}

fn sample_colour(cmds: &[Command], t: f32, default: [f32; 3]) -> [f32; 3] {
    let mut value = default;
    for c in cmds {
        if c.start_time > t {
            break;
        }
        let Some((from, to)) = c.colour() else { continue };
        let v = if t >= c.end_time {
            1.0
        } else if t <= c.start_time {
            0.0
        } else {
            let span = (c.end_time - c.start_time).max(1e-6);
            c.easing.apply((t - c.start_time) / span)
        };
        value = [
            from[0] + (to[0] - from[0]) * v,
            from[1] + (to[1] - from[1]) * v,
            from[2] + (to[2] - from[2]) * v,
        ];
    }
    value
}

/// 参数命令：处于任一命令时间区间内即激活。
fn active(cmds: &[Command], t: f32) -> bool {
    cmds.iter().any(|c| c.start_time <= t && t <= c.end_time && c.parameter().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::osb::parser::parse;

    fn compile_one(text: &str) -> CompiledStoryboard {
        let sb = parse(text).unwrap();
        CompiledStoryboard::compile(sb)
    }

    #[test]
    fn fade_interpolation_and_hold() {
        let cs = compile_one("Sprite,Centre,\"x.png\",0,0\n F,0,0,1000,0,1\n");
        let e = &cs.elements[0];
        assert_eq!(e.state_at(0.0).unwrap().alpha, 0.0);
        assert!((e.state_at(500.0).unwrap().alpha - 0.5).abs() < 1e-4);
        assert!((e.state_at(999.0).unwrap().alpha - 0.999).abs() < 1e-3);
        assert!(e.state_at(5000.0).is_none(), "最后一条命令结束后不再渲染");
    }

    #[test]
    fn lifetime_window() {
        let cs = compile_one("Sprite,Centre,\"x.png\",0,0\n F,0,500,1000,0,1\n");
        let e = &cs.elements[0];
        assert!(e.state_at(100.0).is_none(), "命令窗口之前不可见");
        assert!(e.state_at(1000.0).is_some(), "结束时刻仍可见");
        assert!(e.state_at(2000.0).is_none(), "窗口之后移除");
    }

    #[test]
    fn move_and_movex_compose() {
        let cs = compile_one(
            "Sprite,Centre,\"x.png\",0,0\n M,0,0,1000,0,0,100,100\n MX,0,500,1500,100,300\n",
        );
        let e = &cs.elements[0];
        let s = e.state_at(750.0).unwrap();
        // MX 500..1500 进度 0.25 → x = 100 + 200*0.25 = 150；M 继续驱动 y = 75
        assert!((s.x - 150.0).abs() < 1e-3, "MX 覆盖 x: {}", s.x);
        assert!((s.y - 75.0).abs() < 1e-3, "M 继续驱动 y: {}", s.y);
        let s = e.state_at(250.0).unwrap();
        assert!((s.x - 25.0).abs() < 1e-3);
    }

    #[test]
    fn loop_expansion() {
        let cs = compile_one(
            "Sprite,Centre,\"x.png\",0,0\n L,1000,3\n  M,0,0,200,0,0,50,0\n",
        );
        let e = &cs.elements[0];
        // 迭代 [1000,1200]/[1200,1400]/[1400,1600]；t=1300 处于第 2 次迭代中点
        let s = e.state_at(1300.0).unwrap();
        assert!((s.x - 25.0).abs() < 1e-3, "{}", s.x);
        let s = e.state_at(1250.0).unwrap();
        assert!((s.x - 12.5).abs() < 1e-3, "{}", s.x);
        assert!(e.state_at(999.0).is_none());
        assert!((cs.duration - 1600.0).abs() < 1e-3);
    }

    #[test]
    fn parameters_and_colour() {
        let cs = compile_one(
            "Sprite,Centre,\"x.png\",0,0\n P,0,100,300,A\n P,0,200,400,H\n C,0,0,100,255,255,255,255,0,0\n",
        );
        let e = &cs.elements[0];
        let s = e.state_at(150.0).unwrap();
        assert!(s.additive && !s.flip_h);
        let s = e.state_at(250.0).unwrap();
        assert!(s.additive && s.flip_h);
        let s = e.state_at(399.0).unwrap();
        assert!(!s.additive && s.flip_h, "flip_h 命令 200..400 仍生效");
        assert!((s.colour[0] - 1.0).abs() < 1e-3 && s.colour[1].abs() < 1e-3);
    }

    #[test]
    fn animation_frames() {
        let cs = compile_one(
            "Animation,Centre,\"a.png\",0,0,8,90,LoopForever\n F,0,0,1000,1,1\n",
        );
        let e = &cs.elements[0];
        assert_eq!(e.frame_at(0.0), 0);
        assert_eq!(e.frame_at(1000.0), 1000 / 90 % 8);
        let cs = compile_one(
            "Animation,Centre,\"a.png\",0,0,4,100,LoopOnce\n F,0,0,1000,1,1\n",
        );
        let e = &cs.elements[0];
        assert_eq!(e.frame_at(1_000_000.0), 3);
    }

    #[test]
    fn layers_sorted_stable() {
        let cs = compile_one(
            "//Storyboard Layer 3 (Foreground)\n\
             Sprite,Centre,\"a.png\",0,0\n F,0,0,10,1,1\n\
             //Storyboard Layer 0 (Background)\n\
             Sprite,Centre,\"b.png\",0,0\n F,0,0,10,1,1\n\
             Sprite,Centre,\"c.png\",0,0\n F,0,0,10,1,1\n",
        );
        let layers: Vec<Layer> = cs.elements.iter().map(|e| e.layer).collect();
        assert_eq!(layers, vec![Layer::Background, Layer::Background, Layer::Foreground]);
        assert_eq!(cs.elements[0].path, "b.png"); // 层内保持文件顺序
        assert_eq!(cs.elements[1].path, "c.png");
    }

    #[test]
    fn easing_used_in_interpolation() {
        let cs = compile_one("Sprite,Centre,\"x.png\",0,0\n M,3,0,1000,0,0,100,0\n"); // InQuad
        let s = cs.elements[0].state_at(500.0).unwrap();
        assert!((s.x - 25.0).abs() < 1e-3, "{}", s.x);
    }

    #[test]
    fn position_before_first_command_follows_first_command() {
        // DEEP BLUE SONG 的实际模式：精灵声明在 (0,0)（Centre），淡入早于首条 M；
        // osu! 在首条位置命令开始前取其起始值 (320,240)，而不是声明的 (0,0)。
        let cs = compile_one(
            "Sprite,Centre,\"x.png\",0,0\n F,0,1000,,1\n M,0,9000,10000,320,240,170,290\n",
        );
        let e = &cs.elements[0];
        let s = e.state_at(1000.0).unwrap();
        assert!(
            (s.x - 320.0).abs() < 1e-3 && (s.y - 240.0).abs() < 1e-3,
            "首条 M 开始前应位于其起始值: {:?}",
            (s.x, s.y)
        );
        let s = e.state_at(9500.0).unwrap();
        assert!((s.x - 245.0).abs() < 1e-3, "M 期间线性插值: {}", s.x);
    }

    #[test]
    fn channel_without_commands_keeps_declared_value() {
        // 仅有 MX 时 y 通道无命令，y 保持 Sprite 行声明的值
        let cs = compile_one("Sprite,Centre,\"x.png\",0,77\n F,0,1000,,1\n MX,0,5000,6000,10,300\n");
        let s = cs.elements[0].state_at(1000.0).unwrap();
        assert!((s.x - 10.0).abs() < 1e-3, "{}", s.x);
        assert!((s.y - 77.0).abs() < 1e-3, "{}", s.y);
    }
    #[test]
    fn colour_before_first_command_takes_first_start_value() {
        let cs =
            compile_one("Sprite,Centre,\"x.png\",0,0\n F,0,0,,1\n C,0,5000,6000,255,0,0,0,255,0\n");
        let s = cs.elements[0].state_at(100.0).unwrap();
        assert!(s.colour[0] > 0.99 && s.colour[1].abs() < 1e-6, "{:?}", s.colour);
    }
}
