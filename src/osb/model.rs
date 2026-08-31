//! Storyboard 数据模型 —— `.osb` / `.osu` Events 节的内存表示。
//!
//! 命令、循环、触发器的时间单位均为毫秒；颜色的分量在解析时已归一化到 0..1。

use crate::osb::easing::Easing;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Layer {
    Background = 0,
    Fail = 1,
    Pass = 2,
    Foreground = 3,
    Overlay = 4,
}

impl Layer {
    pub fn from_token(tok: &str) -> Option<Layer> {
        match tok.trim().to_ascii_lowercase().as_str() {
            "background" | "0" => Some(Layer::Background),
            "fail" | "1" => Some(Layer::Fail),
            "pass" | "2" => Some(Layer::Pass),
            "foreground" | "3" => Some(Layer::Foreground),
            "overlay" | "4" => Some(Layer::Overlay),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Layer::Background => "Background",
            Layer::Fail => "Fail",
            Layer::Pass => "Pass",
            Layer::Foreground => "Foreground",
            Layer::Overlay => "Overlay",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    TopLeft,
    Centre,
    CentreLeft,
    CentreRight,
    TopCentre,
    TopRight,
    BottomLeft,
    BottomCentre,
    BottomRight,
    Custom,
}

impl Origin {
    pub fn from_token(tok: &str) -> Option<Origin> {
        match tok.trim().to_ascii_lowercase().as_str() {
            "topleft" | "0" => Some(Origin::TopLeft),
            "centre" | "center" | "1" => Some(Origin::Centre),
            "centreleft" | "centerleft" | "2" => Some(Origin::CentreLeft),
            "centreright" | "centerright" => Some(Origin::CentreRight),
            "topcentre" | "topcenter" | "3" => Some(Origin::TopCentre),
            "topright" | "4" => Some(Origin::TopRight),
            "bottomleft" | "5" => Some(Origin::BottomLeft),
            "bottomcentre" | "bottomcenter" | "6" => Some(Origin::BottomCentre),
            "bottomright" | "7" => Some(Origin::BottomRight),
            "custom" | "8" => Some(Origin::Custom),
            _ => None,
        }
    }

    /// 锚点在精灵内的比例位置（0..1），旋转与翻转围绕该点进行。
    pub fn anchor(self) -> [f32; 2] {
        match self {
            Origin::TopLeft => [0.0, 0.0],
            Origin::Centre | Origin::Custom => [0.5, 0.5],
            Origin::CentreLeft => [0.0, 0.5],
            Origin::CentreRight => [1.0, 0.5],
            Origin::TopCentre => [0.5, 0.0],
            Origin::TopRight => [1.0, 0.0],
            Origin::BottomLeft => [0.0, 1.0],
            Origin::BottomCentre => [0.5, 1.0],
            Origin::BottomRight => [1.0, 1.0],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopType {
    LoopForever,
    LoopOnce,
}

impl LoopType {
    pub fn from_token(tok: &str) -> LoopType {
        match tok.trim().to_ascii_lowercase().as_str() {
            "looponce" => LoopType::LoopOnce,
            _ => LoopType::LoopForever,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parameter {
    FlipHorizontal,
    FlipVertical,
    AdditiveBlending,
}

/// 单条 storyboard 命令携带的效果与起止值。
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Effect {
    Fade { from: f32, to: f32 },
    Move { from: [f32; 2], to: [f32; 2] },
    MoveX { from: f32, to: f32 },
    MoveY { from: f32, to: f32 },
    Scale { from: f32, to: f32 },
    VecScale { from: [f32; 2], to: [f32; 2] },
    Rotate { from: f32, to: f32 }, // 弧度
    Colour { from: [f32; 3], to: [f32; 3] },
    Parameter(Parameter),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Command {
    pub easing: Easing,
    pub effect: Effect,
    pub start_time: f32,
    pub end_time: f32,
}

impl Command {
    /// 该命令对各通道贡献的 (起始值, 终止值)；未涉及返回 None。
    pub fn alpha(&self) -> Option<(f32, f32)> {
        match self.effect {
            Effect::Fade { from, to } => Some((from, to)),
            _ => None,
        }
    }
    pub fn x(&self) -> Option<(f32, f32)> {
        match self.effect {
            Effect::Move { from, to } => Some((from[0], to[0])),
            Effect::MoveX { from, to } => Some((from, to)),
            _ => None,
        }
    }
    pub fn y(&self) -> Option<(f32, f32)> {
        match self.effect {
            Effect::Move { from, to } => Some((from[1], to[1])),
            Effect::MoveY { from, to } => Some((from, to)),
            _ => None,
        }
    }
    pub fn scale_x(&self) -> Option<(f32, f32)> {
        match self.effect {
            Effect::Scale { from, to } => Some((from, to)),
            Effect::VecScale { from, to } => Some((from[0], to[0])),
            _ => None,
        }
    }
    pub fn scale_y(&self) -> Option<(f32, f32)> {
        match self.effect {
            Effect::Scale { from, to } => Some((from, to)),
            Effect::VecScale { from, to } => Some((from[1], to[1])),
            _ => None,
        }
    }
    pub fn rotation(&self) -> Option<(f32, f32)> {
        match self.effect {
            Effect::Rotate { from, to } => Some((from, to)),
            _ => None,
        }
    }
    pub fn colour(&self) -> Option<([f32; 3], [f32; 3])> {
        match self.effect {
            Effect::Colour { from, to } => Some((from, to)),
            _ => None,
        }
    }
    pub fn parameter(&self) -> Option<Parameter> {
        match self.effect {
            Effect::Parameter(p) => Some(p),
            _ => None,
        }
    }
}

/// `L,startTime,totalIterations` 命令循环；子命令时间为相对循环起始的毫秒。
#[derive(Debug, Clone)]
pub struct CommandLoop {
    pub start_time: f32,
    pub total_iterations: u32,
    pub commands: Vec<Command>,
}

/// `T,triggerName,startTime,endTime` 命令触发器；仅在游戏事件触发时激活。
/// 本渲染器解析并保留其信息（--list 可见数量），但不自动激活。
#[derive(Debug, Clone)]
#[allow(dead_code)] // 字段保留用于未来手动激活触发器
pub struct CommandTrigger {
    pub trigger_name: String,
    pub start_time: f32,
    pub end_time: f32,
    pub total_iterations: u32,
    pub commands: Vec<Command>,
}

#[derive(Debug, Clone)]
pub struct Sprite {
    pub layer: Layer,
    pub origin: Origin,
    pub path: String,
    pub x: f32,
    pub y: f32,
    /// 旧版背景行（`0,0,"bg.jpg",x,y`）生成的常驻精灵。
    pub always_visible: bool,
    pub commands: Vec<Command>,
    pub loops: Vec<CommandLoop>,
    pub triggers: Vec<CommandTrigger>,
}

#[derive(Debug, Clone)]
pub struct Animation {
    pub base: Sprite,
    pub frame_count: u32,
    pub frame_delay: f32,
    pub loop_type: LoopType,
}

#[derive(Debug, Clone)]
pub enum Element {
    Sprite(Sprite),
    Animation(Animation),
}

impl Element {
    pub fn sprite(&self) -> &Sprite {
        match self {
            Element::Sprite(s) => s,
            Element::Animation(a) => &a.base,
        }
    }
}

/// 视频/音频元素只解析、不渲染。
#[derive(Debug, Clone)]
#[allow(dead_code)] // 信息保留给 --list / 未来按需使用
pub struct Video {
    pub start_time: f32,
    pub path: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // 信息保留给 --list / 未来按需使用
pub struct Sample {
    pub time: f32,
    pub layer: i32,
    pub path: String,
}

#[derive(Debug, Clone, Default)]
pub struct Storyboard {
    pub elements: Vec<Element>,
    pub videos: Vec<Video>,
    pub samples: Vec<Sample>,
    /// .osu [General] 的 WidescreenStoryboard；None = 未知（纯 .osb），按宽屏处理。
    pub widescreen: Option<bool>,
    /// 解析期间跳过/修正的内容（数量多时只打印前几条）。
    pub warnings: Vec<String>,
}
