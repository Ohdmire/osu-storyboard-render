# osu-storyboard-render

从 storyboard **源代码**（`.osb` 文件或 `.osu` 的 `[Events]` 节）直接渲染 osu! storyboard 的工具，渲染后端为 **wgpu**（Rust）。

不依赖 osu! 客户端：自己解析精灵/命令/循环/触发器源码、展开时间轴，然后用 GPU 实时绘制。

```
cargo run --release -- --demo                      # 播放内置 demo
cargo run --release -- "Songs/xxx/xxx.osb"         # 播放真实 storyboard
cargo run --release -- "beatmap.osz"               # .osz 自动解包（临时目录）
cargo run --release -- --screenshot 30000 out.png "xxx.osz"   # 无窗口渲染一帧
cargo run --release -- --list "xxx.osu"            # 打印解析摘要（含内嵌 Events 的 .osu）
```

## 功能

### 已支持

- **输入**：`.osb`、`.osu`（Events 内嵌 storyboard）、`.osz`（自动解包到临时目录，渲染 **所选难度 `.osu` 的 Events + 共用 `.osb`** 的合并结果，与 osu! 一致；`--diff` 选择难度）；UTF-8 BOM / `osu file format` 头行 / CRLF 均可处理
- **元素**：`Sprite`、`Animation`（`LoopForever` / `LoopOnce`）、旧版数字背景行（`0,0,"bg.jpg",x,y`）
- **命令**：`F`(Fade)、`M`/`MX`/`MY`(Move)、`S`(Scale)、`V`(VecScale)、`R`(Rotate)、`C`(Colour)、`P`(Parameter: `H`/`V` 翻转与 `A` 加色混合)
- **命令组**：`L` 循环（按最长子命令时长展开迭代）、`T` 触发器（解析保留，见下）
- **缓动**：全部 33 种（0..=32），公式与 osu!framework 对齐
- **层与原点**：Background/Fail/Pass/Foreground/Overlay（注释区块与行内显式层两种写法）、10 种 origin 锚点（含 `CentreRight`），旋转/翻转围绕锚点
- **渲染语义**：画家算法（层序 + 文件序）、命令窗口外元素不渲染、重叠命令后启动者生效、值在命令结束后保持、**通道首条命令开始前取其起始值**（lazer `ApplyInitialValue` 语义，Sprite 行 x/y 仅在该通道无命令时生效）、加色混合管线
- **坐标系**：640×480 osu! 虚拟分辨率；读取 `.osu` 的 `WidescreenStoryboard` 标志——宽屏谱面高固定 480、宽随窗口纵横比向两侧扩展，非宽屏谱面固定 4:3（宽窗口两侧黑边），与 osu! 行为一致。`.osz` 里选了 `.osb` 时会从包内 `.osu` 读取该标志
- **素材**：png/jpeg/bmp/tga/webp；路径反斜杠归一化、无扩展名时自动补 `.png`/`.jpg`/`.jpeg`、`sb/` 子目录回退、大小写不敏感回退（应对社区谱面的大小写混乱）
- **运行模式**：winit 窗口实时播放（暂停/快进/调速/重播）、离屏渲染任意时刻为 PNG（无需显示服务器，可用于 CI / 批量出图）

### 不渲染（仅解析并在 `--list` 中计数）

- `Video` / 旧版视频行 —— 需要视频解码器，超出本项目范围
- `Sample` 音效 —— 无音频输出
- `T` 触发器组 —— 触发依赖游戏事件（打击音、Pass/Fail 切换）；渲染器没有游戏状态，故不自动激活。Pass/Fail 层可用 `--fail` 手动切换

## 播放控制

| 按键 | 功能 |
|---|---|
| `Space` | 暂停 / 继续 |
| `←` / `→` | 快退 / 快进 1s（Shift：5s） |
| `R` | 重新播放 |
| `-` / `=` | 0.8× / 1.25× 调速 |
| `Esc` | 退出 |

标题栏实时显示当前时间与倍速。

## 命令行

```
osu-storyboard-render [选项] [storyboard.osb | beatmap.osu | beatmap.osz]

--demo                    内置 demo（未指定文件时的默认值）
--assets <DIR>            素材目录（默认: storyboard 文件所在目录）
--diff <NAME>             打开 .osz 时选择难度（文件名子串匹配，默认包内第一个）
--width <N> --height <N>  窗口/截图尺寸（默认 1600x900；非宽屏谱面自动 4:3 黑边）
--start <MS>              起始时间（毫秒）
--speed <X>               播放倍速
--fail                    显示 Fail 层（默认 Pass 层）
--screenshot <MS> <FILE>  无窗口渲染一帧保存为 PNG 后退出
--list                    解析并打印摘要后退出
```

环境变量 `RUST_LOG=info`（默认）输出 GPU 适配器、可见精灵数、缺失贴图等日志。

## 结构

```
src/
├── main.rs              CLI 与运行模式分发
├── osb/                 storyboard 源码解析
│   ├── parser.rs        [Events] 行级解析（元素/命令/循环/触发器/引号路径）
│   ├── model.rs         数据模型（层、原点、命令效果、元素）
│   ├── easing.rs        33 种缓动
│   └── timeline.rs      循环展开 + 按通道采样求值 state_at(t)
├── render/
│   ├── gpu.rs           wgpu Instance/Adapter/Device
│   ├── texture.rs       贴图加载与路径回退
│   ├── renderer.rs      精灵管线（普通/加色混合）、实例化绘制、build_draws
│   ├── shader.wgsl      顶点变换（锚点/旋转/翻转）+ 采样
│   └── demo.rs          内置 demo（源码 + 程序生成贴图）
├── player.rs            winit 播放器
└── screenshot.rs        离屏渲染 → PNG
```

渲染路径：解析 → 编译（`L` 展开为绝对时间命令，按通道分桶排序）→ 每帧 `state_at(t)` 求值 → 生成实例（位置/尺寸/锚点/旋转/颜色/翻转）→ 单顶点缓冲 + 实例缓冲逐精灵绘制。

## 已验证

- 30 个单元测试（解析、循环组缩进边界、缓动端点、循环展开、通道组合、通道初始值、层排序、投影矩阵、demo 自检）
- 真实谱面（Songs 目录的 `Meaning`、`HAPPY PARTY TRAIN` 等 .osb，上千元素、5 万+ 命令）渲染结果与手工核对的可见性窗口一致

## 依赖

wgpu 24 / winit 0.30 / image 0.25，Linux（Vulkan/GL）、Windows、macOS 均可运行。
