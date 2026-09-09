struct Globals {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals: Globals;

@group(1) @binding(0) var sprite_tex: texture_2d<f32>;
@group(1) @binding(1) var sprite_sampler: sampler;

struct VertexIn {
    @location(0) corner: vec2<f32>, // 单位四边形角点 {0,1}²
    @location(1) pos: vec2<f32>,    // osu! 坐标（640x480 空间）
    @location(2) size: vec2<f32>,   // 缩放后的显示尺寸（osu! 像素）
    @location(3) anchor: vec2<f32>, // 锚点在精灵内的比例位置
    @location(4) rotation: f32,
    @location(5) color: vec4<f32>,  // rgb 染色 + alpha
    @location(6) flip: vec2<f32>,   // 0/1，沿锚点镜像
};

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(v: VertexIn) -> VertexOut {
    let px = v.corner * v.size;
    let anchor_px = v.anchor * v.size;
    let d = px - anchor_px;
    // osu! 正角度为顺时针（y 向下坐标系下的标准旋转矩阵）
    let c = cos(v.rotation);
    let s = sin(v.rotation);
    let dr = vec2<f32>(d.x * c - d.y * s, d.x * s + d.y * c);
    let world = v.pos + dr;
    // flip 语义(lazer DrawableStoryboardSprite: DrawScale 取负 + AdjustOrigin
    // 把 origin 翻到对侧边缘)——两者抵消:四边形保持原位不动,仅纹理在
    // 矩形内整幅镜像。因此几何不镜像、UV 按轴整幅镜像。若几何也镜像,
    // 精灵会跳到 origin 另一侧(My Love 黑幕面板"位置错误"的根因);若 UV
    // 绕 anchor 镜像,边缘锚点(TopCentre 等)UV 越界 [1,2] 被钳到边缘,
    // 整张精灵塌成一条拉伸的边缘纹素(实心色块,柔边全失)。
    var uv = px / v.size;
    uv = mix(uv, 1.0 - uv, v.flip);
    var out: VertexOut;
    out.position = globals.mvp * vec4<f32>(world, 0.0, 1.0);
    out.uv = uv;
    out.color = v.color;
    return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    // 贴图已预乘(RGB 含自身 alpha);color.a 是淡入淡出系数。
    // 预乘结果 = texel_premult.rgb × color.rgb × color.a,
    // 配合 One/OneMinusSrcAlpha 混合,透明边界无黑色插值伪影。
    let texel = textureSample(sprite_tex, sprite_sampler, in.uv);
    let a = texel.a * in.color.a;
    return vec4<f32>(texel.rgb * in.color.rgb * in.color.a, a);
}
