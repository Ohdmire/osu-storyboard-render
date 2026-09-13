// Bilinear fit scaler (compute): samples the source at the destination
// pixel center and writes it out. Final fractional-scale step of the
// Anime4K chain (CNN output is 2x; the composite target rarely is).

struct Params {
    src_size: vec2<f32>,
    dst_size: vec2<f32>,
}

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var dst_tex: texture_storage_2d<rgba8unorm, write>;
@group(1) @binding(0) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let dst = vec2<f32>(params.dst_size);
    if (global_id.x >= u32(params.dst_size.x) || global_id.y >= u32(params.dst_size.y)) {
        return;
    }
    // 像素中心 → 源 UV(线性过滤即双线性)
    let uv = (vec2<f32>(global_id.xy) + 0.5) / dst;
    let c = textureSampleLevel(src_tex, src_sampler, uv, 0.0);
    textureStore(dst_tex, vec2<i32>(global_id.xy), c);
}
