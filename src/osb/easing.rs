//! osu! storyboard 缓动函数，编号与 osu!framework `Easing` 一致（0..=32）。

use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Easing(pub i32);

impl Easing {
    pub const LINEAR: Easing = Easing(0);

    /// 未知的缓动 id 回退为线性（osu! 稳定版行为近似）。
    pub fn from_id(id: i32) -> Easing {
        if (0..=32).contains(&id) {
            Easing(id)
        } else {
            Easing(0)
        }
    }

    /// 输入 clamp 到 [0,1]，输出为插值进度。
    pub fn apply(self, time: f32) -> f32 {
        let t = time.clamp(0.0, 1.0);
        match self.0 {
            0 => t,
            1 => out_quad(t),          // Easing.Out（= OutQuad）
            2 => t * t,                // Easing.In（= InQuad）
            3 => t * t,                // InQuad
            4 => out_quad(t),          // OutQuad
            5 => in_out_quad(t),       // InOutQuad
            6 => t * t * t,            // InCubic
            7 => 1.0 - (1.0 - t).powi(3),
            8 => in_out(t, t * t * t, 1.0 - (1.0 - t).powi(3)),
            9 => t * t * t * t,
            10 => 1.0 - (1.0 - t).powi(4),
            11 => in_out(t, t.powi(4), 1.0 - (1.0 - t).powi(4)),
            12 => t.powi(5),
            13 => 1.0 - (1.0 - t).powi(5),
            14 => in_out(t, t.powi(5), 1.0 - (1.0 - t).powi(5)),
            15 => 1.0 - (t * PI / 2.0).cos(),
            16 => (t * PI / 2.0).sin(),
            17 => -(t * PI).cos() / 2.0 + 0.5,
            18 => {
                if t == 0.0 {
                    0.0
                } else {
                    2.0f32.powf(10.0 * t - 10.0)
                }
            }
            19 => {
                if t == 1.0 {
                    1.0
                } else {
                    1.0 - 2.0f32.powf(-10.0 * t)
                }
            }
            20 => {
                if t == 0.0 {
                    0.0
                } else if t == 1.0 {
                    1.0
                } else if t < 0.5 {
                    2.0f32.powf(20.0 * t - 10.0) / 2.0
                } else {
                    (2.0 - 2.0f32.powf(-20.0 * t + 10.0)) / 2.0
                }
            }
            21 => 1.0 - (1.0 - t * t).sqrt(),
            22 => (1.0 - (t - 1.0) * (t - 1.0)).sqrt(),
            23 => {
                let u = 2.0 * t;
                if t < 0.5 {
                    (1.0 - (1.0 - u * u).sqrt()) / 2.0
                } else {
                    let v = -u + 2.0;
                    ((1.0 - v * v).sqrt() + 1.0) / 2.0
                }
            }
            24 => elastic_in(t),
            25 => elastic_out(t),
            26 => {
                if t == 0.0 || t == 1.0 {
                    t
                } else if t < 0.5 {
                    -(2.0f32.powf(20.0 * t - 10.0) * ((20.0 * t - 11.125) * (2.0 * PI / 3.0)).sin()) / 2.0
                } else {
                    2.0f32.powf(-20.0 * t + 10.0) * ((20.0 * t - 11.125) * (2.0 * PI / 3.0)).sin() / 2.0 + 1.0
                }
            }
            27 => back_in(t),
            28 => back_out(t),
            29 => {
                if t < 0.5 {
                    ((2.0 * t).powi(2) * ((BACK_C3 + 1.0) * 2.0 * t - BACK_C3)) / 2.0
                } else {
                    ((2.0 * t - 2.0).powi(2) * ((BACK_C3 + 1.0) * (2.0 * t - 2.0) + BACK_C3)) / 2.0 + 1.0
                }
            }
            30 => 1.0 - bounce_out(1.0 - t),
            31 => bounce_out(t),
            32 => {
                if t < 0.5 {
                    (1.0 - bounce_out(1.0 - 2.0 * t)) / 2.0
                } else {
                    (1.0 + bounce_out(2.0 * t - 1.0)) / 2.0
                }
            }
            _ => t,
        }
    }
}

const BACK_C1: f32 = 1.70158;
const BACK_C3: f32 = BACK_C1 + 1.0;

fn out_quad(t: f32) -> f32 {
    1.0 - (1.0 - t) * (1.0 - t)
}

fn in_out_quad(t: f32) -> f32 {
    if t < 0.5 {
        2.0 * t * t
    } else {
        1.0 - (-2.0 * t + 2.0).powi(2) / 2.0
    }
}

fn in_out(t: f32, a: f32, b: f32) -> f32 {
    if t < 0.5 {
        a / 2.0
    } else {
        b / 2.0 + 0.5
    }
}

fn elastic_in(t: f32) -> f32 {
    if t == 0.0 || t == 1.0 {
        t
    } else {
        -(2.0f32.powf(10.0 * t - 10.0) * ((10.0 * t - 10.75) * (2.0 * PI / 3.0)).sin())
    }
}

fn elastic_out(t: f32) -> f32 {
    if t == 0.0 || t == 1.0 {
        t
    } else {
        2.0f32.powf(-10.0 * t) * ((10.0 * t - 0.75) * (2.0 * PI / 3.0)).sin() + 1.0
    }
}

fn back_in(t: f32) -> f32 {
    BACK_C3 * t * t * t - BACK_C1 * t * t
}

fn back_out(t: f32) -> f32 {
    let u = t - 1.0;
    1.0 + BACK_C3 * u * u * u + BACK_C1 * u * u
}

fn bounce_out(t: f32) -> f32 {
    const N1: f32 = 7.5625;
    const D1: f32 = 2.75;
    if t < 1.0 / D1 {
        N1 * t * t
    } else if t < 2.0 / D1 {
        let u = t - 1.5 / D1;
        N1 * u * u + 0.75
    } else if t < 2.5 / D1 {
        let u = t - 2.25 / D1;
        N1 * u * u + 0.9375
    } else {
        let u = t - 2.625 / D1;
        N1 * u * u + 0.984375
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoints_are_0_and_1() {
        for id in 0..=32 {
            let e = Easing(id);
            let a = e.apply(0.0);
            let b = e.apply(1.0);
            assert!((a - 0.0).abs() < 1e-4, "easing {id} at 0 -> {a}");
            assert!((b - 1.0).abs() < 1e-4, "easing {id} at 1 -> {b}");
        }
    }

    #[test]
    fn clamps_input() {
        assert_eq!(Easing::LINEAR.apply(-1.0), 0.0);
        assert_eq!(Easing::LINEAR.apply(2.0), 1.0);
    }

    #[test]
    fn known_values() {
        let e = Easing(3); // InQuad
        assert!((e.apply(0.5) - 0.25).abs() < 1e-5);
        let e = Easing(4); // OutQuad
        assert!((e.apply(0.5) - 0.75).abs() < 1e-5);
        let e = Easing(0);
        assert!((e.apply(0.25) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn unknown_id_falls_back_to_linear() {
        assert_eq!(Easing::from_id(99), Easing::LINEAR);
        assert_eq!(Easing::from_id(-3), Easing::LINEAR);
    }
}
