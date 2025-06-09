use glam::Vec2;

// FIXME: AI generated \o/
pub(crate) fn resolve_disk_collision(
    pos1: Vec2,
    vel1: Vec2,
    mass1: f32,
    pos2: Vec2,
    vel2: Vec2,
    mass2: f32,
) -> (Vec2, Vec2) {
    let delta_pos = pos1 - pos2;
    let delta_vel = vel1 - vel2;
    let dist2 = delta_pos.length_squared();

    // Avoid division by zero (shouldn't happen if disks are colliding)
    if dist2 == 0.0 {
        return (vel1, vel2);
    }

    let mass_sum = mass1 + mass2;

    let v1_proj = delta_vel.dot(delta_pos) / dist2;
    let v1_new = vel1 - (2.0 * mass2 / mass_sum) * v1_proj * delta_pos;

    let delta_pos2 = pos2 - pos1;
    let delta_vel2 = vel2 - vel1;
    let v2_proj = delta_vel2.dot(delta_pos2) / dist2;
    let v2_new = vel2 - (2.0 * mass1 / mass_sum) * v2_proj * delta_pos2;

    (v1_new, v2_new)
}

