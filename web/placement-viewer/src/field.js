import { eventPosition } from './data.js';

// Isotropic Gaussian KDE sampled in metric space, truncated at 3 sigma.
// A bounded grid keeps scattered/large datasets from allocating huge textures.
export function densityField(events, frame, bandwidth) {
  if (!Number.isFinite(bandwidth) || bandwidth <= 0) throw new Error('平滑尺度必须为正数');
  if (!events.length) return null;
  const points = events.map((event) => eventPosition(event, frame));
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (const point of points) point.forEach((v, axis) => {
    min[axis] = Math.min(min[axis], v - 3 * bandwidth);
    max[axis] = Math.max(max[axis], v + 3 * bandwidth);
  });
  const span = max.map((v, axis) => v - min[axis]);
  const step = Math.max(bandwidth / 2, ...span.map((v) => v / 63));
  const dimensions = span.map((v) => Math.max(2, Math.ceil(v / step) + 1));
  const size = dimensions.map((n) => (n - 1) * step);
  const values = new Float32Array(dimensions.reduce((a, b) => a * b, 1));
  // Aggregate repeated points before evaluating the kernel.
  const unique = new Map();
  for (const point of points) {
    const key = point.join(',');
    const entry = unique.get(key) ?? { point, count: 0 };
    entry.count++;
    unique.set(key, entry);
  }
  let peak = 0;
  for (const { point, count } of unique.values()) {
    const low = point.map((v, axis) => Math.max(0, Math.ceil((v - 3 * bandwidth - min[axis]) / step)));
    const high = point.map((v, axis) => Math.min(dimensions[axis] - 1, Math.floor((v + 3 * bandwidth - min[axis]) / step)));
    for (let z = low[2]; z <= high[2]; z++) for (let y = low[1]; y <= high[1]; y++) for (let x = low[0]; x <= high[0]; x++) {
      const distance2 = [x, y, z].reduce((sum, v, axis) => sum + ((min[axis] + v * step - point[axis]) / bandwidth) ** 2, 0);
      if (distance2 > 9) continue;
      const index = x + dimensions[0] * (y + dimensions[1] * z);
      values[index] += count * Math.exp(-distance2 / 2);
      peak = Math.max(peak, values[index]);
    }
  }
  if (!peak) return null;
  for (let i = 0; i < values.length; i++) values[i] /= peak;
  return { values, dimensions, min, size, bandwidth, peak, step };
}

export const fieldVertexShader = `
  out vec3 localPosition;
  void main() {
    localPosition = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const fieldFragmentShader = `
  precision highp sampler3D;
  uniform sampler3D densityMap;
  uniform vec3 cameraLocal;
  uniform vec3 dimensions;
  in vec3 localPosition;
  out vec4 outputColor;

  vec3 heatColor(float t) {
    vec3 blue = vec3(0.08, 0.30, 0.80);
    vec3 teal = vec3(0.08, 0.80, 0.65);
    vec3 yellow = vec3(1.0, 0.85, 0.22);
    vec3 red = vec3(1.0, 0.22, 0.08);
    return t < 0.33 ? mix(blue, teal, t / 0.33)
      : t < 0.66 ? mix(teal, yellow, (t - 0.33) / 0.33)
      : mix(yellow, red, (t - 0.66) / 0.34);
  }

  void main() {
    vec3 direction = normalize(localPosition - cameraLocal);
    vec3 safeDirection = mix(vec3(0.000001), direction, greaterThan(abs(direction), vec3(0.000001)));
    vec3 t0 = (-0.5 - cameraLocal) / safeDirection;
    vec3 t1 = (0.5 - cameraLocal) / safeDirection;
    vec3 near = min(t0, t1), far = max(t0, t1);
    float entry = max(0.0, max(near.x, max(near.y, near.z)));
    float exit = min(far.x, min(far.y, far.z));
    if (entry >= exit) discard;
    float dt = (exit - entry) / 128.0;
    // Maximum intensity projection: color AND opacity encode the highest
    // local KDE density on this ray. Alpha compositing made long sparse
    // columns look stronger than compact dense clusters.
    float peakDensity = 0.0;
    for (int i = 0; i < 128; i++) {
      vec3 p = cameraLocal + direction * (entry + (float(i) + 0.5) * dt) + 0.5;
      // Grid endpoints map to texel centers, preserving the metric bounds.
      vec3 uvw = (p * (dimensions - 1.0) + 0.5) / dimensions;
      float value = texture(densityMap, uvw).r;
      peakDensity = max(peakDensity, value);
    }
    if (peakDensity < 0.015) discard;
    outputColor = vec4(heatColor(peakDensity), 0.2 + 0.7 * peakDensity);
    outputColor = linearToOutputTexel(outputColor);
  }
`;
