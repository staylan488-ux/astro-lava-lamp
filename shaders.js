// ============================================================================
// LavaLamp WebGL 2 Shaders (GLSL ES 3.00)
// 1. 3D Volume Builder (80x192x80 RGBA16F): 8D Orthoplex Vector-Metaballs
// 2. Hybrid Analytical + Volume Raymarcher with Exact Analytical Hit Normals,
//    Beer-Lambert Subsurface Scattering, Meniscus TIR, Helical Coil, Thick
//    Borosilicate Glass Light-Piping, & Anisotropic Brushed Metal
// 3. Multi-Mip Dual-Kawase HDR Bloom + ACES Filmic Composite + Dithering
// ============================================================================

(function () {
  'use strict';

  const FULLSCREEN_VERT = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  vUv = pos;
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
`;

  // Shared geometry & parcel evaluation library injected into both Volume & Scene shaders
  const COMMON_WAX_GLSL = `
#define NUM_PARCELS 52
#define VOLUME_MIN vec3(-0.86, 1.24, -0.86)
#define VOLUME_MAX vec3( 0.86, 4.70,  0.86)

layout(std140) uniform ParcelBlock {
  vec4 uParcelData[224]; // 56 * 4 vec4s: [posRad, defTemp, phaseA, phaseB]
};

vec3 safeNormalize(vec3 v) {
  float l2 = dot(v, v);
  return l2 > 1e-10 ? v * inversesqrt(l2) : vec3(0.0, 1.0, 0.0);
}

vec3 sanitizeHdr(vec3 c) {
  if (any(isnan(c)) || any(isinf(c))) return vec3(0.0);
  return clamp(c, vec3(0.0), vec3(32.0));
}

float glassOuterRadius(float y) {
  float v = y - 1.80;
  float dg = 0.18;
  float sq = sqrt(v * v + dg * dg);
  float vp = 0.5 * (sq + v - dg);
  float vm = 0.5 * (sq - v - dg);
  float b = max(0.0, 1.46 - y);
  return 0.838 - 0.130 * vp - 0.195 * vm - 0.55 * b * b;
}

float glassOuterSlope(float y) {
  float v = y - 1.80;
  float dg = 0.18;
  float dSq = v / sqrt(v * v + dg * dg);
  float dvp = 0.5 * (dSq + 1.0);
  float dvm = 0.5 * (dSq - 1.0);
  float b = max(0.0, 1.46 - y);
  float db = (y < 1.46) ? -1.0 : 0.0;
  return -0.130 * dvp - 0.195 * dvm - 1.10 * b * db;
}

float cavityRadius(float y) {
  return glassOuterRadius(y) - 0.044;
}

float meniscusHeight(float r, float cavR) {
  float q = clamp(r / max(cavR, 1e-4), 0.0, 1.0);
  float q2 = q * q;
  return 4.585 + 0.025 * q2 * q2;
}

// Smooth C1 boundary containment dampener so wax cushions flat against inner glass & meniscus
float evalCavityBoundary(vec3 p, out vec3 gradB) {
  float r = length(p.xz);
  float cavR = cavityRadius(p.y);
  float slopeR = glassOuterSlope(p.y);
  float menY = meniscusHeight(r, cavR);

  float hCap = 0.062;

  float dRad = cavR - 0.008 - r;
  float dFlr = p.y - 1.286;
  float dTop = menY - 0.010 - p.y;

  float tRad = clamp(dRad / hCap, 0.0, 1.0);
  float tFlr = clamp(dFlr / hCap, 0.0, 1.0);
  float tTop = clamp(dTop / hCap, 0.0, 1.0);

  float bRad = tRad * tRad * (3.0 - 2.0 * tRad);
  float bFlr = tFlr * tFlr * (3.0 - 2.0 * tFlr);
  float bTop = tTop * tTop * (3.0 - 2.0 * tTop);

  float dbRad = (tRad > 0.0 && tRad < 1.0) ? (6.0 * tRad * (1.0 - tRad) / hCap) : 0.0;
  float dbFlr = (tFlr > 0.0 && tFlr < 1.0) ? (6.0 * tFlr * (1.0 - tFlr) / hCap) : 0.0;
  float dbTop = (tTop > 0.0 && tTop < 1.0) ? (6.0 * tTop * (1.0 - tTop) / hCap) : 0.0;

  vec3 rHat = (r > 1e-5) ? vec3(p.x / r, 0.0, p.z / r) : vec3(1.0, 0.0, 0.0);
  vec3 gRad = vec3(0.0, slopeR, 0.0) - rHat;
  vec3 gFlr = vec3(0.0, 1.0, 0.0);
  float q = clamp(r / max(cavR, 1e-4), 0.0, 1.0);
  float dMenDr = 0.10 * q * q * q / max(cavR, 1e-4);
  vec3 gTop = rHat * dMenDr - vec3(0.0, 1.0, 0.0);

  gradB = (dbRad * bFlr * bTop) * gRad
        + (bRad * dbFlr * bTop) * gFlr
        + (bRad * bFlr * dbTop) * gTop;

  return bRad * bFlr * bTop;
}

float evalCavityBoundaryFast(vec3 p) {
  float r = length(p.xz);
  float cavR = cavityRadius(p.y);
  float menY = meniscusHeight(r, cavR);
  float hCap = 0.062;
  float bRad = smoothstep(0.0, hCap, cavR - 0.008 - r);
  float bFlr = smoothstep(0.0, hCap, p.y - 1.286);
  float bTop = smoothstep(0.0, hCap, menY - 0.010 - p.y);
  return bRad * bFlr * bTop;
}
`;

  // ==========================================================================
  // PASS 1: 3D Scalar Volume Slice Shader (80 x 192 x 80 RGBA16F)
  // Evaluates 8D Orthoplex Vector-Metaballs across the bottle interior.
  // Output:
  //   R = Bounded Wax Field Phi(x) (surface is at Phi == 1.0)
  //   G = Weighted Wax Temperature T(x)
  //   B = Solidification factor s(x)
  //   A = Raw Weight Sum S0(x) (non-zero anywhere inside parcel support halos!)
  // ==========================================================================
  const VOLUME_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;

uniform float uSliceW; // Normalized Z coordinate [0..1] for current 3D slice
uniform float uSimTime;

${COMMON_WAX_GLSL}

void main() {
  vec3 uvw = vec3(vUv.x, vUv.y, uSliceW);
  vec3 p = mix(VOLUME_MIN, VOLUME_MAX, uvw);

  // Fast reject outside circular liquid cavity
  float rHoriz = length(p.xz);
  float cavR = cavityRadius(p.y);
  if (rHoriz > cavR + 0.02 || p.y < 1.26 || p.y > 4.65) {
    fragColor = vec4(0.0, 0.45, 0.0, 0.0);
    return;
  }

  float bWall = evalCavityBoundaryFast(p);
  if (bWall <= 1e-5) {
    fragColor = vec4(0.0, 0.45, 0.0, 0.0);
    return;
  }

  float S0 = 0.0;
  vec4 Va = vec4(0.0);
  vec4 Vb = vec4(0.0);
  float weightedTemp = 0.0;

  for (int i = 0; i < NUM_PARCELS; i++) {
    int base = i * 4;
    vec4 posRad = uParcelData[base];
    vec3 d = p - posRad.xyz;
    float R = 2.0 * posRad.w; // Compact support = 2 * radius
    // Quick bounding box / sphere reject before deformation math
    float maxR = R * 1.28;
    float rawDSq = dot(d, d);
    if (rawDSq >= maxR * maxR) continue;

    vec4 defTemp = uParcelData[base + 1];
    vec3 def = defTemp.xyz;
    float e = length(def);
    vec3 dTilde = d;
    if (e > 1e-4) {
      vec3 axis = def / e;
      // Volume-preserving stretch/squash
      float lamPar = 1.0 / (1.0 + e);
      float lamPerp = sqrt(1.0 + e);
      dTilde = lamPerp * d + (lamPar - lamPerp) * dot(d, axis) * axis;
    }

    float qSq = dot(dTilde, dTilde) / (R * R);
    if (qSq >= 1.0) continue;

    float omq = 1.0 - qSq;
    float w = (64.0 / 27.0) * omq * omq * omq;

    S0 += w;
    Va += w * uParcelData[base + 2];
    Vb += w * uParcelData[base + 3];
    weightedTemp += w * defTemp.w;
  }

  if (S0 <= 1e-5) {
    fragColor = vec4(0.0, 0.45, 0.0, 0.0);
    return;
  }

  float S1 = sqrt(dot(Va, Va) + dot(Vb, Vb) + 1e-8) - 1e-4;
  float m = max(0.0, S0 - S1);
  float m2 = m * m;
  float penalty = (4.2 * m2 * m) / (0.08 + m2);
  float phi = max(0.0, (S1 - penalty) * bWall);
  float temp = weightedTemp / S0;
  float solid = clamp((0.31 - temp) / 0.11, 0.0, 1.0);

  fragColor = vec4(phi, temp, solid, S0);
}
`;

  // ==========================================================================
  // PASS 2: Main Full-Physics Photorealistic Raytracing & Volume Shader
  // ==========================================================================
  const SCENE_FRAG = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vUv;
out vec4 fragColor;

uniform sampler3D uWaxVolume;
uniform vec2 uResolution;
uniform vec3 uCamPos;
uniform mat3 uCamMat;
uniform float uFovScale;
uniform vec2 uLensShift;
uniform float uTime;

// Lamp & Optical Parameters
uniform float uBulbGlow;      // [0 .. 1.4]
uniform float uLiquidTemp;    // [0.08 .. 0.70]
uniform vec3 uWaxColor;       // Linear sRGB wax base color
uniform vec3 uLiquidColor;    // Linear sRGB carrier fluid dye tint
uniform float uLiquidClarity; // Dye density [0 .. 1]
uniform int uMetalType;       // 0=Silver, 1=Brass, 2=MatteBlack, 3=Copper
uniform float uRoomLight;     // Ambient studio brightness [0.05 .. 1.0]

${COMMON_WAX_GLSL}

// Sample precomputed 3D wax field (explicit LOD 0 avoids D3D11 FXC gradient-in-loop link errors!)
vec4 sampleWaxGrid(vec3 p) {
  vec3 uvw = (p - VOLUME_MIN) / (VOLUME_MAX - VOLUME_MIN);
  if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {
    return vec4(0.0, 0.45, 0.0, 0.0);
  }
  return textureLod(uWaxVolume, uvw, 0.0);
}

// Exact analytical evaluation + exact analytical gradient ONLY at the single hit point!
// Eliminates 100% of voxel grid artifacts for infinite-resolution glossy normals!
float evalWaxExact(vec3 p, out vec3 outNormal, out float outTemp, out float outSolid) {
  vec3 gradB;
  float bWall = evalCavityBoundary(p, gradB);
  if (bWall <= 1e-6) {
    outNormal = vec3(0.0, 1.0, 0.0);
    outTemp = 0.5;
    outSolid = 0.0;
    return 0.0;
  }

  float S0 = 0.0;
  vec3 dS0 = vec3(0.0);
  vec4 Va = vec4(0.0);
  vec4 Vb = vec4(0.0);

  vec4 dVa_x = vec4(0.0), dVa_y = vec4(0.0), dVa_z = vec4(0.0);
  vec4 dVb_x = vec4(0.0), dVb_y = vec4(0.0), dVb_z = vec4(0.0);

  float weightedTemp = 0.0;

  for (int i = 0; i < NUM_PARCELS; i++) {
    int base = i * 4;
    vec4 posRad = uParcelData[base];
    vec3 d = p - posRad.xyz;
    float R = 2.0 * posRad.w;
    float maxR = R * 1.28;
    if (dot(d, d) >= maxR * maxR) continue;

    vec4 defTemp = uParcelData[base + 1];
    vec3 def = defTemp.xyz;
    float e = length(def);

    vec3 dTilde = d;
    float lamPar = 1.0;
    float lamPerp = 1.0;
    vec3 axis = vec3(0.0, 1.0, 0.0);

    if (e > 1e-4) {
      axis = def / e;
      lamPar = 1.0 / (1.0 + e);
      lamPerp = sqrt(1.0 + e);
      dTilde = lamPerp * d + (lamPar - lamPerp) * dot(d, axis) * axis;
    }

    float invRSq = 1.0 / (R * R);
    float qSq = dot(dTilde, dTilde) * invRSq;
    if (qSq >= 1.0) continue;

    float omq = 1.0 - qSq;
    float omq2 = omq * omq;
    float w = (64.0 / 27.0) * omq2 * omq;

    // Exact analytical gradient of w_i(x) including symmetric stretch operator M_i^T
    vec3 mTd = (e > 1e-4)
      ? (lamPerp * dTilde + (lamPar - lamPerp) * dot(dTilde, axis) * axis)
      : dTilde;
    vec3 gw = (-128.0 / 9.0 * invRSq * omq2) * mTd;

    vec4 uA = uParcelData[base + 2];
    vec4 uB = uParcelData[base + 3];

    S0 += w;
    dS0 += gw;
    Va += w * uA;
    Vb += w * uB;

    dVa_x += gw.x * uA; dVa_y += gw.y * uA; dVa_z += gw.z * uA;
    dVb_x += gw.x * uB; dVb_y += gw.y * uB; dVb_z += gw.z * uB;

    weightedTemp += w * defTemp.w;
  }

  if (S0 <= 1e-6) {
    outNormal = vec3(0.0, 1.0, 0.0);
    outTemp = 0.5;
    outSolid = 0.0;
    return 0.0;
  }

  float lenSq = dot(Va, Va) + dot(Vb, Vb) + 1e-8;
  float invLen = inversesqrt(lenSq);
  float S1 = (1.0 / invLen) - 1e-4;

  vec3 dS1 = invLen * vec3(
    dot(Va, dVa_x) + dot(Vb, dVb_x),
    dot(Va, dVa_y) + dot(Vb, dVb_y),
    dot(Va, dVa_z) + dot(Vb, dVb_z)
  );

  float m = max(0.0, S0 - S1);
  float m2 = m * m;
  float c = 0.08;
  float denomM = c + m2;
  float penalty = (4.2 * m2 * m) / denomM;
  float dPm = (4.2 * m2 * (3.0 * c + m2)) / (denomM * denomM);

  float rawPhi = S1 - penalty;
  vec3 gradRawPhi = (1.0 + dPm) * dS1 - dPm * dS0;

  float phi = rawPhi * bWall;
  vec3 gradPhi = bWall * gradRawPhi + rawPhi * gradB;

  outTemp = weightedTemp / S0;
  outSolid = clamp((0.31 - outTemp) / 0.11, 0.0, 1.0);

  // Subtle waxy micro-crystalline bump when cold & solidified!
  if (outSolid > 0.02) {
    vec3 sp = p * 22.0;
    vec3 micro = vec3(
      sin(sp.x) * cos(sp.y * 1.3 + sp.z),
      cos(sp.y) * sin(sp.z * 1.2 + sp.x),
      sin(sp.z) * cos(sp.x * 1.1 + sp.y)
    );
    gradPhi += micro * (0.045 * outSolid);
  }

  outNormal = safeNormalize(-gradPhi);
  return phi;
}

// Exact scalar evaluation only (faster for secant root polish)
float evalWaxScalarExact(vec3 p) {
  float bWall = evalCavityBoundaryFast(p);
  if (bWall <= 1e-6) return 0.0;
  float S0 = 0.0;
  vec4 Va = vec4(0.0);
  vec4 Vb = vec4(0.0);
  for (int i = 0; i < NUM_PARCELS; i++) {
    int base = i * 4;
    vec4 posRad = uParcelData[base];
    vec3 d = p - posRad.xyz;
    float R = 2.0 * posRad.w;
    float maxR = R * 1.28;
    if (dot(d, d) >= maxR * maxR) continue;
    vec4 defTemp = uParcelData[base + 1];
    vec3 def = defTemp.xyz;
    float e = length(def);
    vec3 dTilde = d;
    if (e > 1e-4) {
      vec3 axis = def / e;
      float lamPar = 1.0 / (1.0 + e);
      float lamPerp = sqrt(1.0 + e);
      dTilde = lamPerp * d + (lamPar - lamPerp) * dot(d, axis) * axis;
    }
    float qSq = dot(dTilde, dTilde) / (R * R);
    if (qSq >= 1.0) continue;
    float omq = 1.0 - qSq;
    float w = (64.0 / 27.0) * omq * omq * omq;
    S0 += w;
    Va += w * uParcelData[base + 2];
    Vb += w * uParcelData[base + 3];
  }
  if (S0 <= 1e-6) return 0.0;
  float S1 = sqrt(dot(Va, Va) + dot(Vb, Vb) + 1e-8) - 1e-4;
  float m = max(0.0, S0 - S1);
  float m2 = m * m;
  float penalty = (4.2 * m2 * m) / (0.08 + m2);
  return (S1 - penalty) * bWall;
}

// ============================================================================
// Rigid Hardware SDFs: Spun Metal Base, Hollow Collar, Top Cap, Glass, & Coil
// ============================================================================

float baseOuterRadius(float y) {
  float u = y - 1.00;
  float d = 0.08;
  float sq = sqrt(u * u + d * d);
  float up = 0.5 * (sq + u - d);
  float um = 0.5 * (sq - u - d);
  return 0.425 + 0.470 * um + 0.710 * up;
}

float sdBox2D(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// Metal Base: solid lower cone [0..1.21] + thin flared collar cup [1.19..1.565]
float sdMetalBase(vec3 p) {
  float r = length(p.xz);
  float rb = baseOuterRadius(clamp(p.y, 0.0, 1.57));

  // Lower enclosed housing [0.0 .. 1.21]
  float dRadLow = (r - rb) * 0.80;
  float dLow = max(dRadLow, max(-p.y, p.y - 1.21));
  // Subtle rubber base foot groove at y = 0.025
  float foot = sdBox2D(vec2(r - 0.43, p.y - 0.015), vec2(0.435, 0.015)) - 0.006;
  dLow = min(dLow, foot);

  // Upper flared cup shell [1.18 .. 1.565]
  float shellWall = abs(r - rb) * 0.80 - 0.013;
  float dCup = max(shellWall, max(1.18 - p.y, p.y - 1.565));
  // Smooth rolled metal lip at y = 1.565
  float lip = length(vec2(r - baseOuterRadius(1.565), p.y - 1.565)) - 0.015;
  dCup = min(dCup, lip);

  return min(dLow, dCup);
}

// Metal Top Cap [4.60 .. 5.41]: solid spun-metal cone with C1 rounded top & bottom rims
float sdMetalCap(vec3 p) {
  float r = length(p.xz);
  float yc = clamp(p.y, 4.60, 5.41);
  float rc = 0.502 - 0.192 * (yc - 4.60);

  vec2 dTop = vec2((r - (rc - 0.018)) * 0.94, p.y - (5.41 - 0.018));
  float body = max(4.605 - p.y, length(max(dTop, 0.0)) + min(max(dTop.x, dTop.y), 0.0) - 0.018);
  float rim = length(vec2(r - (0.502 - 0.008), p.y - 4.605)) - 0.009;
  return min(body, rim);
}

float sdMetalHardware(vec3 p, out int partId) {
  float dBase = sdMetalBase(p);
  float dCap = sdMetalCap(p);
  if (dBase < dCap) {
    partId = 1; // Base
    return dBase;
  }
  partId = 2; // Cap
  return dCap;
}

vec3 calcMetalNormal(vec3 p) {
  const float e = 0.0012;
  int dummy;
  vec2 k = vec2(1.0, -1.0);
  return normalize(
    k.xyy * sdMetalHardware(p + k.xyy * e, dummy) +
    k.yyx * sdMetalHardware(p + k.yyx * e, dummy) +
    k.yxy * sdMetalHardware(p + k.yxy * e, dummy) +
    k.xxx * sdMetalHardware(p + k.xxx * e, dummy)
  );
}

// Outer Borosilicate Glass Bottle [1.23 .. 4.615] (nests cleanly inside metal cap at y >= 4.60)
float sdGlassOuter(vec3 p) {
  float r = length(p.xz);
  float yg = clamp(p.y, 1.23, 4.615);
  float rg = glassOuterRadius(yg);
  float dSide = (r - rg) * 0.86;
  return max(dSide, max(1.23 - p.y, p.y - 4.615));
}

vec3 calcGlassOuterNormal(vec3 p) {
  float r = max(length(p.xz), 1e-5);
  float slope = glassOuterSlope(p.y);
  return normalize(vec3(p.x / r, -slope, p.z / r));
}

// Inner Liquid Cavity [1.28 .. 4.62]
float sdLiquidCavity(vec3 p) {
  float r = length(p.xz);
  float yc = clamp(p.y, 1.28, 4.64);
  float rc = cavityRadius(yc);
  float menY = meniscusHeight(r, rc);
  float dSide = (r - rc) * 0.86;
  return max(dSide, max(1.28 - p.y, p.y - menY));
}

// 32-Turn Helical Stainless Steel Heating Coil nestled in bottom groove (y ~ 1.32)
float sdHeatingCoil(vec3 p) {
  float r = length(p.xz);
  vec2 pol = vec2(r - 0.415, p.y - 1.320);
  float dTorus = length(pol);
  if (dTorus > 0.065) return dTorus - 0.042;

  float theta = atan(p.z, p.x);
  float turns = 32.0;
  float phi = theta * turns;
  vec2 wireCenter = 0.030 * vec2(cos(phi), sin(phi));
  // Foreshortening factor for helical pitch
  return length(pol - wireCenter) * 0.36 - 0.0092;
}

vec3 calcCoilNormal(vec3 p) {
  const float e = 0.0008;
  vec2 k = vec2(1.0, -1.0);
  return normalize(
    k.xyy * sdHeatingCoil(p + k.xyy * e) +
    k.yyx * sdHeatingCoil(p + k.yyx * e) +
    k.yxy * sdHeatingCoil(p + k.yxy * e) +
    k.xxx * sdHeatingCoil(p + k.xxx * e)
  );
}

// ============================================================================
// Procedural Studio / Cozy Dark Room Environment & Reflections
// ============================================================================
vec3 sampleRoomEnv(vec3 dir, float roughness) {
  float y = clamp(dir.y, -1.0, 1.0);
  // Cozy dark architectural space
  vec3 zenith = vec3(0.028, 0.032, 0.048) * uRoomLight;
  vec3 horizon = vec3(0.055, 0.050, 0.062) * uRoomLight;
  vec3 floorCol = vec3(0.020, 0.017, 0.016) * uRoomLight;

  vec3 base = (y >= 0.0)
    ? mix(horizon, zenith, pow(max(1e-5, y), 0.55))
    : mix(horizon * 0.65, floorCol, pow(max(1e-5, -y), 0.5));

  // Crisp studio softbox / tall window strip reflections (iconic vertical highlights on glass!)
  float az = atan(dir.x, dir.z); // [-pi, pi]
  float blur = mix(0.025, 0.45, roughness);

  // Primary cool-neutral tall softbox at Left-Front (~ -0.82 rad)
  float strip1 = smoothstep(0.18 + blur, 0.02, abs(az + 0.82))
               * smoothstep(-0.25 - blur, 0.05, y)
               * smoothstep(0.82 + blur, 0.45, y);

  // Secondary narrower warm rim softbox at Right (~ +1.18 rad)
  float strip2 = smoothstep(0.11 + blur, 0.015, abs(az - 1.18))
               * smoothstep(-0.20 - blur, 0.08, y)
               * smoothstep(0.75 + blur, 0.40, y);

  // Overhead subtle fill diffuser
  float topLight = smoothstep(0.72 - blur * 0.5, 0.96, y);

  float specBoost = mix(1.0, 0.22, roughness);
  base += vec3(0.92, 0.95, 1.05) * strip1 * (0.35 + 1.35 * uRoomLight) * specBoost;
  base += vec3(1.10, 0.90, 0.72) * strip2 * (0.28 + 1.05 * uRoomLight) * specBoost;
  base += vec3(0.55, 0.58, 0.68) * topLight * (0.15 + 0.65 * uRoomLight) * specBoost;

  return base;
}

// Background behind lamp + tabletop surface at y = 0
vec3 renderBackdropAndTable(vec3 ro, vec3 rd) {
  // Subtle warm atmospheric room wall glow behind the lamp
  vec3 bg = sampleRoomEnv(rd, 0.85) * 0.65;

  // Approximate radial glow on room backdrop behind lamp
  // Closest point on ray to lamp center (0, 2.75, 0)
  vec3 lampCenter = vec3(0.0, 2.75, 0.0);
  float tClosest = max(0.0, dot(lampCenter - ro, rd));
  vec3 pClose = ro + rd * tClosest;
  vec2 dLamp = vec2(length(pClose.xz), (pClose.y - 2.65) * 0.48);
  float aura = exp(-dot(dLamp, dLamp) * 0.55);
  vec3 glowHue = mix(uWaxColor, vec3(1.0, 0.55, 0.18), 0.35);
  bg += glowHue * aura * (0.14 * uBulbGlow);

  // Intersect tabletop plane at y = 0.0
  if (rd.y < -1e-4 && ro.y > 0.0) {
    float tTable = -ro.y / rd.y;
    if (tTable > 0.0 && tTable < 32.0) {
      vec3 pt = ro + rd * tTable;
      float r = length(pt.xz);

      // Dark satin walnut / slate tabletop with micro-grain
      float grain = sin(pt.x * 18.0 + sin(pt.z * 4.5) * 1.5) * 0.5 + 0.5;
      vec3 albedo = mix(vec3(0.032, 0.025, 0.022), vec3(0.046, 0.036, 0.031), grain * 0.45);

      // Contact ambient occlusion directly beneath the metal base (radius 0.877)
      float ao = smoothstep(0.83, 1.08, r);

      // Downward colored caustic / glow spill from the illuminated glass globe above!
      // Blocked right next to the opaque base cone (r < 0.92), peaking in a warm halo ring r ~ 1.15..3.2!
      float baseShadow = smoothstep(0.88, 1.32, r);
      float invFalloff = 1.0 / (1.0 + 0.38 * r * r);
      vec3 lampSpill = mix(uWaxColor, uLiquidColor + vec3(1.0, 0.58, 0.22) * 0.5, 0.38)
                     * (0.85 * uBulbGlow * baseShadow * invFalloff);
      // Subtle warm light leak right at the rubber foot gap r ~ 0.88..0.96
      float seamLeak = exp(-pow((r - 0.90) / 0.06, 2.0)) * 0.18 * uBulbGlow;

      // Blurred glossy tabletop reflection of the glowing lamp column!
      vec3 reflDir = reflect(rd, vec3(0.0, 1.0, 0.0));
      vec3 tableRefl = sampleRoomEnv(reflDir, 0.35) * 0.35;
      // Analytical reflection of the glowing lamp column along reflDir
      float denom = dot(reflDir.xz, reflDir.xz);
      if (denom > 1e-4) {
        float tAxis = -dot(pt.xz, reflDir.xz) / denom;
        if (tAxis > 0.0) {
          vec3 pRefl = pt + reflDir * tAxis;
          if (pRefl.y > 1.45 && pRefl.y < 4.70) {
            float dAxis = length(pRefl.xz);
            float colMask = exp(-dAxis * dAxis / 0.26) * smoothstep(1.45, 1.85, pRefl.y) * smoothstep(4.70, 4.10, pRefl.y);
            // Sample wax volume coarsely at reflected height for real dynamic blob reflection on the desk!
            vec4 vSample = sampleWaxGrid(vec3(0.0, pRefl.y, 0.0));
            vec3 colEmit = mix(uLiquidColor * 0.25, uWaxColor * 1.4 + vec3(0.9, 0.45, 0.1) * 0.5, clamp(vSample.r, 0.25, 1.0));
            tableRefl += colEmit * colMask * uBulbGlow * 0.55;
          }
        }
      }

      float fresnel = 0.04 + 0.45 * pow(1.0 - max(0.0, -rd.y), 5.0);
      vec3 surf = albedo * (vec3(0.16 * uRoomLight) + lampSpill + vec3(1.0, 0.55, 0.2) * seamLeak) * ao
                + tableRefl * fresnel * ao;

      // Smooth distance fog into dark room horizon
      float fog = exp(-0.085 * max(0.0, tTable - 3.5));
      return mix(bg, surf, fog);
    }
  }

  return bg;
}

// ============================================================================
// Brushed Anisotropic Spun Metal Shading (Base & Cap)
// ============================================================================
void getMetalMaterial(int type, out vec3 f0, out float baseRough) {
  if (type == 1) {
    // Warm Brushed Brass / Mid-century Gold
    f0 = vec3(0.94, 0.74, 0.36);
    baseRough = 0.24;
  } else if (type == 2) {
    // Sleek Anodized Matte Black
    f0 = vec3(0.065, 0.068, 0.075);
    baseRough = 0.38;
  } else if (type == 3) {
    // Brushed Rose Copper
    f0 = vec3(0.93, 0.56, 0.44);
    baseRough = 0.25;
  } else {
    // Classic Astro Spun Brushed Aluminum / Silver
    f0 = vec3(0.84, 0.86, 0.89);
    baseRough = 0.23;
  }
}

vec3 shadeBrushedMetal(vec3 p, vec3 n, vec3 rd, int partId) {
  vec3 f0;
  float rough;
  getMetalMaterial(uMetalType, f0, rough);

  // Circumferential lathe brush micro-grooves along Y
  float groove = sin(p.y * 280.0) * 0.5 + sin(p.y * 670.0 + 1.3) * 0.5;
  vec3 nBrushed = normalize(n + vec3(0.0, groove * 0.022, 0.0));

  vec3 v = -rd;
  float ndotv = max(0.001, dot(nBrushed, v));
  vec3 refl = reflect(rd, nBrushed);

  // Anisotropic circumferential brushing spreads azimuthal reflections into wide silky metallic bands!
  vec3 envCenter = sampleRoomEnv(refl, max(0.48, rough * 1.9));
  float horizLenSq = n.x * n.x + n.z * n.z;
  vec3 tangent = (horizLenSq > 1e-6) ? (vec3(-n.z, 0.0, n.x) * inversesqrt(horizLenSq)) : vec3(1.0, 0.0, 0.0);
  vec3 radHoriz = (horizLenSq > 1e-6) ? (vec3(n.x, 0.0, n.z) * inversesqrt(horizLenSq)) : vec3(0.0, 0.0, 1.0);
  vec3 envLeft  = sampleRoomEnv(safeNormalize(refl + tangent * 0.52), 0.58);
  vec3 envRight = sampleRoomEnv(safeNormalize(refl - tangent * 0.52), 0.58);
  vec3 envSpec = envCenter * 0.44 + (envLeft + envRight) * 0.28;

  // Ambient studio metallic sky/room wrap so silver/brass looks like real satin metal, never black plastic
  float hemi = 0.55 + 0.45 * nBrushed.y;
  float sideKey = 0.5 + 0.5 * dot(radHoriz, vec3(-0.65, 0.0, 0.76));
  vec3 satinAmbient = vec3(0.24, 0.25, 0.28) * uRoomLight * hemi * (0.65 + 0.75 * sideKey);

  vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - ndotv, 5.0);
  vec3 color = (envSpec + satinAmbient) * fresnel * (1.0 + groove * 0.035);

  if (uMetalType == 2) {
    // Matte black anodized diffuse component
    color += vec3(0.028, 0.030, 0.035) * (0.35 + 0.65 * max(0.0, n.y)) * uRoomLight;
  }

  // Dynamic bounce light from the glowing wax right next to the metal collar (y~1.56) or cap rim (y~4.60)!
  if (partId == 1) {
    float collarNear = exp(-max(0.0, 1.565 - p.y) * 6.5) * max(0.0, n.y * 0.6 + 0.4);
    float r = length(p.xz);
    bool insideCup = (p.y > 1.20 && r < baseOuterRadius(p.y) - 0.002);
    if (insideCup) {
      float depthFade = smoothstep(1.565, 1.32, p.y);
      color += mix(uWaxColor, vec3(1.0, 0.68, 0.32), 0.45) * f0 * depthFade * uBulbGlow * 0.65;
    } else {
      vec3 rimGlow = mix(uWaxColor, vec3(1.0, 0.62, 0.28), 0.42);
      color += rimGlow * f0 * collarNear * uBulbGlow * 0.42;
    }
  } else {
    // Top cap lower rim catches glow from blobs hovering near the top!
    float capRim = exp(-max(0.0, p.y - 4.60) * 9.0) * max(0.0, -n.y * 0.7 + 0.3);
    vec4 topWax = sampleWaxGrid(vec3(0.0, 4.35, 0.0));
    float topPresence = 0.20 + 0.80 * clamp(topWax.r, 0.0, 1.0);
    color += mix(uWaxColor, uLiquidColor + 0.2, 0.25) * f0 * capRim * uBulbGlow * topPresence * 0.42;
  }

  // Tabletop upward bounce on lower base cone
  if (p.y < 1.05 && n.y < 0.0) {
    color += uWaxColor * f0 * (-n.y) * 0.08 * uBulbGlow;
  }

  return color;
}

// ============================================================================
// Subsurface Scattering & Shading of Molten / Solidified Paraffin Wax
// ============================================================================
vec3 computeWaxExtinction(vec3 baseCol) {
  // Derive physically-plausible chromatic absorption spectrum from user's wax color!
  // Bright channels absorb weakly (transmit deep!), dark channels absorb strongly!
  // This produces true chromatic shift: red wax glows golden-yellow where thin/hot, deep ruby where thick/cool!
  vec3 clamped = clamp(baseCol, vec3(0.03), vec3(0.98));
  return -log(clamped) * 3.2 + vec3(0.22);
}

vec3 shadeWaxSurface(vec3 p, vec3 n, vec3 rd, float temp, float solid) {
  vec3 bulbPos = vec3(0.0, 1.14, 0.0);
  vec3 toBulb = bulbPos - p;
  float distBulb = length(toBulb);
  vec3 lBulb = toBulb / max(distBulb, 1e-4);

  // 1. March optical thickness through uWaxVolume toward the bottom halogen bulb!
  float bulbWaxDepth = 0.0;
  const int SSS_STEPS = 6;
  float stepLen = min(distBulb, 1.15) / float(SSS_STEPS);
  for (int i = 1; i <= SSS_STEPS; i++) {
    vec3 sp = p + lBulb * (stepLen * (float(i) - 0.35));
    vec4 sv = sampleWaxGrid(sp);
    bulbWaxDepth += clamp(sv.r, 0.0, 1.6) * stepLen;
  }

  // 2. March forward along view ray rd to measure thin-neck / silhouette edge translucency!
  float viewWaxDepth = 0.0;
  for (int i = 1; i <= 4; i++) {
    vec3 sp = p + rd * (0.075 * float(i));
    vec4 sv = sampleWaxGrid(sp);
    viewWaxDepth += clamp(sv.r, 0.0, 1.5) * 0.075;
  }
  float thinness = exp(-viewWaxDepth * 3.2);

  // Cold solidified wax is cloudier/denser (higher scattering extinction); hot wax is clearer & luminous!
  float densityMult = mix(0.92, 2.35, solid) * mix(1.22, 0.82, clamp((temp - 0.35) / 0.35, 0.0, 1.0));
  vec3 sigmaWax = computeWaxExtinction(uWaxColor) * densityMult;

  // Chromatic subsurface scattering albedo: even thin/bottom-lit wax carries rich pigment (golden-orange core, never chalky white!)
  vec3 waxTintCore = pow(clamp(uWaxColor, vec3(0.02), vec3(1.0)), vec3(0.52));

  // Halogen 2900K warm incandescent light spectrum + inverse-square spotlight falloff
  float spotCone = mix(0.42, 1.0, pow(max(0.0, -lBulb.y), 1.5));
  float atten = spotCone / (0.34 + 0.48 * distBulb * distBulb);
  vec3 bulbIrradiance = vec3(1.32, 0.86, 0.44) * waxTintCore * (uBulbGlow * 4.2 * atten);

  // Minimum internal scattering path (0.16) ensures pigment absorption even on the bottom dome!
  float wrapBulb = max(0.0, dot(n, lBulb) * 0.52 + 0.48);
  vec3 transmitted = bulbIrradiance * exp(-sigmaWax * (bulbWaxDepth * 1.45 + 0.16)) * (0.42 + 0.68 * wrapBulb);

  // Forward-scattered fiery rim/neck glow when thin necks stretch over the light!
  vec3 neckGlow = bulbIrradiance * exp(-sigmaWax * (bulbWaxDepth * 0.90 + 0.10)) * thinness * 0.48;

  // Deep internal ambient multiple-scattering core (warm turbid wax scatter)
  float heightGlow = exp(-max(0.0, p.y - 1.30) * 0.58);
  vec3 multiScatter = uWaxColor * vec3(1.18, 0.85, 0.62) * (uBulbGlow * (0.26 + 0.72 * heightGlow))
                    * mix(1.0, 0.52, solid);

  // Subtle internal convective density swirl (very delicate, organic!)
  float swirl = sin(p.x * 9.0 + p.y * 5.0 - uTime * 0.35) * cos(p.z * 9.0 - p.y * 4.0 + uTime * 0.25);
  transmitted *= (1.0 + 0.045 * swirl * (1.0 - solid));

  // External studio lighting + waxy subsurface wrap
  vec3 v = -rd;
  float ndotv = max(0.001, dot(n, v));
  float waxRough = mix(0.14, 0.52, solid); // Molten wax is silky smooth; cold solid wax is matte-waxy!

  vec3 roomDiff = sampleRoomEnv(n, 0.9) * uWaxColor * (0.55 + 0.35 * solid);
  // Relative IOR between wax (1.45) and carrier liquid (1.36) is ~1.066 -> delicate silky Fresnel!
  float f0Wax = 0.004 + 0.018 * solid;
  float fresWax = f0Wax + (0.22 - f0Wax) * pow(1.0 - ndotv, 4.0);
  vec3 reflDir = reflect(rd, n);
  vec3 waxSpec = sampleRoomEnv(reflDir, waxRough) * fresWax;
  // Also specular glint of the bottom bulb on the underside of rising droplets!
  vec3 hBulb = safeNormalize(lBulb + v);
  float specPow = mix(48.0, 10.0, solid);
  float bulbGlint = pow(max(1e-5, dot(n, hBulb)), specPow) * max(0.0, dot(n, lBulb));
  waxSpec += vec3(1.2, 0.85, 0.55) * bulbGlint * atten * uBulbGlow * mix(0.35, 0.08, solid);

  return transmitted + neckGlow + multiScatter + roomDiff + waxSpec;
}

// Shade the stainless steel heating coil at the bottom of the bottle
vec3 shadeCoil(vec3 p, vec3 n, vec3 rd) {
  vec3 bulbPos = vec3(0.0, 1.16, 0.0);
  vec3 lBulb = normalize(bulbPos - p);
  vec3 v = -rd;
  vec3 refl = reflect(rd, n);

  vec3 steelF0 = vec3(0.72, 0.74, 0.77);
  float ndotv = max(0.01, dot(n, v));
  vec3 fres = steelF0 + (1.0 - steelF0) * pow(1.0 - ndotv, 5.0);

  vec3 env = sampleRoomEnv(refl, 0.22);
  vec3 h = normalize(lBulb + v);
  float spec = pow(max(0.0, dot(n, h)), 64.0);
  float diff = max(0.0, dot(n, lBulb)) * 0.5 + 0.5;

  vec3 bulbCol = vec3(1.35, 0.92, 0.52) * uBulbGlow;
  return env * fres + steelF0 * bulbCol * (diff * 1.4 + spec * 3.5) + uWaxColor * steelF0 * uBulbGlow * 0.45;
}

// ============================================================================
// Ray-Bounding Cylinder Pre-Check (Lamp fits inside r <= 0.91, y in [0, 5.44])
// ============================================================================
bool intersectLampBounds(vec3 ro, vec3 rd, out float tMin, out float tMax) {
  const float R = 0.91;
  float a = dot(rd.xz, rd.xz);
  float b = dot(ro.xz, rd.xz);
  float c = dot(ro.xz, ro.xz) - R * R;

  tMin = 0.0;
  tMax = 100.0;

  if (a > 1e-7) {
    float disc = b * b - a * c;
    if (disc < 0.0) return false;
    float sq = sqrt(disc);
    float t0 = (-b - sq) / a;
    float t1 = (-b + sq) / a;
    tMin = max(tMin, t0);
    tMax = min(tMax, t1);
  } else if (c > 0.0) {
    return false;
  }

  if (abs(rd.y) > 1e-7) {
    float ty0 = (-0.02 - ro.y) / rd.y;
    float ty1 = ( 5.45 - ro.y) / rd.y;
    tMin = max(tMin, min(ty0, ty1));
    tMax = min(tMax, max(ty0, ty1));
  } else if (ro.y < -0.02 || ro.y > 5.45) {
    return false;
  }

  return tMax > tMin && tMax > 0.0;
}

// ============================================================================
// Trace Liquid Cavity Interior (Wax + Volumetric Bulb Shafts + Coil + Meniscus TIR)
// ============================================================================
vec3 traceLiquidInterior(vec3 cavRo, vec3 cavRd, float maxDist, float pixelHash) {
  // Beer-Lambert liquid dye absorption coefficient
  vec3 clampedLiq = clamp(uLiquidColor, vec3(0.05), vec3(0.995));
  vec3 sigmaLiq = -log(clampedLiq) * (0.22 + 0.95 * uLiquidClarity);

  // 1. Check if ray hits the stainless steel heating coil near the bottom (y < 1.44)
  float tCoil = 1e9;
  if (cavRo.y < 1.44 || cavRd.y < 0.0) {
    float tc = 0.01;
    for (int i = 0; i < 36; i++) {
      if (tc >= maxDist) break;
      vec3 pc = cavRo + cavRd * tc;
      if (pc.y > 1.46 && cavRd.y >= 0.0) break;
      float dc = sdHeatingCoil(pc);
      if (dc < 0.0012) {
        tCoil = tc;
        break;
      }
      tc += max(0.006, dc * 0.85);
    }
  }

  // March wax up to min(maxDist, tCoil): wax in front of the coil naturally occludes it,
  // while coil wires in clear fluid (or glimpsed where wax recedes) remain crisp!
  float marchEnd = min(maxDist, tCoil);

  // 2. Hybrid Volume Sphere-Tracing through uWaxVolume!
  bool hitWax = false;
  bool needsPolish = false;
  float tPrev = 0.0;
  float tBracketHi = 0.0;
  float phiPrev = sampleWaxGrid(cavRo).r;
  float tHit = 0.0;

  if (phiPrev >= 1.0) {
    hitWax = true;
    tHit = 0.002;
  } else {
    float t = 0.008 + pixelHash * 0.012;
    for (int step = 0; step < 96; step++) {
      if (t >= marchEnd) break;
      vec3 p = cavRo + cavRd * t;
      vec4 vox = sampleWaxGrid(p);
      float phi = vox.r;

      if (phi >= 1.0) {
        hitWax = true;
        needsPolish = true;
        tBracketHi = t;
        break;
      }

      tPrev = t;
      phiPrev = phi;

      // Fast adaptive step: large step when outside parcel halos (vox.a == 0), fine step near surface!
      float dt = (vox.a < 0.002)
        ? 0.052
        : clamp((1.0 - phi) * 0.042, 0.014, 0.042);
      t += dt;
    }
  }

  // Polish bracketed root [tPrev, tBracketHi] OUTSIDE the 96-step loop so D3D11 FXC links cleanly!
  if (needsPolish) {
    float a = tPrev;
    float b = tBracketHi;
    float fa = evalWaxScalarExact(cavRo + cavRd * a) - 1.0;
    float fb = evalWaxScalarExact(cavRo + cavRd * b) - 1.0;
    for (int k = 0; k < 3; k++) {
      float denom = fb - fa;
      float mid = (abs(denom) > 1e-5)
        ? clamp(a - fa * (b - a) / denom, a + 0.1 * (b - a), b - 0.1 * (b - a))
        : 0.5 * (a + b);
      float fm = evalWaxScalarExact(cavRo + cavRd * mid) - 1.0;
      if (fm > 0.0) {
        b = mid;
        fb = fm;
      } else {
        a = mid;
        fa = fm;
      }
    }
    tHit = 0.5 * (a + b);
  }

  float travelDist = hitWax ? tHit : marchEnd;
  vec3 surfaceRadiance = vec3(0.0);

  if (hitWax) {
    // Compute infinite-resolution exact analytical surface normal & thermal properties!
    vec3 hitPos = cavRo + cavRd * tHit;
    vec3 waxN;
    float waxTemp, waxSolid;
    evalWaxExact(hitPos, waxN, waxTemp, waxSolid);
    if (dot(waxN, cavRd) > 0.0) waxN = -waxN;
    surfaceRadiance = shadeWaxSurface(hitPos, waxN, cavRd, waxTemp, waxSolid);
  } else if (tCoil < maxDist) {
    // Hit the stainless steel heating coil!
    vec3 pCoil = cavRo + cavRd * tCoil;
    vec3 nCoil = calcCoilNormal(pCoil);
    surfaceRadiance = shadeCoil(pCoil, nCoil, cavRd);
  } else {
    // Ray reached the back/top/bottom boundary of the liquid cavity!
    vec3 exitPos = cavRo + cavRd * maxDist;
    float exitR = length(exitPos.xz);
    float exitCavR = cavityRadius(exitPos.y);
    float menY = meniscusHeight(exitR, exitCavR);

    if (exitPos.y >= menY - 0.018 && cavRd.y > 0.0) {
      // --- TOP LIQUID MENISCUS UNDERSIDE: TOTAL INTERNAL REFLECTION MIRROR! ---
      // Rising blobs reflect upside-down in the shimmering top meniscus just like a real lava lamp!
      float q = clamp(exitR / max(exitCavR, 1e-4), 0.0, 1.0);
      vec3 rHat = (exitR > 1e-4) ? vec3(exitPos.x / exitR, 0.0, exitPos.z / exitR) : vec3(0.0);
      vec3 menNormal = normalize(vec3(0.0, -1.0, 0.0) + rHat * (0.10 * q * q * q));
      vec3 tirDir = reflect(cavRd, menNormal);

      // Compute TIR / Fresnel for liquid (1.36) -> air gap (1.0)
      float cosI = max(0.0, dot(-cavRd, menNormal));
      float sin2T = (1.36 * 1.36) * (1.0 - cosI * cosI);
      float tirWeight = (sin2T >= 1.0) ? 1.0 : (0.04 + 0.96 * pow(1.0 - cosI, 5.0));

      // Trace reflected ray downward into uWaxVolume to reflect glowing blobs below!
      vec3 reflCol = vec3(0.02, 0.02, 0.03) * uRoomLight;
      bool hitRefl = false;
      vec3 hitMp = vec3(0.0);
      vec4 hitMv = vec4(0.0);
      float hitTr = 0.0;
      float tr = 0.02;
      for (int m = 0; m < 28; m++) {
        vec3 mp = exitPos + tirDir * tr;
        if (mp.y < 1.30 || length(mp.xz) > cavityRadius(mp.y)) break;
        vec4 mv = sampleWaxGrid(mp);
        if (mv.r >= 1.0) {
          hitRefl = true;
          hitMp = mp;
          hitMv = mv;
          hitTr = tr;
          break;
        }
        tr += 0.048;
      }
      if (hitRefl) {
        const float ge = 0.025;
        vec3 mn = normalize(-vec3(
          sampleWaxGrid(hitMp + vec3(ge, 0.0, 0.0)).r - sampleWaxGrid(hitMp - vec3(ge, 0.0, 0.0)).r,
          sampleWaxGrid(hitMp + vec3(0.0, ge, 0.0)).r - sampleWaxGrid(hitMp - vec3(0.0, ge, 0.0)).r,
          sampleWaxGrid(hitMp + vec3(0.0, 0.0, ge)).r - sampleWaxGrid(hitMp - vec3(0.0, 0.0, ge)).r
        ));
        reflCol = shadeWaxSurface(hitMp, mn, tirDir, hitMv.g, hitMv.b) * exp(-sigmaLiq * hitTr);
      }
      // Subtle dark inner cap underside where not TIR
      vec3 capUnder = vec3(0.015) * uRoomLight;
      surfaceRadiance = mix(capUnder, reflCol * 0.92, tirWeight);
    } else if (exitPos.y <= 1.295 && exitR < 0.42) {
      // --- HALOGEN REFLECTOR BULB APERTURE UNDERNEATH BOTTOM GLASS PLATE! ---
      float aperture = smoothstep(0.42, 0.12, exitR);
      // Warm incandescent bulb hotspot + faceted dichrioc reflector rings
      float rings = 0.88 + 0.12 * cos(exitR * 65.0);
      surfaceRadiance = vec3(1.55, 1.05, 0.62) * (uBulbGlow * 7.5 * aperture * rings)
                      + vec3(0.06) * uRoomLight;
    } else {
      // Refract out the back glass wall into the room, smoothly blending into the lower metal cup below y = 1.565!
      vec3 backGlassN = -calcGlassOuterNormal(exitPos);
      vec3 outDir = refract(cavRd, backGlassN, 1.0 / 1.12); // net cylindrical lensing
      if (dot(outDir, outDir) < 1e-4) outDir = reflect(cavRd, backGlassN);
      vec3 bgThrough = renderBackdropAndTable(exitPos + cavRd * 0.06, normalize(outDir));

      // Internal back-glass Fresnel reflection
      float cosBack = max(0.0, dot(-cavRd, backGlassN));
      float fBack = 0.035 + 0.35 * pow(1.0 - cosBack, 5.0);
      vec3 backRefl = sampleRoomEnv(reflect(cavRd, backGlassN), 0.08);
      vec3 glassExitCol = mix(bgThrough, backRefl, fBack);

      float cupMask = smoothstep(1.575, 1.525, exitPos.y);
      if (cupMask > 0.001) {
        vec3 cupN = normalize(vec3(-exitPos.x, 0.45, -exitPos.z));
        vec3 cupCol = shadeBrushedMetal(exitPos, cupN, cavRd, 1);
        surfaceRadiance = mix(glassExitCol, cupCol, cupMask);
      } else {
        surfaceRadiance = glassExitCol;
      }
    }
  }

  // 3. Attenuate surface radiance by Beer-Lambert liquid dye absorption!
  vec3 transmittance = exp(-sigmaLiq * travelDist);
  vec3 finalLiq = surfaceRadiance * transmittance;

  // 4. Accumulate Volumetric Tyndall Light-Shafts (Upward Halogen Spotlight with Wax Shadowing!)
  if (uBulbGlow > 0.005 && travelDist > 0.02) {
    const int VOL_STEPS = 7;
    float vStep = min(travelDist, 2.8) / float(VOL_STEPS);
    vec3 bulbPos = vec3(0.0, 1.12, 0.0);
    vec3 volAccum = vec3(0.0);

    for (int i = 0; i < VOL_STEPS; i++) {
      float tv = (float(i) + pixelHash) * vStep;
      vec3 vp = cavRo + cavRd * tv;
      vec3 toBulb = bulbPos - vp;
      float dBulb = length(toBulb);
      vec3 lb = toBulb / max(dBulb, 1e-4);

      // Spotlight cone upward through the bottle
      float cone = smoothstep(0.58, 0.90, -lb.y);
      if (cone > 0.01) {
        // Fast 2-tap shadow lookup in uWaxVolume toward bulb: big blobs cast volumetric shadow cones!
        float s1 = sampleWaxGrid(vp + lb * min(0.28, dBulb * 0.35)).r;
        float s2 = sampleWaxGrid(vp + lb * min(0.65, dBulb * 0.72)).r;
        float occ = clamp(1.0 - 0.55 * max(0.0, s1 - 0.35) - 0.55 * max(0.0, s2 - 0.35), 0.08, 1.0);

        float scatter = cone * occ / (0.28 + 0.55 * dBulb * dBulb);
        vec3 pathTrans = exp(-sigmaLiq * tv);
        volAccum += scatter * pathTrans * vStep;
      }
    }

    vec3 scatterHue = mix(vec3(1.15, 0.82, 0.50), uLiquidColor * vec3(1.2, 0.95, 0.75), 0.55);
    finalLiq += scatterHue * volAccum * (uBulbGlow * 0.42);
  }

  return finalLiq;
}

// ============================================================================
// Main Camera Ray Integrator
// ============================================================================
void main() {
  vec2 fragCoord = vUv * uResolution;
  vec2 ndc = (fragCoord * 2.0 - uResolution) / uResolution.y + uLensShift;

  vec3 ro = uCamPos;
  vec3 rd = normalize(uCamMat * vec3(ndc * uFovScale, -1.0));

  // Temporal/spatial hash for jittered volumetric sampling (zero banding!)
  float pixelHash = fract(52.9829189 * fract(dot(fragCoord, vec2(0.06711056, 0.00583715))));

  float tBoundMin, tBoundMax;
  if (!intersectLampBounds(ro, rd, tBoundMin, tBoundMax)) {
    fragColor = vec4(sanitizeHdr(renderBackdropAndTable(ro, rd)), 1.0);
    return;
  }

  // Sphere-trace outer lamp hardware (Metal Base + Cap) AND Outer Glass Bottle
  float tMetal = 1e9;
  int hitMetalPart = 0;
  {
    float t = max(0.01, tBoundMin);
    for (int i = 0; i < 72; i++) {
      if (t > tBoundMax) break;
      vec3 p = ro + rd * t;
      int pid;
      float dm = sdMetalHardware(p, pid);
      if (dm < 0.0009) {
        tMetal = t;
        hitMetalPart = pid;
        break;
      }
      t += max(0.004, dm * 0.90);
    }
  }

  float tGlassIn = 1e9;
  {
    float t = max(0.01, tBoundMin);
    float limit = min(tBoundMax, tMetal);
    for (int i = 0; i < 64; i++) {
      if (t > limit) break;
      vec3 p = ro + rd * t;
      float dg = sdGlassOuter(p);
      if (dg < 0.0009) {
        tGlassIn = t;
        break;
      }
      t += max(0.004, dg * 0.92);
    }
  }

  // Case 1: Ray hits opaque metal Base or Top Cap first!
  if (tMetal < tGlassIn && tMetal < 1e8) {
    vec3 pHit = ro + rd * tMetal;
    vec3 nHit = calcMetalNormal(pHit);
    vec3 col = shadeBrushedMetal(pHit, nHit, rd, hitMetalPart);
    fragColor = vec4(sanitizeHdr(col), 1.0);
    return;
  }

  // Case 2: Ray hits the Borosilicate Glass Bottle!
  if (tGlassIn < 1e8) {
    vec3 gPos = ro + rd * tGlassIn;
    vec3 gNorm = calcGlassOuterNormal(gPos);
    float cosTheta = max(0.001, dot(-rd, gNorm));

    // Crisp outer glass Schlick Fresnel (n_glass = 1.50 -> F0 = 0.04)
    float fGlass = 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);
    vec3 outerRefl = sampleRoomEnv(reflect(rd, gNorm), 0.025);

    // Refract ray into glass / liquid!
    vec3 refrDir = normalize(refract(rd, gNorm, 1.0 / 1.14));

    // March short step through the 0.044 glass wall to find if ray enters the inner liquid cavity!
    bool entersCavity = false;
    float tCavEnter = 0.0;
    {
      float tc = 0.005;
      for (int i = 0; i < 24; i++) {
        vec3 pc = gPos + refrDir * tc;
        float dc = sdLiquidCavity(pc);
        if (dc <= 0.0008) {
          entersCavity = true;
          tCavEnter = tc;
          break;
        }
        if (sdGlassOuter(pc) > 0.015 || tc > 0.35) break;
        tc += max(0.004, dc * 0.85);
      }
    }

    vec3 transmittedCol;

    if (entersCavity) {
      vec3 cavEnterPos = gPos + refrDir * (tCavEnter + 0.0015);
      // Find exit distance from liquid cavity along refrDir
      float tCavExit = 0.003;
      for (int i = 0; i < 64; i++) {
        vec3 pe = cavEnterPos + refrDir * tCavExit;
        float de = -sdLiquidCavity(pe); // positive while inside cavity!
        if (de < 0.0008) break;
        tCavExit += max(0.005, de * 0.92);
      }

      transmittedCol = traceLiquidInterior(cavEnterPos, refrDir, tCavExit, pixelHash);

      // Subtle thick-glass wall optical edge gleam near silhouette
      float wallChord = clamp(tCavEnter / 0.14, 0.0, 1.0);
      vec3 pipeGlow = mix(vec3(1.0, 0.68, 0.35), uWaxColor, 0.35)
                    * exp(-max(0.0, gPos.y - 1.45) * 0.55) * uBulbGlow * 0.25 * wallChord;
      transmittedCol += pipeGlow;
    } else {
      // --- GRAZING BOROSILICATE GLASS RIM! ---
      // Ray travels purely through the curved glass wall without touching the liquid cavity!
      float tGlassOut = 0.02;
      for (int i = 0; i < 32; i++) {
        vec3 pg = gPos + refrDir * tGlassOut;
        float dg = -sdGlassOuter(pg);
        if (dg < 0.001) break;
        tGlassOut += max(0.008, dg * 0.9);
      }
      vec3 exitGlassPos = gPos + refrDir * (tGlassOut + 0.01);
      vec3 exitNorm = -calcGlassOuterNormal(exitGlassPos);
      vec3 exitDir = refract(refrDir, exitNorm, 1.14);
      if (dot(exitDir, exitDir) < 1e-4) exitDir = reflect(refrDir, exitNorm);

      vec3 rimBg = renderBackdropAndTable(exitGlassPos, normalize(exitDir));
      // Borosilicate faint blue-green tint + authentic upward halogen light-piping inside the glass rim!
      vec3 boroTint = exp(-vec3(0.32, 0.10, 0.16) * tGlassOut * 2.5);
      vec3 lightPipe = mix(vec3(1.15, 0.78, 0.42), uWaxColor + 0.25, 0.4)
                     * exp(-max(0.0, gPos.y - 1.42) * 0.52) * uBulbGlow * 0.55;
      transmittedCol = rimBg * boroTint + lightPipe;
    }

    vec3 finalGlass = mix(transmittedCol, outerRefl, fGlass);
    fragColor = vec4(sanitizeHdr(finalGlass), 1.0);
    return;
  }

  // Case 3: Ray passed through bounding cylinder air space without hitting lamp
  fragColor = vec4(sanitizeHdr(renderBackdropAndTable(ro, rd)), 1.0);
}
`;

  // ==========================================================================
  // PASS 3 & 4: Dual-Kawase HDR Bloom (Soft Threshold + Downsample & Upsample)
  // ==========================================================================
  const BLOOM_DOWN_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uSource;
uniform vec2 uTexelSize;
uniform int uFirstPass;

vec3 softThreshold(vec3 c) {
  float br = max(c.r, max(c.g, c.b));
  float knee = 0.45;
  float thresh = 0.72;
  float soft = clamp(br - thresh + knee, 0.0, 2.0 * knee);
  soft = (soft * soft) / (4.0 * knee + 1e-5);
  float contrib = max(soft, br - thresh) / max(br, 1e-4);
  return c * max(0.0, contrib);
}

void main() {
  vec2 h = uTexelSize * 0.5;
  vec3 c0 = texture(uSource, vUv).rgb * 4.0;
  vec3 c1 = texture(uSource, vUv - h).rgb;
  vec3 c2 = texture(uSource, vUv + h).rgb;
  vec3 c3 = texture(uSource, vUv + vec2(h.x, -h.y)).rgb;
  vec3 c4 = texture(uSource, vUv + vec2(-h.x, h.y)).rgb;
  vec3 res = (c0 + c1 + c2 + c3 + c4) / 8.0;
  if (uFirstPass == 1) {
    res = softThreshold(res);
  }
  if (any(isnan(res)) || any(isinf(res))) {
    res = vec3(0.0);
  }
  fragColor = vec4(clamp(res, vec3(0.0), vec3(24.0)), 1.0);
}
`;

  const BLOOM_UP_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uSource;
uniform sampler2D uHighMip;
uniform vec2 uTexelSize;

void main() {
  vec2 h = uTexelSize;
  vec3 sum = vec3(0.0);
  sum += texture(uSource, vUv + vec2(-h.x * 2.0, 0.0)).rgb;
  sum += texture(uSource, vUv + vec2(-h.x,  h.y)).rgb * 2.0;
  sum += texture(uSource, vUv + vec2( 0.0,  h.y * 2.0)).rgb;
  sum += texture(uSource, vUv + vec2( h.x,  h.y)).rgb * 2.0;
  sum += texture(uSource, vUv + vec2( h.x * 2.0, 0.0)).rgb;
  sum += texture(uSource, vUv + vec2( h.x, -h.y)).rgb * 2.0;
  sum += texture(uSource, vUv + vec2( 0.0, -h.y * 2.0)).rgb;
  sum += texture(uSource, vUv + vec2(-h.x, -h.y)).rgb * 2.0;
  vec3 lowBlur = sum / 12.0;
  vec3 highDetail = texture(uHighMip, vUv).rgb;
  fragColor = vec4(highDetail * 0.62 + lowBlur * 0.78, 1.0);
}
`;

  // ==========================================================================
  // PASS 5: Final HDR Composite (Bloom + ACES Filmic Curve + Anti-Banding Dither)
  // ==========================================================================
  const COMPOSITE_FRAG = `#version 300 es
precision highp float;
in vec2 vUv;
out vec4 fragColor;

uniform sampler2D uSceneTex;
uniform sampler2D uBloomTex;
uniform vec2 uResolution;
uniform float uBloomStrength;

// ACES-inspired filmic tonemapper with natural warm highlight desaturation
// (prevents glowing red/orange wax from clipping to harsh flat neon!)
vec3 filmicTonemap(vec3 x) {
  // Subtle cross-channel luminance crosstalk so intense incandescent cores roll off to warm golden-white
  float lum = dot(x, vec3(0.2126, 0.7152, 0.0722));
  vec3 desat = mix(x, vec3(lum) * vec3(1.06, 0.99, 0.92), clamp((lum - 0.85) / (lum + 2.2), 0.0, 0.58));

  const float a = 2.51;
  const float b = 0.03;
  const float c = 2.43;
  const float d = 0.59;
  const float e = 0.14;
  return clamp((desat * (a * desat + b)) / (desat * (c * desat + d) + e), 0.0, 1.0);
}

void main() {
  // Subtle optical chromatic dispersion only near outer lens corners
  vec2 centered = vUv - 0.5;
  float r2 = dot(centered, centered);
  vec2 caOffset = centered * (0.0018 * r2);

  vec3 hdr;
  hdr.r = texture(uSceneTex, vUv + caOffset).r;
  hdr.g = texture(uSceneTex, vUv).g;
  hdr.b = texture(uSceneTex, vUv - caOffset).b;

  vec3 bloom = texture(uBloomTex, vUv).rgb;
  hdr += bloom * uBloomStrength;

  // Gentle cinema lens vignette
  float vignette = 1.0 - 0.28 * r2;
  hdr *= vignette;

  vec3 ldr = filmicTonemap(hdr);
  // Linear to sRGB gamma
  ldr = pow(max(ldr, vec3(0.0)), vec3(1.0 / 2.2));

  // Interleaved gradient noise dither (+/- 0.75/255) -> eliminates 100% of dark gradient banding!
  vec2 fc = vUv * uResolution;
  float ign = fract(52.9829189 * fract(dot(fc, vec2(0.06711056, 0.00583715))));
  ldr += (ign - 0.5) * (1.5 / 255.0);

  fragColor = vec4(clamp(ldr, 0.0, 1.0), 1.0);
}
`;

  window.LavaLamp = window.LavaLamp || {};
  window.LavaLamp.Shaders = {
    FULLSCREEN_VERT,
    VOLUME_FRAG,
    SCENE_FRAG,
    BLOOM_DOWN_FRAG,
    BLOOM_UP_FRAG,
    COMPOSITE_FRAG
  };
})();
