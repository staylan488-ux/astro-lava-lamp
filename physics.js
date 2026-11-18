// ============================================================================
// LavaLamp Physics Engine
// Thermodynamics, Rayleigh-Benard buoyancy, Plateau-Rayleigh necking,
// Cold-start solidification/stalagmites, and 8D Orthoplex surfactant-film
// coalescence & split dynamics.
// ============================================================================

(function () {
  'use strict';

  const NUM_PARCELS = 52;
  const NUM_SLOTS = 16; // 16 unit vectors of the 8D cross-polytope {+-e_1 ... +-e_8}

  // Exact shared geometry functions (must match shaders.js)
  function glassOuterRadius(y) {
    const v = y - 1.80;
    const dg = 0.18;
    const sq = Math.sqrt(v * v + dg * dg);
    const vp = 0.5 * (sq + v - dg);
    const vm = 0.5 * (sq - v - dg);
    const b = Math.max(0.0, 1.46 - y);
    return 0.838 - 0.130 * vp - 0.195 * vm - 0.55 * b * b;
  }

  function cavityRadius(y) {
    return glassOuterRadius(y) - 0.044;
  }

  function meniscusHeight(r, cavR) {
    const q = cavR > 1e-4 ? Math.min(1.0, r / cavR) : 0.0;
    return 4.585 + 0.025 * q * q * q * q;
  }

  // Build the 16 canonical 8D orthoplex basis vectors:
  // Any two distinct slots have dot product 0 or -1 (never positive!).
  const SLOT_VECTORS = new Array(NUM_SLOTS);
  for (let s = 0; s < NUM_SLOTS; s++) {
    const vec = new Float32Array(8);
    const axis = s >> 1;
    const sign = (s & 1) === 0 ? 1.0 : -1.0;
    vec[axis] = sign;
    SLOT_VECTORS[s] = vec;
  }

  function dot8(a, aOff, b, bOff) {
    let sum = 0.0;
    for (let k = 0; k < 8; k++) {
      sum += a[aOff + k] * b[bOff + k];
    }
    return sum;
  }

  function normalize8(arr, off) {
    let sq = 0.0;
    for (let k = 0; k < 8; k++) sq += arr[off + k] * arr[off + k];
    if (sq > 1e-12) {
      const inv = 1.0 / Math.sqrt(sq);
      for (let k = 0; k < 8; k++) arr[off + k] *= inv;
    } else {
      arr[off] = 1.0;
      for (let k = 1; k < 8; k++) arr[off + k] = 0.0;
    }
  }

  class LavaPhysics {
    constructor() {
      this.count = NUM_PARCELS;

      // Per-parcel state arrays (SoA for zero GC & fast packing)
      this.px = new Float32Array(NUM_PARCELS);
      this.py = new Float32Array(NUM_PARCELS);
      this.pz = new Float32Array(NUM_PARCELS);

      this.vx = new Float32Array(NUM_PARCELS);
      this.vy = new Float32Array(NUM_PARCELS);
      this.vz = new Float32Array(NUM_PARCELS);

      this.fx = new Float32Array(NUM_PARCELS);
      this.fy = new Float32Array(NUM_PARCELS);
      this.fz = new Float32Array(NUM_PARCELS);

      this.radius = new Float32Array(NUM_PARCELS);
      this.mass = new Float32Array(NUM_PARCELS);
      this.temp = new Float32Array(NUM_PARCELS);
      this.dTemp = new Float32Array(NUM_PARCELS);
      this.solid = new Float32Array(NUM_PARCELS);

      // Smoothed deformation vector (stretch direction * stretch magnitude)
      this.defX = new Float32Array(NUM_PARCELS);
      this.defY = new Float32Array(NUM_PARCELS);
      this.defZ = new Float32Array(NUM_PARCELS);

      // 8D phase vector per parcel + target phase slot
      this.phase = new Float32Array(NUM_PARCELS * 8);
      this.targetPhase = new Float32Array(NUM_PARCELS * 8);
      this.slotId = new Int16Array(NUM_PARCELS);

      // Connected component workspace (zero-alloc)
      this.compId = new Int16Array(NUM_PARCELS);
      this.bfsQueue = new Int16Array(NUM_PARCELS);
      this.compSize = new Int16Array(NUM_PARCELS);
      this.compMainSlot = new Int16Array(NUM_PARCELS);
      this.compMinY = new Float32Array(NUM_PARCELS);
      this.compMeanTemp = new Float32Array(NUM_PARCELS);
      this.compCx = new Float32Array(NUM_PARCELS);
      this.compCy = new Float32Array(NUM_PARCELS);
      this.compCz = new Float32Array(NUM_PARCELS);
      this.compVx = new Float32Array(NUM_PARCELS);
      this.compVy = new Float32Array(NUM_PARCELS);
      this.compVz = new Float32Array(NUM_PARCELS);
      this.slotUsed = new Uint8Array(NUM_SLOTS);
      this.slotCx = new Float32Array(NUM_SLOTS);
      this.slotCy = new Float32Array(NUM_SLOTS);
      this.slotCz = new Float32Array(NUM_SLOTS);

      // Surfactant film drainage matrix between the 16 slots
      this.slotDrain = new Float32Array(NUM_SLOTS * NUM_SLOTS);
      this.slotTouched = new Uint8Array(NUM_SLOTS * NUM_SLOTS);

      // Packed UBO float32 buffer: 56 max parcels * 4 vec4s = 56 * 16 floats = 896 floats
      this.uboData = new Float32Array(56 * 16);

      // Lamp operating controls
      this.powerOn = true;
      this.dimmer = 1.0;        // 0.0 .. 1.4 (1.0 = sweet spot)
      this.timeScale = 1.0;     // 0.5x .. 8x
      this.paused = false;

      // Dynamic thermal state
      this.bulbGlow = 1.0;      // Filament incandescence [0..1.4]
      this.liquidTemp = 0.50;   // Bulk carrier liquid temperature [0.08..0.70]
      this.simTime = 0.0;

      // Interactive fluid impulse (glass swirl / tap)
      this.pointerActive = false;
      this.pointerPos = [0, 2.5, 0];
      this.pointerVel = [0, 0, 0];

      // Initialize parcel radii with realistic organic variation
      this._initRadii();
      // Start in warmed-up active equilibrium by default
      this.initWarmedUp();
    }

    _initRadii() {
      // Deterministic pseudo-random organic distribution around ~0.168
      // Total volume ~25% of cavity volume
      let seed = 1337;
      const rand = () => {
        seed = (seed * 1664525 + 1013904223) >>> 0;
        return seed / 4294967296;
      };
      for (let i = 0; i < NUM_PARCELS; i++) {
        let r;
        if (i < 10) {
          // Slightly larger core reservoir/plume parcels
          r = 0.182 + 0.022 * rand();
        } else if (i < 42) {
          // Standard body parcels
          r = 0.156 + 0.026 * rand();
        } else {
          // Smaller neck/satellite capable parcels
          r = 0.132 + 0.020 * rand();
        }
        this.radius[i] = r;
        this.mass[i] = r * r * r * 200.0; // proportional to volume
      }
    }

    _assignSlot(i, slot) {
      this.slotId[i] = slot;
      const sv = SLOT_VECTORS[slot];
      const off = i * 8;
      for (let k = 0; k < 8; k++) {
        this.phase[off + k] = sv[k];
        this.targetPhase[off + k] = sv[k];
      }
    }

    _setTargetSlot(i, slot) {
      this.slotId[i] = slot;
      const sv = SLOT_VECTORS[slot];
      const off = i * 8;
      for (let k = 0; k < 8; k++) {
        this.targetPhase[off + k] = sv[k];
      }
    }

    // Mode 1: Warmed-up equilibrium (immediately active, flowing lava lamp!)
    initWarmedUp() {
      this.powerOn = true;
      if (this.dimmer < 0.2) this.dimmer = 1.0;
      this.bulbGlow = this.dimmer;
      this.liquidTemp = 0.50;
      this.slotDrain.fill(0);

      let idx = 0;

      // Group 0 (Slot 0): Bottom molten reservoir + rising dome (~22 parcels)
      for (let i = 0; i < 22; i++, idx++) {
        const angle = (i / 22) * Math.PI * 2.0 + (i % 3) * 0.35;
        const ring = i < 14 ? 0.38 + (i % 3) * 0.11 : 0.14 * ((i % 4) * 0.3);
        const y = i < 14 ? 1.38 + (i % 3) * 0.07 : 1.56 + (i - 14) * 0.075;
        this.px[idx] = Math.cos(angle) * ring;
        this.pz[idx] = Math.sin(angle) * ring;
        this.py[idx] = y;
        this.vx[idx] = 0;
        this.vy[idx] = i >= 16 ? 0.08 : 0.0;
        this.vz[idx] = 0;
        this.temp[idx] = i >= 15 ? 0.585 : 0.49 + (i % 4) * 0.018;
        this.solid[idx] = 0.0;
        this._assignSlot(idx, 0);
      }

      // Group 1 (Slot 1): Large warm rising plume in lower-mid bottle (9 parcels)
      for (let i = 0; i < 9; i++, idx++) {
        const a = i * 2.39996;
        const rr = i === 0 ? 0.0 : 0.09 * Math.sqrt(i / 9);
        this.px[idx] = -0.14 + Math.cos(a) * rr;
        this.pz[idx] = 0.10 + Math.sin(a) * rr;
        this.py[idx] = 2.45 + (i / 9 - 0.5) * 0.36;
        this.vx[idx] = 0.01;
        this.vy[idx] = 0.20;
        this.vz[idx] = -0.01;
        this.temp[idx] = 0.625 - (1 - i / 9) * 0.03;
        this.solid[idx] = 0.0;
        this._assignSlot(idx, 1);
      }

      // Group 2 (Slot 2): Medium warm blob nearing the top (7 parcels)
      for (let i = 0; i < 7; i++, idx++) {
        const a = i * 2.39996;
        const rr = i === 0 ? 0.0 : 0.08 * Math.sqrt(i / 7);
        this.px[idx] = 0.16 + Math.cos(a) * rr;
        this.pz[idx] = -0.11 + Math.sin(a) * rr;
        this.py[idx] = 3.85 + (i / 7 - 0.5) * 0.26;
        this.vx[idx] = -0.01;
        this.vy[idx] = 0.09;
        this.vz[idx] = 0.01;
        this.temp[idx] = 0.545;
        this.solid[idx] = 0.0;
        this._assignSlot(idx, 2);
      }

      // Group 3 (Slot 3): Cooled blob slowly sinking on the right-back (7 parcels)
      for (let i = 0; i < 7; i++, idx++) {
        const a = i * 2.39996;
        const rr = i === 0 ? 0.0 : 0.08 * Math.sqrt(i / 7);
        this.px[idx] = 0.23 + Math.cos(a) * rr;
        this.pz[idx] = 0.15 + Math.sin(a) * rr;
        this.py[idx] = 3.05 + (i / 7 - 0.5) * 0.25;
        this.vx[idx] = 0.0;
        this.vy[idx] = -0.16;
        this.vz[idx] = 0.0;
        this.temp[idx] = 0.435;
        this.solid[idx] = 0.0;
        this._assignSlot(idx, 3);
      }

      // Group 4 (Slot 4): Smaller blob hovering/cooling at top meniscus (5 parcels)
      for (let i = 0; i < 5; i++, idx++) {
        const a = i * 2.39996;
        const rr = i === 0 ? 0.0 : 0.065;
        this.px[idx] = -0.08 + Math.cos(a) * rr;
        this.pz[idx] = -0.09 + Math.sin(a) * rr;
        this.py[idx] = 4.34 + (i / 5 - 0.5) * 0.16;
        this.vx[idx] = 0.0;
        this.vy[idx] = -0.03;
        this.vz[idx] = 0.0;
        this.temp[idx] = 0.485;
        this.solid[idx] = 0.0;
        this._assignSlot(idx, 4);
      }

      // Group 5 (Slot 5): Cute small satellite droplet in mid-bottle (2 parcels)
      for (; idx < NUM_PARCELS; idx++) {
        this.px[idx] = -0.22 + (idx - 50) * 0.04;
        this.pz[idx] = 0.16;
        this.py[idx] = 2.02 + (idx - 50) * 0.06;
        this.vx[idx] = 0.0;
        this.vy[idx] = 0.14;
        this.vz[idx] = 0.0;
        this.temp[idx] = 0.57;
        this.solid[idx] = 0.0;
        this._assignSlot(idx, 5);
      }

      this.defX.fill(0);
      this.defY.fill(0);
      this.defZ.fill(0);

      // Relax a few substeps so initial spacing is completely organic
      for (let s = 0; s < 15; s++) {
        this._substep(0.016);
      }
      this.packUBO();
    }

    // Mode 2: Authentic Cold Start (solidified cooled puck at bottom, warms up through stalagmite stage!)
    initColdStart() {
      this.powerOn = true;
      if (this.dimmer < 0.5) this.dimmer = 1.0;
      this.bulbGlow = 0.08;
      this.liquidTemp = 0.08;
      this.slotDrain.fill(0);

      for (let i = 0; i < NUM_PARCELS; i++) {
        // Pack into a realistic solidified wax mound resting on the bottom coil
        const layer = Math.floor(i / 14);
        const inLayer = i % 14;
        const maxR = Math.max(0.08, 0.52 - layer * 0.11);
        const ringFrac = inLayer === 0 ? 0.0 : (inLayer < 5 ? 0.48 : 0.95);
        const angle = inLayer * 2.39996 + layer * 0.7;
        this.px[i] = Math.cos(angle) * maxR * ringFrac;
        this.pz[i] = Math.sin(angle) * maxR * ringFrac;
        this.py[i] = 1.35 + layer * 0.135 + (inLayer % 2) * 0.02;
        this.vx[i] = 0;
        this.vy[i] = 0;
        this.vz[i] = 0;
        this.temp[i] = 0.08;
        this.solid[i] = 1.0;
        this._assignSlot(i, 0);
      }

      this.defX.fill(0);
      this.defY.fill(0);
      this.defZ.fill(0);

      for (let s = 0; s < 12; s++) {
        this._substep(0.012);
      }
      this.packUBO();
    }

    // Gentle tap / shake impulse on the bottle
    nudgeBottle(strength = 1.0) {
      const angle = Math.random() * Math.PI * 2.0;
      const sx = Math.cos(angle) * 0.42 * strength;
      const sz = Math.sin(angle) * 0.42 * strength;
      for (let i = 0; i < NUM_PARCELS; i++) {
        const hFrac = Math.max(0.15, (this.py[i] - 1.28) / 3.2);
        this.vx[i] += sx * hFrac + (Math.random() - 0.5) * 0.18 * strength;
        this.vz[i] += sz * hFrac + (Math.random() - 0.5) * 0.18 * strength;
        this.vy[i] += (Math.random() - 0.48) * 0.14 * strength;
      }
    }

    setPointerImpulse(active, worldPos, worldVel) {
      this.pointerActive = active;
      if (active && worldPos && worldVel) {
        this.pointerPos[0] = worldPos[0];
        this.pointerPos[1] = worldPos[1];
        this.pointerPos[2] = worldPos[2];
        this.pointerVel[0] = worldVel[0];
        this.pointerVel[1] = worldVel[1];
        this.pointerVel[2] = worldVel[2];
      }
    }

    step(frameDt) {
      if (this.paused) {
        this.packUBO();
        return;
      }
      const clampedDt = Math.min(frameDt, 0.05);
      const totalDt = clampedDt * this.timeScale;
      // Target substep <= 1/90s for rock-solid spring/cohesion stability even at 8x speed
      const substeps = Math.max(2, Math.min(28, Math.ceil(totalDt / 0.011)));
      const dt = totalDt / substeps;

      for (let s = 0; s < substeps; s++) {
        this._substep(dt);
      }

      // Connected-component split & surfactant film drainage runs at substep end
      this._updateTopologyAndPhases(Math.min(totalDt, 0.12));
      this.packUBO();
    }

    _substep(dt) {
      this.simTime += dt;
      const N = NUM_PARCELS;

      // 1. Update tungsten filament & bulk carrier fluid thermal state
      const targetPower = this.powerOn ? this.dimmer : 0.0;
      this.bulbGlow += (targetPower - this.bulbGlow) * Math.min(1.0, dt * 3.5);

      // Bulk liquid warms slowly toward equilibrium (~0.50 at dimmer=1.0, ~0.08 when off)
      const targetLiq = 0.08 + 0.42 * this.bulbGlow;
      // Warms in ~45s real-time during Cold Start (or ~6s at 8x timelapse!)
      const liqRate = targetLiq > this.liquidTemp ? 0.032 : 0.018;
      this.liquidTemp += (targetLiq - this.liquidTemp) * Math.min(1.0, dt * liqRate);

      // Clear accumulators & compute mean vertical wax flux for incompressible counter-current
      let centralUpFlux = 0.0;
      for (let i = 0; i < N; i++) {
        this.fx[i] = 0.0;
        this.fy[i] = 0.0;
        this.fz[i] = 0.0;
        this.dTemp[i] = 0.0;

        const rSq = this.px[i] * this.px[i] + this.pz[i] * this.pz[i];
        if (rSq < 0.18) {
          centralUpFlux += this.vy[i] * this.mass[i];
        }
      }
      centralUpFlux /= N;

      // Slowly wandering Rayleigh-Benard plume angle at base
      const plumeAng1 = this.simTime * 0.17;
      const plumeAng2 = -this.simTime * 0.11 + 1.7;
      const hotX = 0.20 * Math.cos(plumeAng1) + 0.10 * Math.cos(plumeAng2);
      const hotZ = 0.20 * Math.sin(plumeAng1) + 0.10 * Math.sin(plumeAng2);

      // 2. Single-parcel thermodynamics, buoyancy, coil wetting, & fluid drag
      for (let i = 0; i < N; i++) {
        const x = this.px[i];
        const y = this.py[i];
        const z = this.pz[i];
        const r = Math.sqrt(x * x + z * z);
        const cavR = cavityRadius(y);
        const m = this.mass[i];
        let T = this.temp[i];

        // Solidification factor (1 = solid cold wax, 0 = fully molten above T=0.31)
        const sFactor = Math.max(0.0, Math.min(1.0, (0.31 - T) / 0.11));
        this.solid[i] = sFactor;

        // --- Heat Input from Bottom Halogen Bulb + Coil ---
        const hDist = Math.max(0.0, y - 1.28);
        const bottomHeatZone = Math.exp(-hDist / 0.30);
        // Plume hotspot asymmetry encourages localized dome rising rather than flat pancake lift
        const dxHot = x - hotX;
        const dzHot = z - hotZ;
        const hotSpot = 0.86 + 0.28 * Math.exp(-(dxHot * dxHot + dzHot * dzHot) / 0.16);
        // Coil ring conducts heat strongly at r ~ 0.48, y ~ 1.35
        const dCoilR = r - 0.47;
        const dCoilY = y - 1.36;
        const coilBoost = 1.0 + 0.35 * Math.exp(-(dCoilR * dCoilR + dCoilY * dCoilY) / 0.045);

        const qIn = 0.056 * this.bulbGlow * bottomHeatZone * hotSpot * coilBoost;

        // --- Heat Loss to Ambient Liquid & Glass Walls / Top Cap ---
        const heightFrac = Math.max(0.0, Math.min(1.0, (y - 1.30) / 3.28));
        // Ambient fluid is warmer near base, cooler near top cap
        const localFluidTemp = this.liquidTemp * (1.08 - 0.34 * heightFrac);
        const wallProximity = Math.max(0.0, 1.0 - (cavR - r) / 0.24);
        const topCooling = heightFrac > 0.72 ? 1.0 + 2.1 * ((heightFrac - 0.72) / 0.28) : 1.0;
        const coolCoeff = (0.022 + 0.012 * wallProximity) * topCooling;

        this.dTemp[i] += qIn - coolCoeff * (T - localFluidTemp);

        // --- Thermal Buoyancy (Archimedes + Thermal Expansion) ---
        // Neutral buoyancy at T_neutral = 0.512
        // Cold solid wax is denser (sinks to bottom); molten hot wax > 0.512 rises!
        const tNeutral = 0.512;
        let buoyAccel;
        if (sFactor > 0.01) {
          // Semi-solid / solid wax: denser than cold carrier fluid, unless pushed from below by erupting hot wax!
          buoyAccel = -0.38 * sFactor + 0.55 * (1.0 - sFactor) * Math.tanh((T - tNeutral) * 8.5);
        } else {
          buoyAccel = 0.62 * Math.tanh((T - tNeutral) * 7.8);
        }
        this.fy[i] += m * buoyAccel;

        // --- Stainless Steel Coil Wetting / Capillary Pinning at Bottom ---
        // Real lava lamp coils pin the bottom layer of wax so domes neck off instead of whole-pool liftoff
        if (y < 1.54) {
          const pinWeight = (1.54 - y) / 0.26;
          // Pull gently toward coil ring r = 0.45, y = 1.36
          const targetR = 0.44;
          if (r > 0.03) {
            const radialErr = targetR - r;
            const pull = m * 0.26 * pinWeight * radialErr;
            this.fx[i] += (x / r) * pull;
            this.fz[i] += (z / r) * pull;
          }
          // Anchor very bottom parcels slightly downward against premature full-pool float
          if (y < 1.44 && T < 0.64) {
            this.fy[i] -= m * 0.16 * ((1.44 - y) / 0.16);
          }
        }

        // --- Incompressible Carrier Fluid Counter-Current + Gentle Centering ---
        // Rising central blobs push outer wall fluid slightly downward
        const bgFlowY = (r < 0.36 ? 0.25 : -0.35) * centralUpFlux;
        // Very subtle radial centering in upper bottle due to lubrication pressure from tapered glass wall
        if (r > cavR - 0.22 && r > 1e-4) {
          const lub = ((r - (cavR - 0.22)) / 0.22);
          this.fx[i] -= (x / r) * m * 0.12 * lub * lub;
          this.fz[i] -= (z / r) * m * 0.12 * lub * lub;
        }

        // Viscous Stokes drag (colder liquid / colder wax has slightly higher drag)
        const drag = 1.35 + 1.8 * sFactor + 0.35 * (1.0 - this.liquidTemp);
        this.fx[i] -= m * drag * this.vx[i];
        this.fy[i] -= m * drag * (this.vy[i] - bgFlowY);
        this.fz[i] -= m * drag * this.vz[i];

        // --- Interactive Pointer Swirl on Glass ---
        if (this.pointerActive) {
          const dxp = x - this.pointerPos[0];
          const dyp = y - this.pointerPos[1];
          const dzp = z - this.pointerPos[2];
          const dSq = dxp * dxp + dyp * dyp + dzp * dzp;
          if (dSq < 0.65) {
            const wImp = Math.exp(-dSq / 0.18) * m * 2.8;
            this.fx[i] += this.pointerVel[0] * wImp;
            this.fy[i] += this.pointerVel[1] * wImp;
            this.fz[i] += this.pointerVel[2] * wImp;
          }
        }
      }

      // 3. Pairwise Forces: Cohesion, Plateau-Rayleigh Necking, Surfactant Repulsion, & Conduction
      for (let i = 0; i < N; i++) {
        const xi = this.px[i], yi = this.py[i], zi = this.pz[i];
        const ri = this.radius[i];
        const Ti = this.temp[i];
        const si = this.solid[i];
        const offI = i * 8;

        for (let j = i + 1; j < N; j++) {
          const dx = this.px[j] - xi;
          const dy = this.py[j] - yi;
          const dz = this.pz[j] - zi;
          const dSq = dx * dx + dy * dy + dz * dz;
          const sumR = ri + this.radius[j];
          const maxRange = 1.72 * sumR;

          if (dSq >= maxRange * maxRange || dSq < 1e-10) continue;

          const d = Math.sqrt(dSq);
          const nx = dx / d;
          const ny = dy / d;
          const nz = dz / d;

          // Phase affinity in [-1, 1]; clamp to [0, 1] for intra-droplet vs inter-droplet blend
          const rawDot = dot8(this.phase, offI, this.phase, j * 8);
          const affinity = Math.max(0.0, Math.min(1.0, rawDot));
          const separate = 1.0 - affinity;

          const avgMass = 0.5 * (this.mass[i] + this.mass[j]);
          const avgSolid = 0.5 * (si + this.solid[j]);

          let fMag = 0.0; // positive = attraction, negative = repulsion

          // A. Intra-droplet mechanics (same blob: incompressible core + surface tension)
          if (affinity > 0.01) {
            const dEq = 0.78 * sumR;
            if (d < dEq) {
              // Soft incompressible wax core preserves true multi-parcel droplet volume
              const pen = (dEq - d) / dEq;
              fMag -= affinity * avgMass * 16.5 * pen * Math.sqrt(pen);
            } else if (d < 1.68 * sumR) {
              // Plateau-Rayleigh surface tension: peaks midway, weakens as neck thins so plumes pinch off!
              const u = (d - dEq) / (1.68 * sumR - dEq);
              // Cooler wax has slightly stronger surface tension; solid wax binds stiffly (stalagmites!)
              const surfStrength = 2.65 + 0.95 * (1.0 - 0.5 * (Ti + this.temp[j])) + 3.8 * avgSolid;
              fMag += affinity * avgMass * surfStrength * Math.sin(Math.PI * u);
            }

            // Internal viscous coupling + thermal conduction inside same blob
            if (d < 1.45 * sumR) {
              const overlap = 1.0 - d / (1.45 * sumR);
              const visc = affinity * avgMass * (3.2 + 14.0 * avgSolid) * overlap;
              const dvx = this.vx[j] - this.vx[i];
              const dvy = this.vy[j] - this.vy[i];
              const dvz = this.vz[j] - this.vz[i];

              this.fx[i] += dvx * visc;
              this.fy[i] += dvy * visc;
              this.fz[i] += dvz * visc;
              this.fx[j] -= dvx * visc;
              this.fy[j] -= dvy * visc;
              this.fz[j] -= dvz * visc;

              // Heat conducts smoothly within a coherent blob
              const cond = affinity * 0.42 * overlap * (this.temp[j] - Ti);
              this.dTemp[i] += cond;
              this.dTemp[j] -= cond;
            }
          }

          // B. Inter-droplet surfactant film cushion repulsion (sliding past & squishing!)
          if (separate > 0.01 && d < 1.20 * sumR) {
            const pen = (1.20 * sumR - d) / (1.20 * sumR);
            fMag -= separate * avgMass * 13.0 * pen * pen;

            // Slight tangential lubrication damping on contact
            const dvDotN = (this.vx[j] - this.vx[i]) * nx +
                           (this.vy[j] - this.vy[i]) * ny +
                           (this.vz[j] - this.vz[i]) * nz;
            if (dvDotN < 0.0) {
              const damp = separate * avgMass * 3.0 * pen * dvDotN;
              this.fx[i] += nx * damp;
              this.fy[i] += ny * damp;
              this.fz[i] += nz * damp;
              this.fx[j] -= nx * damp;
              this.fy[j] -= ny * damp;
              this.fz[j] -= nz * damp;
            }
          }

          if (fMag !== 0.0) {
            const fx = nx * fMag;
            const fy = ny * fMag;
            const fz = nz * fMag;
            this.fx[i] += fx;
            this.fy[i] += fy;
            this.fz[i] += fz;
            this.fx[j] -= fx;
            this.fy[j] -= fy;
            this.fz[j] -= fz;
          }
        }
      }

      // 4. Symplectic Euler Integration + Curved Glass Cavity Wall Boundaries
      for (let i = 0; i < N; i++) {
        const invM = 1.0 / this.mass[i];
        this.vx[i] += this.fx[i] * invM * dt;
        this.vy[i] += this.fy[i] * invM * dt;
        this.vz[i] += this.fz[i] * invM * dt;

        // Clamp extreme velocities so numerical spikes are impossible
        const vSq = this.vx[i] * this.vx[i] + this.vy[i] * this.vy[i] + this.vz[i] * this.vz[i];
        if (vSq > 2.25) {
          const s = 1.5 / Math.sqrt(vSq);
          this.vx[i] *= s;
          this.vy[i] *= s;
          this.vz[i] *= s;
        }

        this.px[i] += this.vx[i] * dt;
        this.py[i] += this.vy[i] * dt;
        this.pz[i] += this.vz[i] * dt;

        this.temp[i] = Math.max(0.04, Math.min(0.95, this.temp[i] + this.dTemp[i] * dt));

        // --- Smooth Elastic Cavity Boundary Containment ---
        const ri = this.radius[i] * 0.78;
        const floorY = 1.29 + ri * 0.65;
        if (this.py[i] < floorY) {
          this.py[i] = floorY;
          if (this.vy[i] < 0.0) this.vy[i] *= -0.15;
        }

        const rHoriz = Math.sqrt(this.px[i] * this.px[i] + this.pz[i] * this.pz[i]);
        const cavR = cavityRadius(this.py[i]);
        const topY = meniscusHeight(rHoriz, cavR) - ri * 0.72;
        if (this.py[i] > topY) {
          this.py[i] = topY;
          if (this.vy[i] > 0.0) this.vy[i] *= -0.15;
        }

        const maxR = Math.max(0.05, cavR - ri * 0.78);
        if (rHoriz > maxR) {
          const invR = 1.0 / rHoriz;
          const nx = this.px[i] * invR;
          const nz = this.pz[i] * invR;
          this.px[i] = nx * maxR;
          this.pz[i] = nz * maxR;
          const vRad = this.vx[i] * nx + this.vz[i] * nz;
          if (vRad > 0.0) {
            this.vx[i] -= 1.15 * vRad * nx;
            this.vz[i] -= 1.15 * vRad * nz;
          }
        }

        // Update smooth hydrodynamic parcel deformation (prolate when accelerating/fast, oblate at top/bottom)
        const speed = Math.sqrt(this.vx[i] * this.vx[i] + this.vy[i] * this.vy[i] + this.vz[i] * this.vz[i]);
        let tx = 0.0, ty = 0.0, tz = 0.0;
        if (speed > 0.025 && this.solid[i] < 0.5) {
          const stretch = Math.min(0.24, speed * 0.55) * (1.0 - this.solid[i]);
          tx = (this.vx[i] / speed) * stretch;
          ty = (this.vy[i] / speed) * stretch;
          tz = (this.vz[i] / speed) * stretch;
        }
        // Squash slightly oblate when pressing against top meniscus or floor
        if (this.py[i] > topY - 0.06) {
          ty -= 0.14;
        } else if (this.py[i] < floorY + 0.05) {
          ty -= 0.10;
        }
        const blend = Math.min(1.0, dt * 6.0);
        this.defX[i] += (tx - this.defX[i]) * blend;
        this.defY[i] += (ty - this.defY[i]) * blend;
        this.defZ[i] += (tz - this.defZ[i]) * blend;
      }
    }

    // ========================================================================
    // Connected-Component Split Detection & Surfactant Film Coalescence
    // Guarantees:
    //  1) Newly pinched-off droplets only change slot AFTER zero kernel overlap
    //     (0.000000 visual pop!).
    //  2) Returning droplets touching the bottom coil smoothly drain their film
    //     and rotate their 8D phase vector into the reservoir!
    //  3) Rising hot & sinking cold droplets glide/squish past each other, while
    //     slow co-thermal touching droplets gradually coalesce!
    // ========================================================================
    _updateTopologyAndPhases(dt) {
      const N = NUM_PARCELS;
      this.compId.fill(-1);
      let numComps = 0;

      // 1. Find connected components of parcels that share BOTH spatial kernel overlap
      //    (d < 2*(r_i + r_j)) AND positive phase affinity (dot > 0.35).
      for (let i = 0; i < N; i++) {
        if (this.compId[i] !== -1) continue;
        const cid = numComps++;
        this.compId[i] = cid;
        let qHead = 0, qTail = 0;
        this.bfsQueue[qTail++] = i;

        let count = 0;
        let minY = 1e9;
        let sumT = 0, cx = 0, cy = 0, cz = 0, vx = 0, vy = 0, vz = 0;
        const slotCounts = new Int16Array(NUM_SLOTS);

        while (qHead < qTail) {
          const u = this.bfsQueue[qHead++];
          count++;
          if (this.py[u] < minY) minY = this.py[u];
          sumT += this.temp[u];
          cx += this.px[u];
          cy += this.py[u];
          cz += this.pz[u];
          vx += this.vx[u];
          vy += this.vy[u];
          vz += this.vz[u];
          slotCounts[this.slotId[u]]++;

          const ux = this.px[u], uy = this.py[u], uz = this.pz[u];
          const ru = this.radius[u];
          const uOff = u * 8;

          for (let v = 0; v < N; v++) {
            if (this.compId[v] !== -1) continue;
            const dx = this.px[v] - ux;
            const dy = this.py[v] - uy;
            const dz = this.pz[v] - uz;
            // Compact support radius is 2*r_u + 2*r_v = 2*(ru + rv)
            const supp = 1.98 * (ru + this.radius[v]);
            if (dx * dx + dy * dy + dz * dz < supp * supp) {
              if (dot8(this.phase, uOff, this.phase, v * 8) > 0.35) {
                this.compId[v] = cid;
                this.bfsQueue[qTail++] = v;
              }
            }
          }
        }

        const invC = 1.0 / count;
        this.compSize[cid] = count;
        this.compMinY[cid] = minY;
        this.compMeanTemp[cid] = sumT * invC;
        this.compCx[cid] = cx * invC;
        this.compCy[cid] = cy * invC;
        this.compCz[cid] = cz * invC;
        this.compVx[cid] = vx * invC;
        this.compVy[cid] = vy * invC;
        this.compVz[cid] = vz * invC;

        let bestSlot = this.slotId[i];
        let bestCnt = -1;
        for (let s = 0; s < NUM_SLOTS; s++) {
          if (slotCounts[s] > bestCnt) {
            bestCnt = slotCounts[s];
            bestSlot = s;
          }
        }
        this.compMainSlot[cid] = bestSlot;
      }

      // 2. Identify the main bottom reservoir component (lowest minY with largest size near coil)
      let reservoirComp = 0;
      let bestScore = -1e9;
      for (let c = 0; c < numComps; c++) {
        const score = (this.compMinY[c] < 1.65 ? 100.0 : 0.0) + this.compSize[c] * 2.0 - this.compMinY[c] * 5.0;
        if (score > bestScore) {
          bestScore = score;
          reservoirComp = c;
        }
      }
      // Keep bottom reservoir anchored on Slot 0 for clarity
      this.compMainSlot[reservoirComp] = 0;

      // 3. Detect if two disconnected components share the same slot!
      //    Because we linked edges up to full support 1.98*(r_u + r_v), any two disconnected
      //    components sharing a slot have ZERO field overlap right now -> re-slotting the smaller
      //    pinched-off droplet causes 0.000000 visual change!
      this.slotUsed.fill(0);
      this.slotUsed[0] = 1;
      this.slotCx[0] = this.compCx[reservoirComp];
      this.slotCy[0] = this.compCy[reservoirComp];
      this.slotCz[0] = this.compCz[reservoirComp];

      // Sort other components largest-first so large blobs keep their existing slot
      const order = [];
      for (let c = 0; c < numComps; c++) {
        if (c !== reservoirComp) order.push(c);
      }
      order.sort((a, b) => this.compSize[b] - this.compSize[a]);

      for (let idx = 0; idx < order.length; idx++) {
        const c = order[idx];
        let s = this.compMainSlot[c];
        if (this.slotUsed[s] === 0) {
          this.slotUsed[s] = 1;
          this.slotCx[s] = this.compCx[c];
          this.slotCy[s] = this.compCy[c];
          this.slotCz[s] = this.compCz[c];
        } else {
          // Newly pinched-off droplet! Pick a free slot (or most distant slot if >16 droplets)
          let chosen = -1;
          for (let cand = 1; cand < NUM_SLOTS; cand++) {
            if (this.slotUsed[cand] === 0) {
              chosen = cand;
              break;
            }
          }
          if (chosen === -1) {
            let maxDistSq = -1.0;
            chosen = 1;
            for (let cand = 1; cand < NUM_SLOTS; cand++) {
              const dx = this.compCx[c] - this.slotCx[cand];
              const dy = this.compCy[c] - this.slotCy[cand];
              const dz = this.compCz[c] - this.slotCz[cand];
              const dSq = dx * dx + dy * dy + dz * dz;
              if (dSq > maxDistSq) {
                maxDistSq = dSq;
                chosen = cand;
              }
            }
          }
          this.compMainSlot[c] = chosen;
          this.slotUsed[chosen] = 1;
          this.slotCx[chosen] = this.compCx[c];
          this.slotCy[chosen] = this.compCy[c];
          this.slotCz[chosen] = this.compCz[c];

          // Verify zero overlap with any parcel currently holding `chosen`; snap immediately!
          for (let i = 0; i < N; i++) {
            if (this.compId[i] === c) {
              this._assignSlot(i, chosen);
            }
          }
        }
      }

      // Ensure reservoir parcels target Slot 0
      for (let i = 0; i < N; i++) {
        const c = this.compId[i];
        this._setTargetSlot(i, this.compMainSlot[c]);
      }

      // 4. Surfactant Film Drainage & Coalescence between Distinct Touching Droplets
      this.slotTouched.fill(0);
      for (let i = 0; i < N; i++) {
        const ci = this.compId[i];
        const si = this.compMainSlot[ci];
        const xi = this.px[i], yi = this.py[i], zi = this.pz[i];
        const ri = this.radius[i];

        for (let j = i + 1; j < N; j++) {
          const cj = this.compId[j];
          if (ci === cj) continue;
          const sj = this.compMainSlot[cj];
          if (si === sj) continue;

          const dx = this.px[j] - xi;
          const dy = this.py[j] - yi;
          const dz = this.pz[j] - zi;
          const touchDist = 1.24 * (ri + this.radius[j]);
          if (dx * dx + dy * dy + dz * dz >= touchDist * touchDist) continue;

          const lo = si < sj ? si : sj;
          const hi = si < sj ? sj : si;
          const pairIdx = lo * NUM_SLOTS + hi;
          this.slotTouched[pairIdx] = 1;

          const contactY = 0.5 * (yi + this.py[j]);
          const dvx = this.compVx[ci] - this.compVx[cj];
          const dvy = this.compVy[ci] - this.compVy[cj];
          const dvz = this.compVz[ci] - this.compVz[cj];
          const relSpeed = Math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
          const tempDiff = Math.abs(this.compMeanTemp[ci] - this.compMeanTemp[cj]);

          let drainRate = 0.0;

          // Case A: Heating coil surface-tension breaker at bottom!
          // Returning cool blobs touching the bottom reservoir near the coil coalesce reliably.
          const involvesReservoir = (ci === reservoirComp || cj === reservoirComp);
          const otherComp = ci === reservoirComp ? cj : ci;
          if (involvesReservoir && contactY < 1.76 && this.compVy[otherComp] < 0.05) {
            drainRate = 2.4;
          } else if (contactY < 1.56 && relSpeed < 0.25) {
            drainRate = 1.8;
          } else if (relSpeed < 0.13 && tempDiff < 0.095) {
            // Case B: Two co-thermal blobs dwelling gently against each other (e.g. resting at top or co-rising)
            drainRate = 0.72 * (1.0 - relSpeed / 0.13) * (1.0 - tempDiff / 0.095);
          }

          if (drainRate > 0.0) {
            this.slotDrain[pairIdx] += drainRate * dt;
            if (this.slotDrain[pairIdx] >= 1.0) {
              // Film ruptures! Merge smaller component's target slot into larger (or reservoir) component's slot
              let winnerComp = ci;
              let loserComp = cj;
              if (cj === reservoirComp || (ci !== reservoirComp && this.compSize[cj] > this.compSize[ci])) {
                winnerComp = cj;
                loserComp = ci;
              }
              const winSlot = this.compMainSlot[winnerComp];
              for (let k = 0; k < N; k++) {
                if (this.compId[k] === loserComp) {
                  this._setTargetSlot(k, winSlot);
                }
              }
              this.compMainSlot[loserComp] = winSlot;
              this.slotDrain[pairIdx] = 0.0;
            }
          }
        }
      }

      // Decay film drainage when droplets separate so glancing taps reset cleanly
      for (let p = 0; p < NUM_SLOTS * NUM_SLOTS; p++) {
        if (this.slotTouched[p] === 0 && this.slotDrain[p] > 0.0) {
          this.slotDrain[p] = Math.max(0.0, this.slotDrain[p] - dt * 1.6);
        }
      }

      // 5. Smoothly rotate each parcel's 8D unit phase vector toward its target slot!
      //    This makes film rupture blossom smoothly from a tiny contact neck into a unified blob!
      const rotBlend = Math.min(1.0, dt * 4.8);
      for (let i = 0; i < N; i++) {
        const off = i * 8;
        let diffSq = 0.0;
        for (let k = 0; k < 8; k++) {
          const d = this.targetPhase[off + k] - this.phase[off + k];
          diffSq += d * d;
        }
        if (diffSq > 1e-7) {
          for (let k = 0; k < 8; k++) {
            this.phase[off + k] += (this.targetPhase[off + k] - this.phase[off + k]) * rotBlend;
          }
          normalize8(this.phase, off);
        }
      }
    }

    // Pack into std140 UBO layout:
    // For each parcel i (0..55):
    //   vec4 posRad  : (px, py, pz, radius)
    //   vec4 defTemp : (defX, defY, defZ, temp + solid*2.0 packed or separate)
    //   vec4 phaseA  : phase[0..3]
    //   vec4 phaseB  : phase[4..7]
    packUBO() {
      const buf = this.uboData;
      for (let i = 0; i < NUM_PARCELS; i++) {
        const base = i * 16;
        buf[base + 0] = this.px[i];
        buf[base + 1] = this.py[i];
        buf[base + 2] = this.pz[i];
        buf[base + 3] = this.radius[i];

        buf[base + 4] = this.defX[i];
        buf[base + 5] = this.defY[i];
        buf[base + 6] = this.defZ[i];
        // Pack temperature in integer/frac or cleanly: temp is in [0.04..0.95], solid in [0..1]
        // Even cleaner: solid is a deterministic smooth function of temp: clamp((0.31 - T)/0.11, 0, 1)!
        // So storing exact `this.temp[i]` in .w gives BOTH exact temp AND exact solidness in GLSL!
        buf[base + 7] = this.temp[i];

        const pOff = i * 8;
        buf[base + 8]  = this.phase[pOff + 0];
        buf[base + 9]  = this.phase[pOff + 1];
        buf[base + 10] = this.phase[pOff + 2];
        buf[base + 11] = this.phase[pOff + 3];

        buf[base + 12] = this.phase[pOff + 4];
        buf[base + 13] = this.phase[pOff + 5];
        buf[base + 14] = this.phase[pOff + 6];
        buf[base + 15] = this.phase[pOff + 7];
      }
      return buf;
    }

    // Diagnostic stats for UI readout
    getThermometerReading() {
      // Map liquidTemp [0.08..0.66] to realistic Celsius [21°C room temp .. 66°C overheated]
      // At sweet spot liquidTemp = 0.50 -> ~54°C (classic lava lamp operating temp: 52-56°C!)
      const celsius = 21.0 + (this.liquidTemp - 0.08) * (33.0 / 0.42);
      let status = 'Off · Cooling';
      if (this.powerOn && this.dimmer > 0.02) {
        if (this.liquidTemp < 0.28) status = 'Warming Up · Solid';
        else if (this.liquidTemp < 0.43) status = 'Melting · Stalagmites';
        else if (this.dimmer < 0.72) status = 'Low · Lazy Dome';
        else if (this.dimmer <= 1.16) status = 'Optimal Flow';
        else status = 'Overheated · Frantic';
      } else if (this.liquidTemp < 0.14) {
        status = 'Off · Room Temp';
      }
      return { celsius: Math.round(celsius), status };
    }
  }

  window.LavaLamp = window.LavaLamp || {};
  window.LavaLamp.LavaPhysics = LavaPhysics;
  window.LavaLamp.glassOuterRadius = glassOuterRadius;
  window.LavaLamp.cavityRadius = cavityRadius;
})();
