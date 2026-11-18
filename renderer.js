// ============================================================================
// LavaLamp WebGL 2 Photorealistic Renderer
// Manages 3D Scalar Volume (80x192x80 RGBA16F), UBOs, Hybrid Raymarcher,
// Multi-Mip Dual-Kawase HDR Bloom, Orbit Camera, & Ray-Bottle Unprojection.
// ============================================================================

(function () {
  'use strict';

  const VOL_X = 80;
  const VOL_Y = 192;
  const VOL_Z = 80;

  function hexToLinearRgb(hex) {
    const clean = hex.replace('#', '');
    const r = parseInt(clean.substring(0, 2), 16) / 255;
    const g = parseInt(clean.substring(2, 4), 16) / 255;
    const b = parseInt(clean.substring(4, 6), 16) / 255;
    return [
      Math.pow(r, 2.2),
      Math.pow(g, 2.2),
      Math.pow(b, 2.2)
    ];
  }

  class LavaRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      const gl = canvas.getContext('webgl2', {
        alpha: false,
        antialias: false,
        depth: false,
        stencil: false,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: false
      });

      if (!gl) {
        throw new Error('WebGL 2 is required to run this simulation.');
      }
      this.gl = gl;

      // Enable float / half-float render targets & linear filtering
      this.extColorBufferFloat = gl.getExtension('EXT_color_buffer_float');
      this.extColorBufferHalfFloat = gl.getExtension('EXT_color_buffer_half_float');
      this.extFloatLinear = gl.getExtension('OES_texture_float_linear');

      this.useHalfFloat = Boolean(this.extColorBufferFloat || this.extColorBufferHalfFloat);

      // Empty VAO for attribute-less fullscreen triangle (`gl_VertexID`)
      this.emptyVao = gl.createVertexArray();

      // Compile all shader programs
      const S = window.LavaLamp.Shaders;
      this.progVolume = this._createProgram('Volume3D', S.FULLSCREEN_VERT, S.VOLUME_FRAG);
      this.progScene = this._createProgram('SceneRaymarch', S.FULLSCREEN_VERT, S.SCENE_FRAG);
      this.progBloomDown = this._createProgram('BloomDown', S.FULLSCREEN_VERT, S.BLOOM_DOWN_FRAG);
      this.progBloomUp = this._createProgram('BloomUp', S.FULLSCREEN_VERT, S.BLOOM_UP_FRAG);
      this.progComposite = this._createProgram('Composite', S.FULLSCREEN_VERT, S.COMPOSITE_FRAG);

      // Bind shared `ParcelBlock` UBO to binding point 0 on both Volume & Scene programs
      this.parcelUbo = gl.createBuffer();
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.parcelUbo);
      gl.bufferData(gl.UNIFORM_BUFFER, 56 * 16 * 4, gl.DYNAMIC_DRAW);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, this.parcelUbo);

      const volBlockIdx = gl.getUniformBlockIndex(this.progVolume, 'ParcelBlock');
      if (volBlockIdx !== gl.INVALID_INDEX) {
        gl.uniformBlockBinding(this.progVolume, volBlockIdx, 0);
      }
      const sceneBlockIdx = gl.getUniformBlockIndex(this.progScene, 'ParcelBlock');
      if (sceneBlockIdx !== gl.INVALID_INDEX) {
        gl.uniformBlockBinding(this.progScene, sceneBlockIdx, 0);
      }

      // Cache uniform locations
      this._cacheUniforms();

      // Build 3D scalar volume texture & cached slice FBOs (fast D3D11/ANGLE RTV reuse!)
      this._initVolume3D();

      // Screen FBOs (created in resize)
      this.hdrFbo = null;
      this.hdrTex = null;
      this.bloomDownFbos = [];
      this.bloomDownTexs = [];
      this.bloomUpFbos = [];
      this.bloomUpTexs = [];
      this.bloomSizes = [];

      this.renderScale = 1.0; // Quality scale
      this.width = 0;
      this.height = 0;

      // Visual & Material settings
      this.waxHex = '#ff2a12';
      this.liquidHex = '#5222b8';
      this.waxColorLinear = hexToLinearRgb(this.waxHex);
      this.liquidColorLinear = hexToLinearRgb(this.liquidHex);
      this.liquidClarity = 0.55;
      this.metalType = 0; // 0=Silver, 1=Brass, 2=MatteBlack, 3=Copper
      this.roomLight = 0.36;
      this.bloomStrength = 0.34;

      // Orbit Camera State (smoothly damped, full head-to-toe lamp + tabletop framing!)
      this.camYaw = 0.24;
      this.camPitch = 0.08;
      this.camDist = 8.65;
      this.targetYaw = 0.24;
      this.targetPitch = 0.08;
      this.targetDist = 8.65;
      this.camTarget = [0.0, 2.58, 0.0];
      this.fovDeg = 42.0;
      this.lensShiftX = 0.0;

      this.camPos = new Float32Array(3);
      this.camMat = new Float32Array(9);

      this.resize();
      this._updateCamera(1.0);
    }

    _compileShader(name, type, source) {
      const gl = this.gl;
      const sh = gl.createShader(type);
      gl.shaderSource(sh, source);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        const info = gl.getShaderInfoLog(sh);
        gl.deleteShader(sh);
        throw new Error(`[${name}] Shader compile error:\n${info}`);
      }
      return sh;
    }

    _createProgram(name, vsSrc, fsSrc) {
      const gl = this.gl;
      const vs = this._compileShader(name + ':VS', gl.VERTEX_SHADER, vsSrc);
      const fs = this._compileShader(name + ':FS', gl.FRAGMENT_SHADER, fsSrc);
      const prog = gl.createProgram();
      gl.attachShader(prog, vs);
      gl.attachShader(prog, fs);
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const info = gl.getProgramInfoLog(prog);
        const vsLog = gl.getShaderInfoLog(vs);
        const fsLog = gl.getShaderInfoLog(fs);
        gl.deleteShader(vs);
        gl.deleteShader(fs);
        gl.deleteProgram(prog);
        throw new Error(`[${name}] Program link error: ${info} | VS: ${vsLog} | FS: ${fsLog}`);
      }
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      return prog;
    }

    _cacheUniforms() {
      const gl = this.gl;
      this.uVol = {
        sliceW: gl.getUniformLocation(this.progVolume, 'uSliceW'),
        simTime: gl.getUniformLocation(this.progVolume, 'uSimTime')
      };

      this.uScene = {
        waxVolume: gl.getUniformLocation(this.progScene, 'uWaxVolume'),
        resolution: gl.getUniformLocation(this.progScene, 'uResolution'),
        camPos: gl.getUniformLocation(this.progScene, 'uCamPos'),
        camMat: gl.getUniformLocation(this.progScene, 'uCamMat'),
        fovScale: gl.getUniformLocation(this.progScene, 'uFovScale'),
        lensShift: gl.getUniformLocation(this.progScene, 'uLensShift'),
        time: gl.getUniformLocation(this.progScene, 'uTime'),
        bulbGlow: gl.getUniformLocation(this.progScene, 'uBulbGlow'),
        liquidTemp: gl.getUniformLocation(this.progScene, 'uLiquidTemp'),
        waxColor: gl.getUniformLocation(this.progScene, 'uWaxColor'),
        liquidColor: gl.getUniformLocation(this.progScene, 'uLiquidColor'),
        liquidClarity: gl.getUniformLocation(this.progScene, 'uLiquidClarity'),
        metalType: gl.getUniformLocation(this.progScene, 'uMetalType'),
        roomLight: gl.getUniformLocation(this.progScene, 'uRoomLight')
      };

      this.uBloomDown = {
        source: gl.getUniformLocation(this.progBloomDown, 'uSource'),
        texelSize: gl.getUniformLocation(this.progBloomDown, 'uTexelSize'),
        firstPass: gl.getUniformLocation(this.progBloomDown, 'uFirstPass')
      };

      this.uBloomUp = {
        source: gl.getUniformLocation(this.progBloomUp, 'uSource'),
        highMip: gl.getUniformLocation(this.progBloomUp, 'uHighMip'),
        texelSize: gl.getUniformLocation(this.progBloomUp, 'uTexelSize')
      };

      this.uComp = {
        sceneTex: gl.getUniformLocation(this.progComposite, 'uSceneTex'),
        bloomTex: gl.getUniformLocation(this.progComposite, 'uBloomTex'),
        resolution: gl.getUniformLocation(this.progComposite, 'uResolution'),
        bloomStrength: gl.getUniformLocation(this.progComposite, 'uBloomStrength')
      };
    }

    _getRenderableFormat() {
      const gl = this.gl;
      if (this.useHalfFloat) {
        return {
          internalFormat: gl.RGBA16F,
          format: gl.RGBA,
          type: gl.HALF_FLOAT,
          filter: gl.LINEAR
        };
      }
      return {
        internalFormat: gl.RGBA8,
        format: gl.RGBA,
        type: gl.UNSIGNED_BYTE,
        filter: gl.LINEAR
      };
    }

    _initVolume3D() {
      const gl = this.gl;
      const fmt = this._getRenderableFormat();

      this.volTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_3D, this.volTex);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, fmt.filter);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, fmt.filter);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
      gl.texStorage3D(gl.TEXTURE_3D, 1, fmt.internalFormat, VOL_X, VOL_Y, VOL_Z);

      this.volSliceFbos = new Array(VOL_Z);
      for (let z = 0; z < VOL_Z; z++) {
        const fbo = gl.createFramebuffer();
        gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
        gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, this.volTex, 0, z);
        if (z === 0) {
          const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
          if (status !== gl.FRAMEBUFFER_COMPLETE && this.useHalfFloat) {
            // Fallback to RGBA8 if driver rejects RGBA16F 3D layer attachment
            this.useHalfFloat = false;
            gl.bindFramebuffer(gl.FRAMEBUFFER, null);
            gl.deleteFramebuffer(fbo);
            gl.deleteTexture(this.volTex);
            this._initVolume3D();
            return;
          }
        }
        this.volSliceFbos[z] = fbo;
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _create2DTarget(w, h) {
      const gl = this.gl;
      const fmt = this._getRenderableFormat();
      const tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, fmt.internalFormat, w, h, 0, fmt.format, fmt.type, null);

      const fbo = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      return { tex, fbo, w, h };
    }

    _destroyScreenTargets() {
      const gl = this.gl;
      if (this.hdrFbo) gl.deleteFramebuffer(this.hdrFbo);
      if (this.hdrTex) gl.deleteTexture(this.hdrTex);
      for (const f of this.bloomDownFbos) gl.deleteFramebuffer(f);
      for (const t of this.bloomDownTexs) gl.deleteTexture(t);
      for (const f of this.bloomUpFbos) gl.deleteFramebuffer(f);
      for (const t of this.bloomUpTexs) gl.deleteTexture(t);
      this.bloomDownFbos = [];
      this.bloomDownTexs = [];
      this.bloomUpFbos = [];
      this.bloomUpTexs = [];
      this.bloomSizes = [];
    }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2.0) * this.renderScale;
      const w = Math.max(64, Math.floor(this.canvas.clientWidth * dpr));
      const h = Math.max(64, Math.floor(this.canvas.clientHeight * dpr));
      if (w === this.width && h === this.height && this.hdrFbo) return;

      this.width = w;
      this.height = h;
      this.canvas.width = w;
      this.canvas.height = h;

      this._destroyScreenTargets();

      const hdr = this._create2DTarget(w, h);
      this.hdrTex = hdr.tex;
      this.hdrFbo = hdr.fbo;

      // 4-level Dual-Kawase bloom pyramid
      let bw = w, bh = h;
      for (let i = 0; i < 4; i++) {
        bw = Math.max(16, Math.floor(bw * 0.5));
        bh = Math.max(16, Math.floor(bh * 0.5));
        const down = this._create2DTarget(bw, bh);
        this.bloomDownTexs.push(down.tex);
        this.bloomDownFbos.push(down.fbo);
        this.bloomSizes.push([bw, bh]);
      }

      // 3 upsample targets (from level 3 -> 2 -> 1 -> 0)
      for (let i = 2; i >= 0; i--) {
        const [uw, uh] = this.bloomSizes[i];
        const up = this._create2DTarget(uw, uh);
        this.bloomUpTexs.push(up.tex);
        this.bloomUpFbos.push(up.fbo);
      }

      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    }

    setColors(waxHex, liquidHex, clarity) {
      if (waxHex) {
        this.waxHex = waxHex;
        this.waxColorLinear = hexToLinearRgb(waxHex);
      }
      if (liquidHex) {
        this.liquidHex = liquidHex;
        this.liquidColorLinear = hexToLinearRgb(liquidHex);
      }
      if (typeof clarity === 'number') {
        this.liquidClarity = clarity;
      }
    }

    orbit(deltaYaw, deltaPitch) {
      this.targetYaw -= deltaYaw;
      // Keep pitch between -0.34 rad (low dramatic tabletop view) and +0.85 rad (top-down inspection)
      this.targetPitch = Math.max(-0.34, Math.min(0.85, this.targetPitch + deltaPitch));
    }

    zoom(delta) {
      this.targetDist = Math.max(3.35, Math.min(10.8, this.targetDist * Math.exp(delta)));
    }

    _updateCamera(dt) {
      const blend = Math.min(1.0, dt * 12.0);
      this.camYaw += (this.targetYaw - this.camYaw) * blend;
      this.camPitch += (this.targetPitch - this.camPitch) * blend;
      this.camDist += (this.targetDist - this.camDist) * blend;

      // Architectural lens shift: center lamp in open canvas left of UI sidebar on desktop,
      // and glide smoothly to dead-center when Zen mode hides the UI!
      const cw = this.canvas.clientWidth || 1024;
      const ch = Math.max(1, this.canvas.clientHeight || 768);
      const isZen = document.body.classList.contains('zen-mode');
      let targetShiftX = 0.0;
      if (!isZen && cw > 720) {
        // Panel occupies ~326px on the right -> shift optical center left by ~163px in NDC-Y units!
        targetShiftX = (326.0 / ch) * 0.48;
      }
      this.lensShiftX += (targetShiftX - this.lensShiftX) * Math.min(1.0, dt * 9.0);

      const cp = Math.cos(this.camPitch);
      const sp = Math.sin(this.camPitch);
      const cy = Math.cos(this.camYaw);
      const sy = Math.sin(this.camYaw);

      const tx = this.camTarget[0];
      const ty = this.camTarget[1];
      const tz = this.camTarget[2];

      let cx = tx + this.camDist * cp * sy;
      let cyPos = ty + this.camDist * sp;
      let cz = tz + this.camDist * cp * cy;

      // Never allow camera below tabletop plane y = 0.16!
      if (cyPos < 0.16) cyPos = 0.16;

      this.camPos[0] = cx;
      this.camPos[1] = cyPos;
      this.camPos[2] = cz;

      // Forward = normalize(target - camPos)
      let fx = tx - cx;
      let fy = ty - cyPos;
      let fz = tz - cz;
      const fl = Math.hypot(fx, fy, fz) || 1.0;
      fx /= fl; fy /= fl; fz /= fl;

      // Right = normalize(cross(forward, worldUp))
      let rx = -fz;
      let ry = 0.0;
      let rz = fx;
      const rl = Math.hypot(rx, ry, rz) || 1.0;
      rx /= rl; rz /= rl;

      // Up = cross(right, forward)
      const ux = ry * fz - rz * fy;
      const uy = rz * fx - rx * fz;
      const uz = rx * fy - ry * fx;

      // Column-major mat3 for GLSL: col0 = right, col1 = up, col2 = -forward
      this.camMat[0] = rx;  this.camMat[1] = ry;  this.camMat[2] = rz;
      this.camMat[3] = ux;  this.camMat[4] = uy;  this.camMat[5] = uz;
      this.camMat[6] = -fx; this.camMat[7] = -fy; this.camMat[8] = -fz;
    }

    // Raycast screen pointer against the glass bottle interior for direct tactile swirling!
    hitTestGlassInterior(clientX, clientY) {
      const rect = this.canvas.getBoundingClientRect();
      const u = (clientX - rect.left) / Math.max(1, rect.width);
      const v = 1.0 - (clientY - rect.top) / Math.max(1, rect.height);
      const aspect = rect.width / Math.max(1, rect.height);
      const ndcX = (u * 2.0 - 1.0) * aspect + this.lensShiftX;
      const ndcY = (v * 2.0 - 1.0);

      const fovScale = Math.tan((this.fovDeg * 0.5 * Math.PI) / 180.0);
      const lx = ndcX * fovScale;
      const ly = ndcY * fovScale;
      const lz = -1.0;

      const m = this.camMat;
      let dx = m[0] * lx + m[3] * ly + m[6] * lz;
      let dy = m[1] * lx + m[4] * ly + m[7] * lz;
      let dz = m[2] * lx + m[5] * ly + m[8] * lz;
      const dl = Math.hypot(dx, dy, dz) || 1.0;
      dx /= dl; dy /= dl; dz /= dl;

      const ox = this.camPos[0], oy = this.camPos[1], oz = this.camPos[2];

      // Intersect ray with vertical axis plane / bottle mid-chord
      const a = dx * dx + dz * dz;
      if (a < 1e-6) return null;
      const tClosest = -(ox * dx + oz * dz) / a;
      if (tClosest <= 0.1) return null;

      const px = ox + dx * tClosest;
      const py = oy + dy * tClosest;
      const pz = oz + dz * tClosest;

      if (py < 1.54 || py > 4.60) return null;
      const glassR = window.LavaLamp.glassOuterRadius(py);
      const rAxis = Math.hypot(px, pz);
      if (rAxis > glassR) return null;

      return [px, py, pz];
    }

    render(physics, dt, elapsed) {
      this.resize();
      this._updateCamera(dt);

      const gl = this.gl;
      gl.bindVertexArray(this.emptyVao);
      gl.disable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);

      // 1. Upload packed parcel UBO once for both Volume & Scene shaders
      gl.bindBuffer(gl.UNIFORM_BUFFER, this.parcelUbo);
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, physics.uboData);

      // 2. Pass 1: Update 80x192x80 3D Wax Scalar Field Volume
      gl.useProgram(this.progVolume);
      gl.uniform1f(this.uVol.simTime, physics.simTime);
      gl.viewport(0, 0, VOL_X, VOL_Y);

      for (let z = 0; z < VOL_Z; z++) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.volSliceFbos[z]);
        gl.uniform1f(this.uVol.sliceW, (z + 0.5) / VOL_Z);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      }

      // 3. Pass 2: Full Photorealistic Raytraced Scene into HDR FBO
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.hdrFbo);
      gl.viewport(0, 0, this.width, this.height);
      gl.useProgram(this.progScene);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_3D, this.volTex);
      gl.uniform1i(this.uScene.waxVolume, 0);

      gl.uniform2f(this.uScene.resolution, this.width, this.height);
      gl.uniform3fv(this.uScene.camPos, this.camPos);
      gl.uniformMatrix3fv(this.uScene.camMat, false, this.camMat);
      gl.uniform1f(this.uScene.fovScale, Math.tan((this.fovDeg * 0.5 * Math.PI) / 180.0));
      gl.uniform2f(this.uScene.lensShift, this.lensShiftX, 0.0);
      gl.uniform1f(this.uScene.time, elapsed);

      gl.uniform1f(this.uScene.bulbGlow, physics.bulbGlow);
      gl.uniform1f(this.uScene.liquidTemp, physics.liquidTemp);
      gl.uniform3fv(this.uScene.waxColor, this.waxColorLinear);
      gl.uniform3fv(this.uScene.liquidColor, this.liquidColorLinear);
      gl.uniform1f(this.uScene.liquidClarity, this.liquidClarity);
      gl.uniform1i(this.uScene.metalType, this.metalType);
      gl.uniform1f(this.uScene.roomLight, this.roomLight);

      gl.drawArrays(gl.TRIANGLES, 0, 3);

      // 4. Pass 3: Dual-Kawase HDR Bloom Downsample Pyramid (4 levels)
      gl.useProgram(this.progBloomDown);
      gl.uniform1i(this.uBloomDown.source, 0);

      let prevTex = this.hdrTex;
      let prevW = this.width;
      let prevH = this.height;

      for (let i = 0; i < 4; i++) {
        const [bw, bh] = this.bloomSizes[i];
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomDownFbos[i]);
        gl.viewport(0, 0, bw, bh);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, prevTex);
        gl.uniform2f(this.uBloomDown.texelSize, 1.0 / prevW, 1.0 / prevH);
        gl.uniform1i(this.uBloomDown.firstPass, i === 0 ? 1 : 0);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        prevTex = this.bloomDownTexs[i];
        prevW = bw;
        prevH = bh;
      }

      // 5. Pass 4: Dual-Kawase Progressive Upsample Pyramid Blend (zero GL_BLEND needed)
      gl.useProgram(this.progBloomUp);
      gl.uniform1i(this.uBloomUp.source, 0);
      gl.uniform1i(this.uBloomUp.highMip, 1);

      let currUpTex = this.bloomDownTexs[3];
      let currUpW = this.bloomSizes[3][0];
      let currUpH = this.bloomSizes[3][1];

      for (let step = 0; step < 3; step++) {
        const targetLevel = 2 - step;
        const [uw, uh] = this.bloomSizes[targetLevel];

        gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomUpFbos[step]);
        gl.viewport(0, 0, uw, uh);

        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, currUpTex);

        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, this.bloomDownTexs[targetLevel]);

        gl.uniform2f(this.uBloomUp.texelSize, 1.0 / currUpW, 1.0 / currUpH);
        gl.drawArrays(gl.TRIANGLES, 0, 3);

        currUpTex = this.bloomUpTexs[step];
        currUpW = uw;
        currUpH = uh;
      }

      // 6. Pass 5: Final Filmic Composite to Screen
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, this.width, this.height);
      gl.useProgram(this.progComposite);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.hdrTex);
      gl.uniform1i(this.uComp.sceneTex, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, currUpTex);
      gl.uniform1i(this.uComp.bloomTex, 1);

      gl.uniform2f(this.uComp.resolution, this.width, this.height);
      gl.uniform1f(this.uComp.bloomStrength, this.bloomStrength);

      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }

  window.LavaLamp = window.LavaLamp || {};
  window.LavaLamp.LavaRenderer = LavaRenderer;
})();
