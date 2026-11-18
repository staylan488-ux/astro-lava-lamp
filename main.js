// ============================================================================
// Astro Physical Lava Lamp — Main Application Controller
// ============================================================================

(function () {
  'use strict';

  const PRESETS = [
    {
      name: 'Astro',
      wax: '#ff2b12',
      liquid: '#4d1ec2',
      clarity: 0.56,
      metal: 0 // Brushed Silver
    },
    {
      name: 'Magma',
      wax: '#eb170e',
      liquid: '#ffb319',
      clarity: 0.48,
      metal: 1 // Brushed Brass
    },
    {
      name: 'Cyber',
      wax: '#9dff14',
      liquid: '#0968c7',
      clarity: 0.62,
      metal: 0 // Silver
    },
    {
      name: 'Plum',
      wax: '#ff2a8d',
      liquid: '#17279e',
      clarity: 0.58,
      metal: 2 // Matte Black
    },
    {
      name: 'Solar',
      wax: '#ff7a14',
      liquid: '#fff2d6',
      clarity: 0.16,
      metal: 1 // Brass
    },
    {
      name: 'Obsidian',
      wax: '#c920ff',
      liquid: '#d6e4ff',
      clarity: 0.18,
      metal: 2 // Matte Black
    }
  ];

  function init() {
    const canvas = document.getElementById('lampCanvas');
    const errorBanner = document.getElementById('errorBanner');

    let physics, renderer;
    try {
      physics = new window.LavaLamp.LavaPhysics();
      renderer = new window.LavaLamp.LavaRenderer(canvas);
    } catch (err) {
      console.error(err);
      errorBanner.style.display = 'flex';
      errorBanner.textContent = 'Initialization Error: ' + (err && err.message ? err.message : String(err));
      return;
    }

    // UI Elements
    const statusDot = document.getElementById('statusDot');
    const thermoReadout = document.getElementById('thermoReadout');
    const btnToggleUi = document.getElementById('btnToggleUi');
    const btnFullscreen = document.getElementById('btnFullscreen');
    const btnPower = document.getElementById('btnPower');
    const btnNudge = document.getElementById('btnNudge');
    const btnWarmFlow = document.getElementById('btnWarmFlow');
    const btnColdStart = document.getElementById('btnColdStart');

    const sliderDimmer = document.getElementById('sliderDimmer');
    const dimmerLabel = document.getElementById('dimmerLabel');
    const sliderSpeed = document.getElementById('sliderSpeed');
    const speedLabel = document.getElementById('speedLabel');

    const presetGrid = document.getElementById('presetGrid');
    const colorWax = document.getElementById('colorWax');
    const colorLiquid = document.getElementById('colorLiquid');
    const sliderClarity = document.getElementById('sliderClarity');
    const clarityLabel = document.getElementById('clarityLabel');

    const metalButtons = document.getElementById('metalButtons');
    const sliderRoom = document.getElementById('sliderRoom');
    const roomLabel = document.getElementById('roomLabel');
    const sliderBloom = document.getElementById('sliderBloom');
    const bloomLabel = document.getElementById('bloomLabel');
    const qualityButtons = document.getElementById('qualityButtons');
    const qualityLabel = document.getElementById('qualityLabel');

    // Build Colorway Preset Cards
    function setActiveMetalUI(metalIdx) {
      renderer.metalType = metalIdx;
      metalButtons.querySelectorAll('.btn').forEach((b) => {
        b.classList.toggle('active', Number(b.dataset.metal) === metalIdx);
      });
    }

    function applyPreset(index) {
      const p = PRESETS[index];
      if (!p) return;
      renderer.setColors(p.wax, p.liquid, p.clarity);
      colorWax.value = p.wax;
      colorLiquid.value = p.liquid;
      sliderClarity.value = String(Math.round(p.clarity * 100));
      clarityLabel.textContent = Math.round(p.clarity * 100) + '%';
      setActiveMetalUI(p.metal);

      presetGrid.querySelectorAll('.preset-card').forEach((card, i) => {
        card.classList.toggle('active', i === index);
      });
    }

    PRESETS.forEach((p, i) => {
      const btn = document.createElement('button');
      btn.className = 'preset-card' + (i === 0 ? ' active' : '');
      const sw = document.createElement('span');
      sw.className = 'swatch-dual';
      sw.style.background = `linear-gradient(135deg, ${p.wax} 50%, ${p.liquid} 50%)`;
      const lbl = document.createElement('span');
      lbl.textContent = p.name;
      btn.appendChild(sw);
      btn.appendChild(lbl);
      btn.addEventListener('click', () => applyPreset(i));
      presetGrid.appendChild(btn);
    });

    applyPreset(0);

    // Clear active preset highlight when custom colors edited
    function clearPresetHighlight() {
      presetGrid.querySelectorAll('.preset-card').forEach((c) => c.classList.remove('active'));
    }

    colorWax.addEventListener('input', () => {
      clearPresetHighlight();
      renderer.setColors(colorWax.value, null, null);
    });

    colorLiquid.addEventListener('input', () => {
      clearPresetHighlight();
      renderer.setColors(null, colorLiquid.value, null);
    });

    sliderClarity.addEventListener('input', () => {
      const val = Number(sliderClarity.value) / 100;
      clarityLabel.textContent = Math.round(val * 100) + '%';
      renderer.setColors(null, null, val);
    });

    // Power & Dimmer
    function updatePowerUI() {
      btnPower.classList.toggle('active', physics.powerOn);
      btnPower.textContent = physics.powerOn ? 'Lamp Power: ON' : 'Lamp Power: OFF';
      statusDot.classList.toggle('off', !physics.powerOn || physics.dimmer < 0.02);
      const pct = Math.round(physics.dimmer * 100);
      const watts = physics.powerOn ? Math.round(40 * physics.dimmer) : 0;
      dimmerLabel.textContent = `${watts}W · ${pct}%`;
    }

    btnPower.addEventListener('click', () => {
      physics.powerOn = !physics.powerOn;
      updatePowerUI();
    });

    sliderDimmer.addEventListener('input', () => {
      physics.dimmer = Number(sliderDimmer.value) / 100;
      if (physics.dimmer > 0.02 && !physics.powerOn) {
        physics.powerOn = true;
      }
      updatePowerUI();
    });

    btnNudge.addEventListener('click', () => {
      physics.nudgeBottle(1.0);
    });

    btnWarmFlow.addEventListener('click', () => {
      physics.dimmer = 1.0;
      sliderDimmer.value = '100';
      physics.initWarmedUp();
      updatePowerUI();
      updateThermoUI();
    });

    btnColdStart.addEventListener('click', () => {
      if (physics.dimmer < 0.7) {
        physics.dimmer = 1.0;
        sliderDimmer.value = '100';
      }
      physics.initColdStart();
      updatePowerUI();
      updateThermoUI();
    });

    sliderSpeed.addEventListener('input', () => {
      const spd = Number(sliderSpeed.value) / 10;
      physics.timeScale = spd;
      speedLabel.textContent = spd.toFixed(1) + '×';
    });

    // Hardware & Environment
    metalButtons.querySelectorAll('.btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        setActiveMetalUI(Number(btn.dataset.metal));
      });
    });

    sliderRoom.addEventListener('input', () => {
      const v = Number(sliderRoom.value) / 100;
      renderer.roomLight = v;
      roomLabel.textContent = Math.round(v * 100) + '%';
    });

    sliderBloom.addEventListener('input', () => {
      const v = Number(sliderBloom.value) / 100;
      renderer.bloomStrength = v;
      bloomLabel.textContent = Math.round(v * 100) + '%';
    });

    qualityButtons.querySelectorAll('.btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const sc = Number(btn.dataset.scale);
        renderer.renderScale = sc;
        renderer.resize();
        qualityButtons.querySelectorAll('.btn').forEach((b) => b.classList.toggle('active', b === btn));
        const names = { '0.75': 'Perf (0.75×)', '1': 'High (1.0×)', '1.25': 'Ultra (1.25×)' };
        qualityLabel.textContent = names[String(sc)] || `${sc}×`;
      });
    });

    // Zen Mode & Fullscreen
    function toggleZenMode() {
      const isZen = document.body.classList.toggle('zen-mode');
      btnToggleUi.textContent = isZen ? 'Show UI' : 'Hide UI';
    }

    btnToggleUi.addEventListener('click', toggleZenMode);

    btnFullscreen.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });

    document.addEventListener('fullscreenchange', () => {
      btnFullscreen.textContent = document.fullscreenElement ? 'Exit Full' : 'Fullscreen';
    });

    window.addEventListener('keydown', (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      const k = e.key.toLowerCase();
      if (k === 'h') toggleZenMode();
      else if (k === 'f') btnFullscreen.click();
      else if (k === ' ') {
        e.preventDefault();
        physics.paused = !physics.paused;
      } else if (k === 't') {
        physics.nudgeBottle(1.0);
      } else if (k === 'c') {
        btnColdStart.click();
      }
    });

    // ========================================================================
    // Pointer Interactions:
    // - Drag directly on the glass bottle -> Tactile 3D fluid swirl!
    // - Drag anywhere in the room (or right-drag) -> Smooth Orbit Camera!
    // - Wheel / Pinch -> Smooth Camera Zoom!
    // ========================================================================
    let dragMode = null; // 'orbit' | 'swirl' | null
    let lastClientX = 0;
    let lastClientY = 0;
    let lastWorldHit = null;
    let activePointers = new Map();
    let prevPinchDist = 0;

    canvas.addEventListener('contextmenu', (e) => e.preventDefault());

    canvas.addEventListener('pointerdown', (e) => {
      canvas.setPointerCapture(e.pointerId);
      activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });

      if (activePointers.size === 2) {
        dragMode = 'pinch';
        physics.setPointerImpulse(false);
        const pts = Array.from(activePointers.values());
        prevPinchDist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        return;
      }

      lastClientX = e.clientX;
      lastClientY = e.clientY;

      if (e.button === 2 || e.altKey) {
        dragMode = 'orbit';
      } else {
        const hit = renderer.hitTestGlassInterior(e.clientX, e.clientY);
        if (hit) {
          dragMode = 'swirl';
          lastWorldHit = hit;
          canvas.style.cursor = 'pointer';
        } else {
          dragMode = 'orbit';
          canvas.style.cursor = 'grabbing';
        }
      }
    });

    canvas.addEventListener('pointermove', (e) => {
      if (activePointers.has(e.pointerId)) {
        activePointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      }

      if (dragMode === 'pinch' && activePointers.size === 2) {
        const pts = Array.from(activePointers.values());
        const dist = Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y);
        if (prevPinchDist > 10) {
          renderer.zoom((prevPinchDist - dist) * 0.0045);
        }
        prevPinchDist = dist;
        return;
      }

      if (!dragMode) {
        // Subtle hover cursor feedback over glass globe
        const hoverGlass = renderer.hitTestGlassInterior(e.clientX, e.clientY);
        canvas.style.cursor = hoverGlass ? 'pointer' : 'grab';
        return;
      }

      const dx = e.clientX - lastClientX;
      const dy = e.clientY - lastClientY;
      lastClientX = e.clientX;
      lastClientY = e.clientY;

      if (dragMode === 'orbit') {
        renderer.orbit(dx * 0.0055, dy * 0.0045);
      } else if (dragMode === 'swirl') {
        const hit = renderer.hitTestGlassInterior(e.clientX, e.clientY);
        if (hit && lastWorldHit) {
          const vx = (hit[0] - lastWorldHit[0]) * 18.0;
          const vy = (hit[1] - lastWorldHit[1]) * 18.0;
          const vz = (hit[2] - lastWorldHit[2]) * 18.0;
          physics.setPointerImpulse(true, hit, [vx, vy, vz]);
          lastWorldHit = hit;
        } else {
          physics.setPointerImpulse(false);
        }
      }
    });

    function endPointer(e) {
      activePointers.delete(e.pointerId);
      if (activePointers.size === 0) {
        dragMode = null;
        lastWorldHit = null;
        physics.setPointerImpulse(false);
        canvas.style.cursor = 'grab';
      }
    }

    canvas.addEventListener('pointerup', endPointer);
    canvas.addEventListener('pointercancel', endPointer);

    canvas.addEventListener('dblclick', (e) => {
      const hit = renderer.hitTestGlassInterior(e.clientX, e.clientY);
      if (hit) {
        physics.nudgeBottle(0.85);
      }
    });

    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      renderer.zoom(e.deltaY * 0.0011);
    }, { passive: false });

    window.addEventListener('resize', () => renderer.resize());

    // Live Thermometer Status Loop
    function updateThermoUI() {
      const info = physics.getThermometerReading();
      thermoReadout.textContent = `${info.celsius}°C · ${info.status}`;
    }
    updatePowerUI();
    updateThermoUI();

    // Main Animation Loop
    let prevTime = performance.now();
    let elapsed = 0.0;
    let lastUiTick = 0.0;

    function frame(now) {
      const dt = Math.min(0.05, Math.max(0.001, (now - prevTime) * 0.001));
      prevTime = now;
      elapsed += dt;

      physics.step(dt);
      renderer.render(physics, dt, elapsed);

      if (elapsed - lastUiTick > 0.25) {
        lastUiTick = elapsed;
        updateThermoUI();
      }

      requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
