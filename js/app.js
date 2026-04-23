import { NeuralNetwork } from './neural-network.js';
import { NetworkVisualizer } from './network-visualizer.js';
import { DrawingCanvas } from './drawing-canvas.js';
import {
  tryAutoLoad, loadFromFiles, getTrainingBatch, evaluateTestSet,
  isLoaded, DOWNLOAD_URLS, loadTestSamples
} from './mnist-data.js';

// ============================================================
// Slide Navigation
// ============================================================
let currentSlide = 0;
let totalSlides = 0;

function initSlides() {
  const slides = document.querySelectorAll('.slide');
  totalSlides = slides.length;
  const dotsContainer = document.getElementById('slide-dots');

  for (let i = 0; i < totalSlides; i++) {
    const dot = document.createElement('div');
    dot.className = 'slide-dot' + (i === 0 ? ' active' : '');
    dot.addEventListener('click', () => goToSlide(i));
    dotsContainer.appendChild(dot);
  }

  document.getElementById('prev-btn').addEventListener('click', () => goToSlide(currentSlide - 1));
  document.getElementById('next-btn').addEventListener('click', () => goToSlide(currentSlide + 1));

  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); goToSlide(currentSlide + 1); }
    if (e.key === 'ArrowLeft') { e.preventDefault(); goToSlide(currentSlide - 1); }
    if (e.key === 'Home') { e.preventDefault(); goToSlide(0); }
    if (e.key === 'End') { e.preventDefault(); goToSlide(totalSlides - 1); }
  });
}

function goToSlide(idx) {
  if (idx < 0 || idx >= totalSlides) return;
  const slides = document.querySelectorAll('.slide');
  const dots = document.querySelectorAll('.slide-dot');

  slides[currentSlide].classList.remove('active');
  dots[currentSlide].classList.remove('active');

  currentSlide = idx;

  slides[currentSlide].classList.add('active');
  dots[currentSlide].classList.add('active');

  document.getElementById('prev-btn').disabled = currentSlide === 0;
  document.getElementById('next-btn').disabled = currentSlide === totalSlides - 1;

  // Lazy-init demo on first visit to demo slide
  if (currentSlide === totalSlides - 1 && !demoInitialized) {
    initDemo();
  }

  // Resize visualizer when switching to demo slide
  if (currentSlide === totalSlides - 1 && visualizer) {
    setTimeout(() => { visualizer.resize(); visualizer.draw(network); }, 50);
  }

  // Lazy-init the new content slides
  const slide = document.querySelectorAll('.slide')[currentSlide];
  if (slide) {
    if (slide.querySelector('#digit-gallery') && !trainingDataInitialized) initTrainingDataSlide();
    if (slide.querySelector('#weight-grid') && !modelSlideInitialized) initModelSlide();
    if (slide.querySelector('.why-card.video-card')) initVideoCards(slide);
  }
}

// ============================================================
// Static Neural Network Diagram (Slide 3 - "What")
// ============================================================
function drawStaticDiagram() {
  const canvas = document.getElementById('diagram-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;

  ctx.fillStyle = '#0a0a1a';
  ctx.fillRect(0, 0, W, H);

  const layers = [4, 6, 6, 3];
  const labels = ['Input', 'Hidden 1', 'Hidden 2', 'Output'];
  const colors = ['#6c5ce7', '#a29bfe', '#a29bfe', '#00cec9'];
  const layerX = [];
  const pad = 60;

  for (let i = 0; i < layers.length; i++) {
    layerX.push(pad + (W - pad * 2) / (layers.length - 1) * i);
  }

  // Connections
  for (let l = 0; l < layers.length - 1; l++) {
    const fromCount = layers[l];
    const toCount = layers[l + 1];
    for (let f = 0; f < fromCount; f++) {
      for (let t = 0; t < toCount; t++) {
        const fy = H / 2 - (fromCount - 1) * 20 + f * 40;
        const ty = H / 2 - (toCount - 1) * 20 + t * 40;
        const alpha = 0.08 + Math.random() * 0.12;
        ctx.beginPath();
        ctx.moveTo(layerX[l] + 12, fy);
        ctx.lineTo(layerX[l + 1] - 12, ty);
        ctx.strokeStyle = `rgba(162, 155, 254, ${alpha})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  // Nodes
  for (let l = 0; l < layers.length; l++) {
    const count = layers[l];
    for (let n = 0; n < count; n++) {
      const y = H / 2 - (count - 1) * 20 + n * 40;
      ctx.beginPath();
      ctx.arc(layerX[l], y, 12, 0, Math.PI * 2);
      ctx.fillStyle = colors[l];
      ctx.globalAlpha = 0.7;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#4a4a8a';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // Labels
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'center';
  for (let l = 0; l < layers.length; l++) {
    ctx.fillStyle = '#8888bb';
    ctx.fillText(labels[l], layerX[l], H - 12);
  }
}

// ============================================================
// Key Concepts Visualizations (Slide 4)
// ============================================================
const CONCEPT_COLORS = {
  bg: '#0a0a1a',
  grid: 'rgba(162, 155, 254, 0.08)',
  axis: 'rgba(162, 155, 254, 0.4)',
  node: '#2d2d5e',
  nodeStroke: '#a29bfe',
  text: '#e0e0ff',
  textDim: '#8888bb',
  positive: '#00cec9',
  negative: '#ff6b6b',
  activation: '#ffeaa7',
  accent: '#a29bfe',
};

let weightState = { w: 0.75, phase: 0 };
let biasState = { b: 0.0 };
let activationState = { type: 'sigmoid' };
let combinedState = { x: 0.5 };

function initConceptDemos() {
  const wSlider = document.getElementById('weight-slider');
  const bSlider = document.getElementById('bias-slider');
  if (!wSlider || !bSlider) return;

  wSlider.addEventListener('input', (e) => {
    weightState.w = parseFloat(e.target.value);
    document.getElementById('weight-value').textContent = weightState.w.toFixed(2);
    document.getElementById('weight-w-readout').textContent = weightState.w.toFixed(2);
    const out = 1.0 * weightState.w;
    document.getElementById('weight-out-readout').textContent = out.toFixed(2);
    drawWeightDemo();
    updateCombinedDemo();
  });

  bSlider.addEventListener('input', (e) => {
    biasState.b = parseFloat(e.target.value);
    document.getElementById('bias-value').textContent = biasState.b.toFixed(2);
    document.getElementById('bias-b-readout').textContent = 'b = ' + biasState.b.toFixed(2);
    drawBiasDemo();
    updateCombinedDemo();
  });

  document.querySelectorAll('.act-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.act-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activationState.type = btn.dataset.act;
      updateActivationLabels();
      drawActivationDemo();
      updateCombinedDemo();
    });
  });

  const xSlider = document.getElementById('input-x-slider');
  if (xSlider) {
    xSlider.addEventListener('input', (e) => {
      combinedState.x = parseFloat(e.target.value);
      document.getElementById('input-x-value').textContent = combinedState.x.toFixed(2);
      updateCombinedDemo();
    });
  }

  // Initial draw
  drawWeightDemo();
  drawBiasDemo();
  drawActivationDemo();
  updateCombinedDemo();

  // Animate the weight pulse and combined pipeline continuously
  function animate() {
    weightState.phase = (weightState.phase + 0.008) % 1;
    drawWeightDemo();
    drawCombinedDemo();
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  // Redraw the non-animated demos when the window resizes
  let resizeTimer;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      drawBiasDemo();
      drawActivationDemo();
    }, 100);
  });
}

function updateCombinedDemo() {
  const w = weightState.w;
  const b = biasState.b;
  const x = combinedState.x;
  const z = w * x + b;
  const y = applyActivation(z, activationState.type);

  const actNames = { sigmoid: 'σ', relu: 'ReLU', tanh: 'tanh' };
  const actSymbol = actNames[activationState.type] || 'f';

  const els = {
    w: document.getElementById('combined-w'),
    b: document.getElementById('combined-b'),
    act: document.getElementById('combined-act'),
    actname: document.getElementById('combined-actname'),
    y: document.getElementById('combined-y'),
    eq: document.getElementById('combined-equation'),
  };
  if (!els.w) return;

  els.w.textContent = w.toFixed(2);
  els.b.textContent = b.toFixed(2);
  els.act.textContent = activationState.type;
  els.actname.textContent = actSymbol;
  els.y.textContent = y.toFixed(3);

  els.eq.innerHTML =
    `<span class="eq-step">${w.toFixed(2)} &middot; ${x.toFixed(2)} + ${b.toFixed(2)} = ${z.toFixed(3)}</span>` +
    `<span class="eq-arrow">&rarr;</span>` +
    `<span class="eq-result">y = ${y.toFixed(3)}</span>`;

  drawCombinedDemo();
}

function drawCombinedDemo() {
  const canvas = document.getElementById('combined-canvas');
  if (!canvas) return;
  const { ctx, w: W, h: H } = setupConceptCanvas(canvas);

  ctx.fillStyle = CONCEPT_COLORS.bg;
  ctx.fillRect(0, 0, W, H);

  const weight = weightState.w;
  const bias = biasState.b;
  const x = combinedState.x;
  const z = weight * x + bias;
  const y = applyActivation(z, activationState.type);
  const actNames = { sigmoid: 'σ(·)', relu: 'ReLU', tanh: 'tanh' };

  // Layout: 4 operation stages between 5 nodes
  // [x] →(×w)→ [w·x] →(+b)→ [z] →(f)→ [y]
  const nodeCount = 5;
  const leftPad = 42;
  const rightPad = 42;
  const spacing = (W - leftPad - rightPad) / (nodeCount - 1);
  const cy = H / 2;
  const r = 22;

  const nodes = [
    { x: leftPad + spacing * 0, label: 'x',       value: x,  color: '#6c5ce7' },
    { x: leftPad + spacing * 1, label: 'w·x',     value: weight * x, color: CONCEPT_COLORS.positive },
    { x: leftPad + spacing * 2, label: 'z',       value: z,  color: CONCEPT_COLORS.activation },
    { x: leftPad + spacing * 3, label: actNames[activationState.type] || 'f', value: null, color: CONCEPT_COLORS.accent, isOp: true },
    { x: leftPad + spacing * 4, label: 'y',       value: y,  color: CONCEPT_COLORS.positive, isOutput: true },
  ];
  const ops = ['× w', '+ b', 'activate', ''];

  // Connections with pulse
  const phase = weightState.phase;
  for (let i = 0; i < nodes.length - 1; i++) {
    const a = nodes[i];
    const b = nodes[i + 1];
    ctx.beginPath();
    ctx.moveTo(a.x + r, cy);
    ctx.lineTo(b.x - r, cy);
    ctx.strokeStyle = 'rgba(162, 155, 254, 0.35)';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.stroke();

    // operation label above the line
    if (ops[i]) {
      ctx.fillStyle = CONCEPT_COLORS.textDim;
      ctx.font = '600 11px ui-monospace, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(ops[i], (a.x + b.x) / 2, cy - 14);
    }

    // pulse
    const px = (a.x + r) + ((b.x - r) - (a.x + r)) * ((phase + i * 0.25) % 1);
    ctx.beginPath();
    ctx.arc(px, cy, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = CONCEPT_COLORS.accent;
    ctx.shadowColor = CONCEPT_COLORS.accent;
    ctx.shadowBlur = 8;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  // Nodes
  for (const n of nodes) {
    const intensity = n.value !== null ? Math.min(Math.abs(n.value), 2) / 2 : 0.7;
    drawNeuron(ctx, n.x, cy, r, n.color, intensity);

    // Value or operation label inside
    ctx.fillStyle = CONCEPT_COLORS.text;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (n.isOp) {
      ctx.font = '600 11px ui-monospace, monospace';
      ctx.fillText(n.label, n.x, cy);
    } else {
      ctx.font = '600 13px system-ui, sans-serif';
      ctx.fillText(n.value.toFixed(2), n.x, cy);
    }

    // Label below
    ctx.fillStyle = CONCEPT_COLORS.textDim;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textBaseline = 'top';
    const subLabel = n.isOp ? 'activation' : n.label;
    ctx.fillText(subLabel, n.x, cy + r + 6);

    // Highlight output
    if (n.isOutput) {
      ctx.strokeStyle = hexWithAlpha(CONCEPT_COLORS.positive, 0.6);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(n.x, cy, r + 4, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
}

function setupConceptCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  // Use clientWidth/Height (CSS pixels) as the source of truth.
  // Fall back to the declared attribute size when the element isn't laid out yet
  // (e.g. parent slide is display:none on initial draw).
  const w = canvas.clientWidth || parseInt(canvas.getAttribute('width'), 10) || 340;
  const h = canvas.clientHeight || parseInt(canvas.getAttribute('height'), 10) || 180;
  const targetW = Math.round(w * dpr);
  const targetH = Math.round(h * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

function drawWeightDemo() {
  const canvas = document.getElementById('weight-canvas');
  if (!canvas) return;
  const { ctx, w, h } = setupConceptCanvas(canvas);
  const { w: weight, phase } = weightState;

  ctx.fillStyle = CONCEPT_COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  const inX = w * 0.18;
  const outX = w * 0.82;
  const cy = h * 0.5;
  const r = 22;

  // Connection line
  const mag = Math.min(Math.abs(weight), 2) / 2;
  const thickness = 1.5 + mag * 7;
  const lineColor = weight >= 0 ? CONCEPT_COLORS.positive : CONCEPT_COLORS.negative;
  const alpha = 0.3 + mag * 0.7;

  ctx.beginPath();
  ctx.moveTo(inX + r, cy);
  ctx.lineTo(outX - r, cy);
  ctx.strokeStyle = hexWithAlpha(lineColor, alpha);
  ctx.lineWidth = thickness;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Pulse traveling along the line
  if (Math.abs(weight) > 0.02) {
    const direction = weight >= 0 ? 1 : -1;
    const p = direction > 0 ? phase : 1 - phase;
    const px = (inX + r) + ((outX - r) - (inX + r)) * p;
    const pulseR = 3 + mag * 4;
    ctx.beginPath();
    ctx.arc(px, cy, pulseR, 0, Math.PI * 2);
    ctx.fillStyle = hexWithAlpha(lineColor, 0.9);
    ctx.shadowColor = lineColor;
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  // Input neuron
  drawNeuron(ctx, inX, cy, r, '#6c5ce7');
  ctx.fillStyle = CONCEPT_COLORS.text;
  ctx.font = '600 14px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('1.0', inX, cy);
  ctx.fillStyle = CONCEPT_COLORS.textDim;
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText('input', inX, cy + r + 14);

  // Output neuron
  const out = 1.0 * weight;
  drawNeuron(ctx, outX, cy, r, weight >= 0 ? CONCEPT_COLORS.positive : CONCEPT_COLORS.negative, Math.min(Math.abs(out), 2) / 2);
  ctx.fillStyle = CONCEPT_COLORS.text;
  ctx.font = '600 13px system-ui, sans-serif';
  ctx.fillText(out.toFixed(2), outX, cy);
  ctx.fillStyle = CONCEPT_COLORS.textDim;
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText('output', outX, cy + r + 14);

  // Weight label on connection
  ctx.fillStyle = CONCEPT_COLORS.textDim;
  ctx.font = '11px ui-monospace, monospace';
  ctx.fillText(`w = ${weight.toFixed(2)}`, (inX + outX) / 2, cy - thickness / 2 - 10);
}

function drawBiasDemo() {
  const canvas = document.getElementById('bias-canvas');
  if (!canvas) return;
  const { ctx, w, h } = setupConceptCanvas(canvas);
  const { b } = biasState;

  ctx.fillStyle = CONCEPT_COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  // Plot area
  const pad = 22;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;
  const xMin = -6, xMax = 6;
  const yMin = 0, yMax = 1;

  const xToPx = (x) => pad + ((x - xMin) / (xMax - xMin)) * plotW;
  const yToPx = (y) => pad + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  drawGrid(ctx, pad, plotW, plotH, xMin, xMax, yMin, yMax, 2, 0.25);

  // Reference sigmoid (b = 0) - faint
  ctx.beginPath();
  for (let i = 0; i <= 120; i++) {
    const x = xMin + (xMax - xMin) * (i / 120);
    const y = sigmoid(x);
    const px = xToPx(x);
    const py = yToPx(y);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = 'rgba(162, 155, 254, 0.25)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Shifted sigmoid σ(x + b)
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (xMax - xMin) * (i / 200);
    const y = sigmoid(x + b);
    const px = xToPx(x);
    const py = yToPx(y);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = CONCEPT_COLORS.activation;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = CONCEPT_COLORS.activation;
  ctx.shadowBlur = 8;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Firing midpoint (where σ = 0.5, i.e. x = -b)
  const fireX = -b;
  if (fireX >= xMin && fireX <= xMax) {
    const fx = xToPx(fireX);
    const fy = yToPx(0.5);
    ctx.beginPath();
    ctx.moveTo(fx, yToPx(0));
    ctx.lineTo(fx, yToPx(1));
    ctx.strokeStyle = 'rgba(255, 234, 167, 0.3)';
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.beginPath();
    ctx.arc(fx, fy, 4, 0, Math.PI * 2);
    ctx.fillStyle = CONCEPT_COLORS.activation;
    ctx.fill();
  }

  // Axis labels
  ctx.fillStyle = CONCEPT_COLORS.textDim;
  ctx.font = '10px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText('1', pad - 4, yToPx(1));
  ctx.fillText('0', pad - 4, yToPx(0));
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('x', w - pad + 10, yToPx(0) - 6);
}

function drawActivationDemo() {
  const canvas = document.getElementById('activation-canvas');
  if (!canvas) return;
  const { ctx, w, h } = setupConceptCanvas(canvas);
  const type = activationState.type;

  ctx.fillStyle = CONCEPT_COLORS.bg;
  ctx.fillRect(0, 0, w, h);

  const pad = 22;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;
  const xMin = -5, xMax = 5;
  let yMin = -1.2, yMax = 1.2;
  if (type === 'sigmoid') { yMin = -0.2; yMax = 1.2; }
  if (type === 'relu') { yMin = -0.5; yMax = 4; }

  const xToPx = (x) => pad + ((x - xMin) / (xMax - xMin)) * plotW;
  const yToPx = (y) => pad + plotH - ((y - yMin) / (yMax - yMin)) * plotH;

  // Grid and axes
  drawGrid(ctx, pad, plotW, plotH, xMin, xMax, yMin, yMax, 1, (yMax - yMin) / 4);

  // x-axis at y=0
  if (yMin < 0 && yMax > 0) {
    ctx.beginPath();
    ctx.moveTo(pad, yToPx(0));
    ctx.lineTo(pad + plotW, yToPx(0));
    ctx.strokeStyle = CONCEPT_COLORS.axis;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  // y-axis at x=0
  ctx.beginPath();
  ctx.moveTo(xToPx(0), pad);
  ctx.lineTo(xToPx(0), pad + plotH);
  ctx.strokeStyle = CONCEPT_COLORS.axis;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Curve
  ctx.beginPath();
  for (let i = 0; i <= 200; i++) {
    const x = xMin + (xMax - xMin) * (i / 200);
    const y = applyActivation(x, type);
    const px = xToPx(x);
    const py = yToPx(y);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.strokeStyle = CONCEPT_COLORS.positive;
  ctx.lineWidth = 2.5;
  ctx.shadowColor = CONCEPT_COLORS.positive;
  ctx.shadowBlur = 10;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Axis labels
  ctx.fillStyle = CONCEPT_COLORS.textDim;
  ctx.font = '10px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('z', pad + plotW + 6, yToPx(0) - 4);
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText(yMax.toFixed(yMax >= 10 ? 0 : 1), pad - 4, yToPx(yMax));
  ctx.fillText(yMin.toFixed(yMin <= -10 ? 0 : 1), pad - 4, yToPx(yMin));
}

function updateActivationLabels() {
  const type = activationState.type;
  const formulaEl = document.getElementById('activation-formula');
  const readoutEl = document.getElementById('activation-readout');
  if (!formulaEl || !readoutEl) return;
  if (type === 'sigmoid') {
    formulaEl.innerHTML = '&sigma;(z) = 1 / (1 + e<sup>&minus;z</sup>)';
    readoutEl.textContent = 'Squashes any input into (0, 1).';
  } else if (type === 'relu') {
    formulaEl.innerHTML = 'ReLU(z) = max(0, z)';
    readoutEl.textContent = 'Passes positives through, zeros out negatives.';
  } else if (type === 'tanh') {
    formulaEl.innerHTML = 'tanh(z) = (e<sup>z</sup> &minus; e<sup>&minus;z</sup>) / (e<sup>z</sup> + e<sup>&minus;z</sup>)';
    readoutEl.textContent = 'Squashes input into (-1, 1), zero-centered.';
  }
}

function applyActivation(x, type) {
  if (type === 'sigmoid') return sigmoid(x);
  if (type === 'relu') return Math.max(0, x);
  if (type === 'tanh') return Math.tanh(x);
  return x;
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function drawGrid(ctx, pad, plotW, plotH, xMin, xMax, yMin, yMax, xStep, yStep) {
  ctx.strokeStyle = CONCEPT_COLORS.grid;
  ctx.lineWidth = 1;
  for (let x = Math.ceil(xMin / xStep) * xStep; x <= xMax; x += xStep) {
    const px = pad + ((x - xMin) / (xMax - xMin)) * plotW;
    ctx.beginPath();
    ctx.moveTo(px, pad);
    ctx.lineTo(px, pad + plotH);
    ctx.stroke();
  }
  for (let y = Math.ceil(yMin / yStep) * yStep; y <= yMax; y += yStep) {
    const py = pad + plotH - ((y - yMin) / (yMax - yMin)) * plotH;
    ctx.beginPath();
    ctx.moveTo(pad, py);
    ctx.lineTo(pad + plotW, py);
    ctx.stroke();
  }
}

function drawNeuron(ctx, x, y, r, color, intensity = 0.7) {
  const grad = ctx.createRadialGradient(x, y, 2, x, y, r);
  grad.addColorStop(0, hexWithAlpha(color, 0.4 + intensity * 0.4));
  grad.addColorStop(1, hexWithAlpha(color, 0.15));
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();
  ctx.strokeStyle = hexWithAlpha(color, 0.9);
  ctx.lineWidth = 1.5;
  ctx.stroke();
}

function hexWithAlpha(hex, alpha) {
  if (hex.startsWith('rgba')) return hex;
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map(c => c + c).join('');
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ============================================================
// Why Slide Video Cards (lazy-load on first reveal)
// ============================================================
const initializedVideoCards = new WeakSet();

function initVideoCards(slide) {
  const cards = slide.querySelectorAll('.why-card.video-card');
  cards.forEach(card => {
    if (initializedVideoCards.has(card)) return;
    initializedVideoCards.add(card);

    const container = card.querySelector('.why-card-video');
    if (!container) return;
    const type = card.dataset.videoType;

    if (type === 'local') {
      const src = card.dataset.videoSrc;
      const v = document.createElement('video');
      v.src = src;
      v.muted = true;
      v.loop = true;
      v.autoplay = true;
      v.playsInline = true;
      v.preload = 'auto';
      v.setAttribute('disablepictureinpicture', '');
      v.addEventListener('loadeddata', () => v.classList.add('loaded'));
      container.appendChild(v);
      // Some browsers require an explicit play() call after element is in DOM
      v.play().catch(() => { /* autoplay blocked - will need user interaction */ });
    } else if (type === 'youtube') {
      const id = card.dataset.videoId;
      const iframe = document.createElement('iframe');
      const params = new URLSearchParams({
        autoplay: '1',
        mute: '1',
        loop: '1',
        playlist: id,           // Required for loop to work
        controls: '0',          // Hide play/pause/progress bar
        modestbranding: '1',    // Reduce YouTube branding
        showinfo: '0',          // Legacy: hide video title
        rel: '0',               // No "More videos" overlay (same channel only)
        iv_load_policy: '3',    // Hide annotations
        playsinline: '1',       // Inline play on iOS
        disablekb: '1',         // Disable keyboard controls
        fs: '0',                // Disable fullscreen button
        cc_load_policy: '0',    // Don't auto-show captions
        autohide: '1',          // Hide controls after a few seconds (legacy)
      });
      iframe.src = `https://www.youtube-nocookie.com/embed/${id}?${params.toString()}`;
      iframe.allow = 'autoplay; encrypted-media';
      iframe.setAttribute('frameborder', '0');
      iframe.setAttribute('tabindex', '-1');
      iframe.title = 'Video';
      iframe.addEventListener('load', () => iframe.classList.add('loaded'));
      container.appendChild(iframe);

      // Apply optional per-card zoom (e.g. videos with built-in side letterboxing)
      const zoom = parseFloat(card.dataset.zoom);
      if (!isNaN(zoom) && zoom !== 1) {
        iframe.style.setProperty('--video-zoom', String(zoom));
      }
    }
  });
}

// ============================================================
// Training Data Slide (Slide 7)
// ============================================================
let trainingDataInitialized = false;
let trainingDataSamples = null;
let selectedSampleIdx = 0;

async function initTrainingDataSlide() {
  if (trainingDataInitialized) return;
  trainingDataInitialized = true;

  const gallery = document.getElementById('digit-gallery');
  const pixelCanvas = document.getElementById('pixel-grid-canvas');
  const labelEl = document.getElementById('selected-digit-label');
  if (!gallery || !pixelCanvas) return;

  gallery.innerHTML = '<div class="weight-grid-status">Loading MNIST samples...</div>';

  try {
    trainingDataSamples = await loadTestSamples(12);
  } catch (e) {
    gallery.innerHTML = `<div class="weight-grid-status">Failed to load samples: ${e.message}</div>`;
    return;
  }

  gallery.innerHTML = '';
  trainingDataSamples.forEach((sample, idx) => {
    const tile = document.createElement('div');
    tile.className = 'digit-tile' + (idx === selectedSampleIdx ? ' active' : '');
    tile.innerHTML = `
      <canvas width="28" height="28"></canvas>
      <span class="digit-tile-label">${sample.label}</span>
    `;
    const c = tile.querySelector('canvas');
    drawPixelArray(c, sample.pixels, 28, 28, false);
    tile.addEventListener('click', () => {
      selectedSampleIdx = idx;
      gallery.querySelectorAll('.digit-tile').forEach((t, i) => {
        t.classList.toggle('active', i === idx);
      });
      drawPixelGridDetail(pixelCanvas, sample);
      if (labelEl) labelEl.textContent = `digit "${sample.label}"`;
    });
    gallery.appendChild(tile);
  });

  // Initial detail draw
  if (trainingDataSamples.length > 0) {
    drawPixelGridDetail(pixelCanvas, trainingDataSamples[0]);
    if (labelEl) labelEl.textContent = `digit "${trainingDataSamples[0].label}"`;
  }
}

function drawPixelArray(canvas, pixels, w, h, invert) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(w, h);
  for (let i = 0; i < pixels.length; i++) {
    const v = invert ? (1 - pixels[i]) : pixels[i];
    const px = Math.round(v * 255);
    imageData.data[i * 4] = px;
    imageData.data[i * 4 + 1] = px;
    imageData.data[i * 4 + 2] = px;
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function drawPixelGridDetail(canvas, sample) {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 420;
  const cssH = canvas.clientHeight || 420;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cssW, cssH);

  const cellW = cssW / 28;
  const cellH = cssH / 28;
  const showText = cellW > 13;

  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = sample.pixels[y * 28 + x];
      const px = Math.round(v * 255);
      ctx.fillStyle = `rgb(${px},${px},${px})`;
      ctx.fillRect(x * cellW, y * cellH, cellW, cellH);

      if (showText && v > 0.05) {
        // Show numeric value (0-99) on lighter pixels
        ctx.fillStyle = v > 0.5 ? '#000' : '#a29bfe';
        ctx.font = `${Math.min(cellW * 0.42, 10)}px ui-monospace, monospace`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const display = Math.round(v * 99);
        ctx.fillText(String(display), x * cellW + cellW / 2, y * cellH + cellH / 2);
      }
    }
  }

  // Subtle grid lines
  ctx.strokeStyle = 'rgba(108, 92, 231, 0.15)';
  ctx.lineWidth = 0.5;
  for (let i = 1; i < 28; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cellW, 0);
    ctx.lineTo(i * cellW, cssH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * cellH);
    ctx.lineTo(cssW, i * cellH);
    ctx.stroke();
  }
}

// ============================================================
// Trained Model Slide (Slide 8)
// ============================================================
let modelSlideInitialized = false;

async function initModelSlide() {
  if (modelSlideInitialized) return;
  modelSlideInitialized = true;

  const status = document.getElementById('weight-grid-status');
  const grid = document.getElementById('weight-grid');
  const snippet = document.getElementById('json-snippet');
  const stats = document.getElementById('model-stats');
  const paramCount = document.getElementById('param-count');
  if (!status || !grid || !snippet) return;

  try {
    status.textContent = 'Fetching mnist-model-1776705741158.json...';
    const resp = await fetch(PRETRAINED_MODEL_URL);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const text = await resp.text();
    const data = JSON.parse(text);

    // Render weight tiles for first 12 hidden-layer-1 neurons
    status.classList.add('hidden');
    grid.innerHTML = '';
    const firstLayerWeights = data.weights[0]; // [128][784]
    const tileCount = Math.min(12, firstLayerWeights.length);

    for (let i = 0; i < tileCount; i++) {
      const tile = document.createElement('div');
      tile.className = 'weight-tile';
      tile.title = `Hidden layer 1, neuron ${i}`;
      const c = document.createElement('canvas');
      c.width = 28;
      c.height = 28;
      tile.appendChild(c);
      drawWeightHeatmap(c, firstLayerWeights[i]);
      grid.appendChild(tile);
    }

    // Compute stats
    let totalWeights = 0, totalBiases = 0;
    for (const layer of data.weights) {
      for (const row of layer) totalWeights += row.length;
    }
    for (const b of data.biases) totalBiases += b.length;
    const totalParams = totalWeights + totalBiases;

    if (paramCount) paramCount.textContent = totalParams.toLocaleString();

    // Render JSON snippet (truncated)
    snippet.innerHTML = renderJsonSnippet(data);

    // Render stat tiles
    if (stats) {
      stats.innerHTML = `
        <div class="model-stat">
          <span class="model-stat-label">Architecture</span>
          <span class="model-stat-value">${data.layerSizes.join(' → ')}</span>
        </div>
        <div class="model-stat">
          <span class="model-stat-label">Total Parameters</span>
          <span class="model-stat-value">${totalParams.toLocaleString()}</span>
        </div>
        <div class="model-stat">
          <span class="model-stat-label">Weights</span>
          <span class="model-stat-value">${totalWeights.toLocaleString()}</span>
        </div>
        <div class="model-stat">
          <span class="model-stat-label">Biases</span>
          <span class="model-stat-value">${totalBiases.toLocaleString()}</span>
        </div>
      `;
    }
  } catch (e) {
    status.textContent = `Failed to load model: ${e.message}`;
    snippet.textContent = `Error: ${e.message}`;
  }
}

function drawWeightHeatmap(canvas, weights) {
  // weights: array of 784 floats. Map to 28x28 with diverging colormap (red = neg, cyan = pos).
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(28, 28);
  // Find max abs value for normalization
  let maxAbs = 0.001;
  for (const w of weights) maxAbs = Math.max(maxAbs, Math.abs(w));
  for (let i = 0; i < 784; i++) {
    const v = weights[i] / maxAbs; // -1 to 1
    let r, g, b;
    if (v >= 0) {
      // Positive: black to cyan
      r = 0;
      g = Math.round(206 * v);
      b = Math.round(201 * v);
    } else {
      // Negative: black to red
      r = Math.round(255 * -v);
      g = Math.round(107 * -v);
      b = Math.round(107 * -v);
    }
    imageData.data[i * 4] = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
}

function renderJsonSnippet(data) {
  // Render an abbreviated, syntax-highlighted view of the JSON
  const layerSizes = JSON.stringify(data.layerSizes);
  const w0_preview = data.weights[0][0].slice(0, 4).map(v => v.toFixed(3)).join(', ');
  const w0_total = data.weights[0].length;
  const b0_preview = Array.from(data.biases[0]).slice(0, 4).map(v => v.toFixed(3)).join(', ');
  const b0_total = data.biases[0].length;

  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  return [
    '{',
    `  <span class="jk">"layerSizes"</span>: <span class="jn">${esc(layerSizes)}</span>,`,
    `  <span class="jk">"weights"</span>: [`,
    `    <span class="jp">// layer 0: ${w0_total} neurons × 784 inputs</span>`,
    `    [`,
    `      [<span class="jn">${esc(w0_preview)}</span>, <span class="jp">... 780 more</span>],`,
    `      <span class="jp">// ... ${w0_total - 1} more rows</span>`,
    `    ],`,
    `    <span class="jp">// layer 1: 64 neurons × 128 inputs</span>`,
    `    [<span class="jp">...</span>],`,
    `    <span class="jp">// layer 2: 10 neurons × 64 inputs</span>`,
    `    [<span class="jp">...</span>]`,
    `  ],`,
    `  <span class="jk">"biases"</span>: [`,
    `    [<span class="jn">${esc(b0_preview)}</span>, <span class="jp">... ${b0_total - 4} more</span>],`,
    `    [<span class="jp">... 64 values</span>],`,
    `    [<span class="jp">... 10 values</span>]`,
    `  ]`,
    '}'
  ].join('\n');
}

// ============================================================
// Live Demo (Slide 9)
// ============================================================
let network = null;
let drawCanvas = null;
let visualizer = null;
let isTraining = false;
let stopRequested = false;
let modelTrained = false;
let pendingFiles = null;
let demoInitialized = false;

let trainBtn, stopBtn, saveBtn, loadFileBtn, loadUrlBtn;
let statusDot, statusText, modelInfo;
let progressArea, progressFill, progressEpoch, progressAccuracy, progressLoss;
let mnistPanel, fileStatus;

function initDemo() {
  if (demoInitialized) return;
  demoInitialized = true;

  drawCanvas = new DrawingCanvas('draw-canvas');
  visualizer = new NetworkVisualizer(document.getElementById('network-canvas'));

  trainBtn = document.getElementById('train-btn');
  stopBtn = document.getElementById('stop-btn');
  saveBtn = document.getElementById('save-btn');
  loadFileBtn = document.getElementById('load-file-btn');
  loadUrlBtn = document.getElementById('load-url-btn');
  statusDot = document.querySelector('.status-dot');
  statusText = document.getElementById('status-text');
  modelInfo = document.getElementById('model-info');
  progressArea = document.getElementById('progress-area');
  progressFill = document.getElementById('progress-fill');
  progressEpoch = document.getElementById('progress-epoch');
  progressAccuracy = document.getElementById('progress-accuracy');
  progressLoss = document.getElementById('progress-loss');
  mnistPanel = document.getElementById('mnist-panel');
  fileStatus = document.getElementById('file-status');

  setupOutputBars();
  setupDemoControls();
  setupDownloadLinks();
  updateButtons();
  visualizer.draw(network);
}

function setupOutputBars() {
  const container = document.getElementById('output-bars');
  for (let i = 0; i < 10; i++) {
    const row = document.createElement('div');
    row.className = 'output-bar-row';
    row.innerHTML = `
      <span class="output-bar-label">${i}</span>
      <div class="output-bar-track">
        <div class="output-bar-fill" id="bar-fill-${i}" style="width:0%"></div>
      </div>
      <span class="output-bar-pct" id="bar-pct-${i}">0%</span>
    `;
    container.appendChild(row);
  }
}

function setupDownloadLinks() {
  const list = document.getElementById('download-list');
  for (const item of DOWNLOAD_URLS) {
    const li = document.createElement('li');
    li.innerHTML = `<a href="${item.url}" target="_blank" rel="noopener">${item.name}</a> <span class="dl-desc">${item.desc}</span>`;
    list.appendChild(li);
  }
}

function setupDemoControls() {
  trainBtn.addEventListener('click', startTraining);
  stopBtn.addEventListener('click', () => { stopRequested = true; });
  saveBtn.addEventListener('click', saveModelToFile);
  loadFileBtn.addEventListener('click', () => document.getElementById('model-file-input').click());
  loadUrlBtn.addEventListener('click', openUrlModal);
  setupUrlModal();

  document.getElementById('model-file-input').addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) loadModelFromFile(file);
    e.target.value = '';
  });

  document.getElementById('clear-btn').addEventListener('click', () => {
    drawCanvas.clear();
    document.getElementById('prediction-value').textContent = '-';
    document.getElementById('confidence-value').textContent = '-';
    resetBars();
    visualizer.draw(network);
  });

  document.getElementById('draw-canvas').addEventListener('mouseup', () => {
    if (drawCanvas.hasDrawing && modelTrained) setTimeout(predict, 50);
  });
  document.getElementById('draw-canvas').addEventListener('touchend', () => {
    if (drawCanvas.hasDrawing && modelTrained) setTimeout(predict, 50);
  });

  document.getElementById('show-weights').addEventListener('change', (e) => {
    visualizer.showWeights = e.target.checked;
    visualizer.draw(network);
  });
  document.getElementById('show-values').addEventListener('change', (e) => {
    visualizer.showValues = e.target.checked;
    visualizer.draw(network);
  });

  document.getElementById('mnist-files').addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;
    const names = files.map(f => f.name);
    const hasTrainImg = names.includes('train-images-idx3-ubyte.gz');
    const hasTrainLbl = names.includes('train-labels-idx1-ubyte.gz');
    const hasTestImg = names.includes('t10k-images-idx3-ubyte.gz');
    const hasTestLbl = names.includes('t10k-labels-idx1-ubyte.gz');

    if (hasTrainImg && hasTrainLbl) {
      pendingFiles = files;
      const count = (hasTestImg && hasTestLbl) ? 4 : 2;
      fileStatus.textContent = `${count} file(s) ready`;
      fileStatus.className = 'file-status ready';
    } else {
      pendingFiles = null;
      const missing = [];
      if (!hasTrainImg) missing.push('train-images');
      if (!hasTrainLbl) missing.push('train-labels');
      fileStatus.textContent = `Missing: ${missing.join(', ')}`;
      fileStatus.className = 'file-status error';
    }
  });
}

function resetBars() {
  for (let i = 0; i < 10; i++) {
    document.getElementById(`bar-fill-${i}`).style.width = '0%';
    document.getElementById(`bar-fill-${i}`).className = 'output-bar-fill';
    document.getElementById(`bar-pct-${i}`).textContent = '0%';
  }
}

function setStatus(state, text) {
  statusDot.className = 'status-dot ' + state;
  statusText.textContent = text;
}

function updateButtons() {
  trainBtn.disabled = isTraining;
  stopBtn.disabled = !isTraining;
  saveBtn.disabled = !modelTrained || isTraining;
  loadFileBtn.disabled = isTraining;
  loadUrlBtn.disabled = isTraining;
}

async function ensureMNIST() {
  if (isLoaded()) return true;
  setStatus('training', 'Trying to download MNIST...');
  const autoOk = await tryAutoLoad((msg) => setStatus('training', msg));
  if (autoOk) { mnistPanel.classList.add('hidden'); return true; }

  if (pendingFiles) {
    try {
      await loadFromFiles(pendingFiles, (msg) => setStatus('training', msg));
      mnistPanel.classList.add('hidden');
      fileStatus.textContent = 'Loaded!';
      fileStatus.className = 'file-status ready';
      return true;
    } catch (e) {
      fileStatus.textContent = 'Error: ' + e.message;
      fileStatus.className = 'file-status error';
    }
  }

  mnistPanel.classList.remove('hidden');
  setStatus('untrained', 'Download MNIST files and select them above');
  return false;
}

async function startTraining() {
  if (isTraining) return;
  isTraining = true;
  stopRequested = false;
  progressArea.classList.remove('hidden');
  updateButtons();

  const dataOk = await ensureMNIST();
  if (!dataOk) { isTraining = false; updateButtons(); return; }

  const epochs = parseInt(document.getElementById('epochs-input').value) || 10;
  const learningRate = parseFloat(document.getElementById('lr-input').value) || 0.01;
  const batchSize = parseInt(document.getElementById('batch-input').value) || 32;

  setStatus('training', 'Training on 60,000 MNIST samples...');
  network = new NeuralNetwork([784, 128, 64, 10]);

  const batchesPerEpoch = Math.floor(60000 / batchSize);
  let bestTestAcc = 0;
  let lr = learningRate;

  for (let epoch = 0; epoch < epochs; epoch++) {
    if (stopRequested) break;
    let epochLoss = 0, epochCorrect = 0, epochTotal = 0;

    if (epoch >= Math.floor(epochs * 0.5)) lr = learningRate * 0.5;
    if (epoch >= Math.floor(epochs * 0.75)) lr = learningRate * 0.1;

    for (let b = 0; b < batchesPerEpoch; b++) {
      if (stopRequested) break;
      const batch = getTrainingBatch(batchSize);
      const result = network.trainBatch(batch, lr, 0.9);
      epochLoss += result.loss;
      epochCorrect += result.accuracy * batchSize;
      epochTotal += batchSize;

      if (b % 50 === 0) {
        const pct = ((epoch * batchesPerEpoch + b) / (epochs * batchesPerEpoch)) * 100;
        progressFill.style.width = pct + '%';
        progressEpoch.textContent = `Epoch ${epoch + 1}/${epochs}`;
        progressAccuracy.textContent = `Train: ${(epochCorrect / epochTotal * 100).toFixed(1)}%`;
        progressLoss.textContent = `Loss: ${(epochLoss / (b + 1)).toFixed(3)}`;
        await sleep(0);
      }
    }

    const testAcc = evaluateTestSet(network, 2000);
    if (testAcc !== null && testAcc > bestTestAcc) bestTestAcc = testAcc;
    progressAccuracy.textContent = `Test: ${testAcc ? (testAcc * 100).toFixed(1) : 'N/A'}%`;
    await sleep(0);
  }

  progressFill.style.width = '100%';
  isTraining = false;
  modelTrained = true;
  stopRequested = false;

  const finalTestAcc = evaluateTestSet(network, 10000);
  const accStr = finalTestAcc !== null ? (finalTestAcc * 100).toFixed(1) : (bestTestAcc * 100).toFixed(1);
  setStatus('trained', `Trained (${accStr}% test accuracy)`);
  progressAccuracy.textContent = `Test: ${accStr}%`;
  updateButtons();
  visualizer.draw(network);
}

function saveModelToFile() {
  if (!network || !modelTrained) return;
  const json = network.serialize();
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `mnist-model-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  modelInfo.textContent = `Saved as ${a.download}`;
}

async function loadModelFromFile(file) {
  try {
    const text = await file.text();
    network = NeuralNetwork.deserialize(text);
    modelTrained = true;
    setStatus('trained', 'Model loaded from file');
    modelInfo.textContent = `Loaded: ${file.name}`;
    updateButtons();
    visualizer.draw(network);
  } catch (e) {
    modelInfo.textContent = 'Load failed: ' + e.message;
  }
}

const PRETRAINED_MODEL_URL = 'https://raw.githubusercontent.com/Temperdox/neural_network/refs/heads/master/mnist-model-1776705741158.json';

let urlModalSetup = false;
function setupUrlModal() {
  if (urlModalSetup) return;
  urlModalSetup = true;

  const modal = document.getElementById('url-modal');
  const closeBtn = document.getElementById('url-modal-close');
  const cancelBtn = document.getElementById('url-modal-cancel');
  const loadBtn = document.getElementById('url-modal-load');
  const pretrainedBtn = document.getElementById('load-pretrained-btn');
  const input = document.getElementById('modal-url-input');
  const errorEl = document.getElementById('modal-error');
  if (!modal) return;

  function close() {
    modal.classList.add('hidden');
    errorEl.classList.add('hidden');
    errorEl.textContent = '';
    input.value = '';
  }
  function showError(msg) {
    errorEl.textContent = msg;
    errorEl.classList.remove('hidden');
  }

  closeBtn.addEventListener('click', close);
  cancelBtn.addEventListener('click', close);
  modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) close();
  });

  pretrainedBtn.addEventListener('click', async () => {
    const ok = await loadModelFromUrlString(PRETRAINED_MODEL_URL, 'pretrained model', showError);
    if (ok) close();
  });

  loadBtn.addEventListener('click', async () => {
    const url = input.value.trim();
    if (!url) { showError('Please enter a URL.'); return; }
    const ok = await loadModelFromUrlString(url, 'model', showError);
    if (ok) close();
  });

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') loadBtn.click();
  });
}

function openUrlModal() {
  setupUrlModal();
  const modal = document.getElementById('url-modal');
  if (!modal) return;
  modal.classList.remove('hidden');
  setTimeout(() => {
    const input = document.getElementById('modal-url-input');
    if (input) input.focus();
  }, 50);
}

async function loadModelFromUrlString(url, label, onError) {
  try {
    modelInfo.textContent = 'Downloading...';
    setStatus('training', `Downloading ${label}...`);
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const text = await resp.text();
    network = NeuralNetwork.deserialize(text);
    modelTrained = true;
    setStatus('trained', `Loaded ${label}`);
    modelInfo.textContent = `Loaded: ${label}`;
    updateButtons();
    visualizer.draw(network);
    return true;
  } catch (e) {
    setStatus('untrained', 'Load failed');
    modelInfo.textContent = 'Failed: ' + e.message;
    if (onError) onError('Load failed: ' + e.message);
    return false;
  }
}

function predict() {
  if (isTraining || !network || !modelTrained) return;
  const imageData = drawCanvas.getImageData();
  const output = network.forward(imageData);

  let maxIdx = 0, maxVal = output[0];
  for (let i = 1; i < output.length; i++) {
    if (output[i] > maxVal) { maxVal = output[i]; maxIdx = i; }
  }

  document.getElementById('prediction-value').textContent = maxIdx;
  document.getElementById('confidence-value').textContent = (maxVal * 100).toFixed(1) + '%';

  for (let i = 0; i < 10; i++) {
    const pct = output[i] * 100;
    const fill = document.getElementById(`bar-fill-${i}`);
    fill.style.width = pct + '%';
    fill.className = 'output-bar-fill' + (i === maxIdx ? ' top' : '');
    document.getElementById(`bar-pct-${i}`).textContent = pct.toFixed(1) + '%';
  }
  visualizer.draw(network);
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================================
// Init
// ============================================================
function initRangeSliderFills() {
  const sliders = document.querySelectorAll('input[type="range"]');
  sliders.forEach(slider => {
    const update = () => {
      const min = parseFloat(slider.min) || 0;
      const max = parseFloat(slider.max) || 100;
      const val = parseFloat(slider.value);
      const pct = ((val - min) / (max - min)) * 100;
      slider.style.setProperty('--range-pct', pct + '%');
    };
    slider.addEventListener('input', update);
    update();
  });
}

function setupHowtoModal() {
  const modal = document.getElementById('howto-modal');
  const openBtn = document.getElementById('howto-btn');
  const closeBtn = document.getElementById('howto-modal-close');
  const okBtn = document.getElementById('howto-modal-ok');
  if (!modal || !openBtn) return;

  function open() { modal.classList.remove('hidden'); }
  function close() { modal.classList.add('hidden'); }

  openBtn.addEventListener('click', open);
  closeBtn.addEventListener('click', close);
  okBtn.addEventListener('click', close);
  modal.addEventListener('click', (e) => { if (e.target === modal) close(); });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) close();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initSlides();
  drawStaticDiagram();
  initConceptDemos();
  initRangeSliderFills();
  setupHowtoModal();
});
