import { NeuralNetwork } from './neural-network.js';
import { NetworkVisualizer } from './network-visualizer.js';
import { DrawingCanvas } from './drawing-canvas.js';
import {
  tryAutoLoad, loadFromFiles, getTrainingBatch, evaluateTestSet,
  isLoaded, DOWNLOAD_URLS
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
// Live Demo (Slide 7)
// ============================================================
let network = null;
let drawCanvas = null;
let visualizer = null;
let isTraining = false;
let stopRequested = false;
let modelTrained = false;
let pendingFiles = null;
let demoInitialized = false;

let trainBtn, stopBtn, saveBtn, loadFileBtn, loadUrlBtn, predictBtn;
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
  predictBtn = document.getElementById('predict-btn');
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
  loadUrlBtn.addEventListener('click', loadModelFromUrl);

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

  predictBtn.addEventListener('click', predict);

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
  predictBtn.disabled = !modelTrained || isTraining;
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

async function loadModelFromUrl() {
  const url = prompt('Enter URL to model .json file:\n(e.g. GitHub raw link)');
  if (!url || !url.trim()) return;
  try {
    modelInfo.textContent = 'Downloading...';
    setStatus('training', 'Downloading model...');
    const resp = await fetch(url.trim());
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const text = await resp.text();
    network = NeuralNetwork.deserialize(text);
    modelTrained = true;
    setStatus('trained', 'Model loaded from URL');
    modelInfo.textContent = 'Loaded from URL';
    updateButtons();
    visualizer.draw(network);
  } catch (e) {
    setStatus('untrained', 'Load failed');
    modelInfo.textContent = 'Failed: ' + e.message;
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
document.addEventListener('DOMContentLoaded', () => {
  initSlides();
  drawStaticDiagram();
});
