/**
 * MNIST dataset loader.
 *
 * Strategy:
 * 1. Try fetching from multiple mirrors (some have CORS, some don't)
 * 2. If all fail, signal the caller so it can show a manual file-upload UI
 * 3. Accept raw .gz files dropped/selected by the user and parse them in-browser
 *
 * IDX file format:
 *   Images: [magic=2051][count][rows][cols][pixel bytes...]
 *   Labels: [magic=2049][count][label bytes...]
 */

// Multiple mirrors to try in order
const MIRRORS = [
  // GitHub-hosted raw mirrors tend to have CORS
  {
    trainImages: 'https://raw.githubusercontent.com/lorenmh/mnist_handwritten_json/master/mnist_handwritten_train.json',
    type: 'json'
  },
  // Direct S3 (often blocked by CORS from localhost)
  {
    trainImages: 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
    trainLabels: 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz',
    testImages: 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz',
    testLabels: 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz',
    type: 'idx'
  },
];

// The 4 files the user needs to provide if auto-download fails
export const REQUIRED_FILES = {
  trainImages: 'train-images-idx3-ubyte.gz',
  trainLabels: 'train-labels-idx1-ubyte.gz',
  testImages: 't10k-images-idx3-ubyte.gz',
  testLabels: 't10k-labels-idx1-ubyte.gz',
};

export const DOWNLOAD_URLS = [
  { name: 'train-images-idx3-ubyte.gz', url: 'https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', desc: 'Training images (9.9 MB)' },
  { name: 'train-labels-idx1-ubyte.gz', url: 'https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', desc: 'Training labels (29 KB)' },
  { name: 't10k-images-idx3-ubyte.gz', url: 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', desc: 'Test images (1.6 MB)' },
  { name: 't10k-labels-idx1-ubyte.gz', url: 'https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', desc: 'Test labels (4.4 KB)' },
];

let trainImages = null;
let trainLabels = null;
let testImages = null;
let testLabels = null;

// --- IDX binary parsers ---

function parseImages(buffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0);
  if (magic !== 2051) throw new Error('Invalid image file magic: ' + magic);

  const count = view.getUint32(4);
  const rows = view.getUint32(8);
  const cols = view.getUint32(12);
  const imageSize = rows * cols;
  const images = new Array(count);

  for (let i = 0; i < count; i++) {
    const offset = 16 + i * imageSize;
    const img = new Float32Array(imageSize);
    for (let j = 0; j < imageSize; j++) {
      img[j] = view.getUint8(offset + j) / 255;
    }
    images[i] = img;
  }
  return images;
}

function parseLabels(buffer) {
  const view = new DataView(buffer);
  const magic = view.getUint32(0);
  if (magic !== 2049) throw new Error('Invalid label file magic: ' + magic);

  const count = view.getUint32(4);
  const labels = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    labels[i] = view.getUint8(8 + i);
  }
  return labels;
}

// --- Decompression ---

async function decompressGzip(data) {
  // data can be ArrayBuffer or Uint8Array
  const blob = new Blob([data]);
  const ds = new DecompressionStream('gzip');
  const stream = blob.stream().pipeThrough(ds);
  const reader = stream.getReader();

  const chunks = [];
  let totalLength = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    totalLength += value.length;
  }

  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }
  return result.buffer;
}

// --- Fetch with timeout ---

async function fetchWithTimeout(url, timeoutMs = 10000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const resp = await fetch(url, { signal: controller.signal });
    clearTimeout(timer);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp;
  } catch (e) {
    clearTimeout(timer);
    throw e;
  }
}

// --- Auto-download attempts ---

async function tryFetchIdx(onProgress) {
  const mirror = MIRRORS[1]; // idx mirror
  onProgress('Downloading MNIST training images...');
  const trainImgResp = await fetchWithTimeout(mirror.trainImages, 30000);
  const trainImgBuf = await decompressGzip(await trainImgResp.arrayBuffer());

  onProgress('Downloading MNIST training labels...');
  const trainLblResp = await fetchWithTimeout(mirror.trainLabels, 15000);
  const trainLblBuf = await decompressGzip(await trainLblResp.arrayBuffer());

  onProgress('Downloading MNIST test images...');
  const testImgResp = await fetchWithTimeout(mirror.testImages, 20000);
  const testImgBuf = await decompressGzip(await testImgResp.arrayBuffer());

  onProgress('Downloading MNIST test labels...');
  const testLblResp = await fetchWithTimeout(mirror.testLabels, 10000);
  const testLblBuf = await decompressGzip(await testLblResp.arrayBuffer());

  onProgress('Parsing MNIST data...');
  trainImages = parseImages(trainImgBuf);
  trainLabels = parseLabels(trainLblBuf);
  testImages = parseImages(testImgBuf);
  testLabels = parseLabels(testLblBuf);
}

/**
 * Try to auto-download MNIST. Returns true if successful, false if CORS blocked.
 */
export async function tryAutoLoad(onProgress) {
  if (trainImages) return true;

  try {
    await tryFetchIdx(onProgress);
    console.log(`MNIST loaded: ${trainImages.length} train, ${testImages.length} test`);
    return true;
  } catch (e) {
    console.warn('Auto-download failed:', e.message);
    return false;
  }
}

/**
 * Load MNIST from user-provided files (File objects from <input type="file">).
 * Expects an array of File objects. Matches by filename.
 */
export async function loadFromFiles(files, onProgress) {
  const fileMap = {};
  for (const f of files) {
    fileMap[f.name] = f;
  }

  // Match files
  const tiFile = fileMap[REQUIRED_FILES.trainImages];
  const tlFile = fileMap[REQUIRED_FILES.trainLabels];
  const eiFile = fileMap[REQUIRED_FILES.testImages];
  const elFile = fileMap[REQUIRED_FILES.testLabels];

  if (!tiFile || !tlFile) {
    throw new Error('Missing required files. Need at least: ' +
      REQUIRED_FILES.trainImages + ' and ' + REQUIRED_FILES.trainLabels);
  }

  onProgress('Reading training images...');
  const trainImgBuf = await decompressGzip(await tiFile.arrayBuffer());
  trainImages = parseImages(trainImgBuf);

  onProgress('Reading training labels...');
  const trainLblBuf = await decompressGzip(await tlFile.arrayBuffer());
  trainLabels = parseLabels(trainLblBuf);

  if (eiFile && elFile) {
    onProgress('Reading test images...');
    const testImgBuf = await decompressGzip(await eiFile.arrayBuffer());
    testImages = parseImages(testImgBuf);

    onProgress('Reading test labels...');
    const testLblBuf = await decompressGzip(await elFile.arrayBuffer());
    testLabels = parseLabels(testLblBuf);
  }

  console.log(`MNIST loaded from files: ${trainImages.length} train` +
    (testImages ? `, ${testImages.length} test` : ''));
}

// --- Training/evaluation API ---

export function getTrainingBatch(batchSize) {
  if (!trainImages) throw new Error('MNIST not loaded yet');
  const batch = [];
  for (let i = 0; i < batchSize; i++) {
    const idx = Math.floor(Math.random() * trainImages.length);
    batch.push({
      input: Array.from(trainImages[idx]),
      label: trainLabels[idx]
    });
  }
  return batch;
}

export function evaluateTestSet(network, maxSamples = 1000) {
  if (!testImages) return null;
  const count = Math.min(maxSamples, testImages.length);
  let correct = 0;
  for (let i = 0; i < count; i++) {
    const output = network.forward(Array.from(testImages[i]));
    let maxIdx = 0;
    for (let j = 1; j < output.length; j++) {
      if (output[j] > output[maxIdx]) maxIdx = j;
    }
    if (maxIdx === testLabels[i]) correct++;
  }
  return correct / count;
}

export function isLoaded() {
  return trainImages !== null;
}
