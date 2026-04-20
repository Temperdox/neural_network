/**
 * Feedforward neural network with momentum SGD, implemented from scratch.
 * Architecture: 784 -> 128 -> 64 -> 10
 * Activation: ReLU (hidden), Softmax (output)
 * Training: Mini-batch SGD with momentum
 */

export class NeuralNetwork {
  constructor(layerSizes) {
    this.layerSizes = layerSizes;
    this.weights = [];
    this.biases = [];
    this.activations = [];
    this.preActivations = [];

    // Momentum velocities
    this.vWeights = [];
    this.vBiases = [];

    // He initialization (better for ReLU)
    for (let i = 0; i < layerSizes.length - 1; i++) {
      const fanIn = layerSizes[i];
      const fanOut = layerSizes[i + 1];
      const scale = Math.sqrt(2.0 / fanIn);

      const w = [];
      const vw = [];
      for (let r = 0; r < fanOut; r++) {
        const row = new Float64Array(fanIn);
        const vrow = new Float64Array(fanIn);
        for (let c = 0; c < fanIn; c++) {
          row[c] = gaussianRandom() * scale;
        }
        w.push(row);
        vw.push(vrow);
      }
      this.weights.push(w);
      this.vWeights.push(vw);

      this.biases.push(new Float64Array(fanOut));
      this.vBiases.push(new Float64Array(fanOut));
    }
  }

  forward(input) {
    this.activations = [Array.isArray(input) ? input.slice() : Array.from(input)];
    this.preActivations = [];

    let current = this.activations[0];

    for (let layer = 0; layer < this.weights.length; layer++) {
      const w = this.weights[layer];
      const b = this.biases[layer];
      const outSize = w.length;
      const next = new Array(outSize);

      for (let j = 0; j < outSize; j++) {
        let sum = b[j];
        const wRow = w[j];
        for (let k = 0; k < current.length; k++) {
          sum += wRow[k] * current[k];
        }
        next[j] = sum;
      }

      this.preActivations.push(next.slice());

      const isLastLayer = layer === this.weights.length - 1;
      if (isLastLayer) {
        current = softmax(next);
      } else {
        current = next.map(relu);
      }

      this.activations.push(current.slice());
    }

    return current;
  }

  /**
   * Train on a mini-batch with momentum SGD.
   * Returns average loss over the batch.
   */
  trainBatch(batch, learningRate, momentum = 0.9) {
    const numLayers = this.weights.length;
    const batchSize = batch.length;

    // Accumulate gradients
    const gradW = [];
    const gradB = [];
    for (let l = 0; l < numLayers; l++) {
      const gw = [];
      for (let r = 0; r < this.weights[l].length; r++) {
        gw.push(new Float64Array(this.weights[l][r].length));
      }
      gradW.push(gw);
      gradB.push(new Float64Array(this.biases[l].length));
    }

    let totalLoss = 0;
    let correct = 0;

    for (const sample of batch) {
      const output = this.forward(sample.input);

      // Loss
      totalLoss -= Math.log(output[sample.label] + 1e-10);

      // Accuracy
      let maxIdx = 0;
      for (let i = 1; i < output.length; i++) {
        if (output[i] > output[maxIdx]) maxIdx = i;
      }
      if (maxIdx === sample.label) correct++;

      // Backprop
      const deltas = new Array(numLayers);

      // Output delta: softmax + cross-entropy => output - oneHot
      const outputDelta = output.slice();
      outputDelta[sample.label] -= 1;
      deltas[numLayers - 1] = outputDelta;

      // Hidden deltas
      for (let l = numLayers - 2; l >= 0; l--) {
        const size = this.layerSizes[l + 1];
        const delta = new Array(size);
        for (let j = 0; j < size; j++) {
          let sum = 0;
          for (let k = 0; k < this.layerSizes[l + 2]; k++) {
            sum += this.weights[l + 1][k][j] * deltas[l + 1][k];
          }
          delta[j] = this.preActivations[l][j] > 0 ? sum : 0;
        }
        deltas[l] = delta;
      }

      // Accumulate gradients
      for (let l = 0; l < numLayers; l++) {
        for (let j = 0; j < this.weights[l].length; j++) {
          const dj = deltas[l][j];
          const act = this.activations[l];
          const gwRow = gradW[l][j];
          for (let k = 0; k < gwRow.length; k++) {
            gwRow[k] += dj * act[k];
          }
          gradB[l][j] += dj;
        }
      }
    }

    // Apply gradients with momentum
    const scale = 1 / batchSize;
    for (let l = 0; l < numLayers; l++) {
      for (let j = 0; j < this.weights[l].length; j++) {
        const wRow = this.weights[l][j];
        const vwRow = this.vWeights[l][j];
        const gwRow = gradW[l][j];
        for (let k = 0; k < wRow.length; k++) {
          vwRow[k] = momentum * vwRow[k] - learningRate * gwRow[k] * scale;
          wRow[k] += vwRow[k];
        }
      }
      for (let j = 0; j < this.biases[l].length; j++) {
        this.vBiases[l][j] = momentum * this.vBiases[l][j] - learningRate * gradB[l][j] * scale;
        this.biases[l][j] += this.vBiases[l][j];
      }
    }

    return {
      loss: totalLoss / batchSize,
      accuracy: correct / batchSize
    };
  }

  serialize() {
    return JSON.stringify({
      layerSizes: this.layerSizes,
      weights: this.weights.map(layer => layer.map(row => Array.from(row))),
      biases: this.biases.map(b => Array.from(b))
    });
  }

  static deserialize(json) {
    const data = JSON.parse(json);
    const nn = new NeuralNetwork(data.layerSizes);
    nn.weights = data.weights.map(layer =>
      layer.map(row => new Float64Array(row))
    );
    nn.biases = data.biases.map(b => new Float64Array(b));
    // Reset velocity (no momentum state saved)
    nn.vWeights = nn.weights.map(layer =>
      layer.map(row => new Float64Array(row.length))
    );
    nn.vBiases = nn.biases.map(b => new Float64Array(b.length));
    return nn;
  }
}

function relu(x) {
  return x > 0 ? x : 0;
}

function softmax(arr) {
  let max = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) max = arr[i];
  }
  const exps = new Array(arr.length);
  let sum = 0;
  for (let i = 0; i < arr.length; i++) {
    exps[i] = Math.exp(arr[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < arr.length; i++) {
    exps[i] /= sum;
  }
  return exps;
}

function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
