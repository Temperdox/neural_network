/**
 * Renders the neural network visualization on a canvas.
 * Shows nodes (colored by activation), connections (colored by weight),
 * and labels for each layer.
 */

export class NetworkVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.showWeights = true;
    this.showValues = true;

    // Colors
    this.colors = {
      bg: '#0a0a1a',
      node: '#2d2d5e',
      nodeStroke: '#4a4a8a',
      text: '#e0e0ff',
      textDim: '#8888bb',
      positive: '#00cec9',
      negative: '#ff6b6b',
      activation: '#ffeaa7',
    };

    // Layout config - how many nodes to display per layer (for large layers we sample)
    this.maxDisplayNodes = [16, 16, 12, 10]; // input(sampled), hidden1, hidden2, output
    this.layerLabels = ['Input\n(784 pixels)', 'Hidden 1\n(128 neurons)', 'Hidden 2\n(64 neurons)', 'Output\n(10 digits)'];

    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    const wrapper = this.canvas.parentElement;
    const rect = wrapper.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = rect.width;
    const height = Math.max(480, rect.height);

    this.canvas.width = width * dpr;
    this.canvas.height = height * dpr;
    this.canvas.style.width = width + 'px';
    this.canvas.style.height = height + 'px';
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = width;
    this.height = height;
  }

  /**
   * Draw the network given its current state.
   * @param {NeuralNetwork} network - the network to visualize
   */
  draw(network) {
    const ctx = this.ctx;
    const W = this.width;
    const H = this.height;

    // Clear
    ctx.fillStyle = this.colors.bg;
    ctx.fillRect(0, 0, W, H);

    if (!network || !network.activations || network.activations.length === 0) {
      ctx.fillStyle = this.colors.textDim;
      ctx.font = '14px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Draw a digit and click Predict to see the network activate', W / 2, H / 2);
      return;
    }

    const numLayers = network.activations.length;
    const layerX = [];
    const padding = 80;
    const labelSpace = 40;
    const usableWidth = W - padding * 2;

    for (let i = 0; i < numLayers; i++) {
      layerX.push(padding + (usableWidth / (numLayers - 1)) * i);
    }

    // Compute display nodes for each layer (sample from large layers)
    const displayIndices = [];
    const nodePositions = []; // [layer][displayIdx] => {x, y, realIdx}

    for (let l = 0; l < numLayers; l++) {
      const totalNodes = network.activations[l].length;
      const maxDisplay = this.maxDisplayNodes[l] || 12;
      const indices = [];

      if (totalNodes <= maxDisplay) {
        for (let i = 0; i < totalNodes; i++) indices.push(i);
      } else {
        // Evenly sample, always include first and last
        for (let i = 0; i < maxDisplay; i++) {
          indices.push(Math.floor((i / (maxDisplay - 1)) * (totalNodes - 1)));
        }
      }

      displayIndices.push(indices);

      const nodeCount = indices.length;
      const nodeRadius = Math.min(14, (H - labelSpace * 2 - 40) / (nodeCount * 2.8));
      const totalHeight = nodeCount * nodeRadius * 2.8;
      const startY = (H - labelSpace) / 2 - totalHeight / 2 + nodeRadius;

      const positions = [];
      for (let i = 0; i < nodeCount; i++) {
        positions.push({
          x: layerX[l],
          y: startY + i * nodeRadius * 2.8,
          r: nodeRadius,
          realIdx: indices[i],
        });
      }
      nodePositions.push(positions);
    }

    // Draw connections between layers
    if (this.showWeights) {
      for (let l = 0; l < numLayers - 1; l++) {
        const weightsMatrix = network.weights[l];
        const fromPositions = nodePositions[l];
        const toPositions = nodePositions[l + 1];

        // Find max abs weight for normalization
        let maxAbsW = 0;
        for (let t = 0; t < toPositions.length; t++) {
          for (let f = 0; f < fromPositions.length; f++) {
            const w = Math.abs(weightsMatrix[toPositions[t].realIdx][fromPositions[f].realIdx]);
            if (w > maxAbsW) maxAbsW = w;
          }
        }
        if (maxAbsW === 0) maxAbsW = 1;

        for (let t = 0; t < toPositions.length; t++) {
          for (let f = 0; f < fromPositions.length; f++) {
            const w = weightsMatrix[toPositions[t].realIdx][fromPositions[f].realIdx];
            const norm = w / maxAbsW;
            const absNorm = Math.abs(norm);

            // Only draw if weight is significant enough
            if (absNorm < 0.05) continue;

            const alpha = Math.pow(absNorm, 1.5) * 0.6;
            const color = norm > 0 ? this.colors.positive : this.colors.negative;

            ctx.beginPath();
            ctx.moveTo(fromPositions[f].x + fromPositions[f].r, fromPositions[f].y);
            ctx.lineTo(toPositions[t].x - toPositions[t].r, toPositions[t].y);
            ctx.strokeStyle = this.hexToRgba(color, alpha);
            ctx.lineWidth = absNorm * 2;
            ctx.stroke();
          }
        }
      }
    }

    // Draw nodes
    for (let l = 0; l < numLayers; l++) {
      const positions = nodePositions[l];
      const activations = network.activations[l];
      const isOutputLayer = l === numLayers - 1;

      // Find max activation for this layer (for normalization)
      let maxAct = 0;
      for (const act of activations) {
        if (Math.abs(act) > maxAct) maxAct = Math.abs(act);
      }
      if (maxAct === 0) maxAct = 1;

      for (const pos of positions) {
        const act = activations[pos.realIdx];
        const normAct = isOutputLayer ? act : Math.min(1, Math.abs(act) / maxAct);

        // Node circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, pos.r, 0, Math.PI * 2);

        // Fill with activation intensity
        const brightness = Math.floor(normAct * 255);
        if (isOutputLayer) {
          // Output nodes: yellow for high activation
          const r = Math.floor(255 * normAct);
          const g = Math.floor(234 * normAct);
          const b = Math.floor(167 * normAct);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
        } else {
          // Hidden/input: blue-white gradient
          ctx.fillStyle = `rgb(${Math.floor(45 + brightness * 0.6)},${Math.floor(45 + brightness * 0.6)},${Math.floor(94 + brightness * 0.63)})`;
        }
        ctx.fill();

        // Stroke
        ctx.strokeStyle = isOutputLayer && normAct > 0.5
          ? this.colors.activation
          : this.colors.nodeStroke;
        ctx.lineWidth = isOutputLayer && normAct > 0.5 ? 2 : 1;
        ctx.stroke();

        // Value label
        if (this.showValues && pos.r >= 8) {
          ctx.fillStyle = normAct > 0.6 ? '#000' : this.colors.text;
          ctx.font = `${Math.max(8, pos.r * 0.7)}px system-ui, sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';

          if (isOutputLayer) {
            // Show digit label and probability
            ctx.fillStyle = normAct > 0.5 ? '#000' : this.colors.text;
            ctx.fillText(pos.realIdx.toString(), pos.x, pos.y);
          } else if (l === 0) {
            // Input: show pixel intensity
            const pct = Math.round(act * 100);
            if (pct > 0) {
              ctx.font = `${Math.max(6, pos.r * 0.55)}px system-ui, sans-serif`;
              ctx.fillText(pct + '%', pos.x, pos.y);
            }
          } else {
            // Hidden: show activation value
            ctx.font = `${Math.max(6, pos.r * 0.55)}px system-ui, sans-serif`;
            ctx.fillText(act.toFixed(1), pos.x, pos.y);
          }
        }
      }

      // Ellipsis for sampled layers
      const totalNodes = network.activations[l].length;
      const maxDisplay = this.maxDisplayNodes[l] || 12;
      if (totalNodes > maxDisplay && positions.length >= 2) {
        const midIdx = Math.floor(positions.length / 2);
        const midY = (positions[midIdx - 1].y + positions[midIdx].y) / 2;
        ctx.fillStyle = this.colors.textDim;
        ctx.font = '12px system-ui, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('...', layerX[l], midY);
        ctx.font = '9px system-ui, sans-serif';
        ctx.fillText(`(${totalNodes} total)`, layerX[l], midY + 14);
      }

      // Layer labels
      ctx.fillStyle = this.colors.textDim;
      ctx.font = '11px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      const labelLines = this.layerLabels[l].split('\n');
      const labelY = H - labelSpace + 4;
      labelLines.forEach((line, li) => {
        ctx.fillStyle = li === 0 ? this.colors.text : this.colors.textDim;
        ctx.font = li === 0 ? 'bold 11px system-ui, sans-serif' : '10px system-ui, sans-serif';
        ctx.fillText(line, layerX[l], labelY + li * 14);
      });
    }
  }

  hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }
}
