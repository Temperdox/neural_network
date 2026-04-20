/**
 * Handles the drawing canvas for digit input.
 * Supports mouse and touch events.
 * Exports the drawn image as a 28x28 grayscale array matching MNIST preprocessing:
 * - Digit is fit into a 20x20 box
 * - Centered in 28x28 using center of mass
 */

export class DrawingCanvas {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.drawing = false;
    this.lastPoint = null;

    this.ctx.lineWidth = 20;
    this.ctx.lineCap = 'round';
    this.ctx.lineJoin = 'round';
    this.ctx.strokeStyle = '#ffffff';

    this.clear();
    this.setupEvents();
  }

  clear() {
    this.ctx.fillStyle = '#000000';
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.hasDrawing = false;
  }

  setupEvents() {
    this.canvas.addEventListener('mousedown', (e) => this.startDraw(e));
    this.canvas.addEventListener('mousemove', (e) => this.continueDraw(e));
    this.canvas.addEventListener('mouseup', () => this.endDraw());
    this.canvas.addEventListener('mouseleave', () => this.endDraw());

    this.canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      this.startDraw(e.touches[0]);
    });
    this.canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      this.continueDraw(e.touches[0]);
    });
    this.canvas.addEventListener('touchend', (e) => {
      e.preventDefault();
      this.endDraw();
    });
  }

  getPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  startDraw(e) {
    this.drawing = true;
    this.hasDrawing = true;
    this.lastPoint = this.getPos(e);

    this.ctx.beginPath();
    this.ctx.arc(this.lastPoint.x, this.lastPoint.y, this.ctx.lineWidth / 2, 0, Math.PI * 2);
    this.ctx.fillStyle = '#ffffff';
    this.ctx.fill();
  }

  continueDraw(e) {
    if (!this.drawing) return;
    const pos = this.getPos(e);

    this.ctx.beginPath();
    this.ctx.moveTo(this.lastPoint.x, this.lastPoint.y);
    this.ctx.lineTo(pos.x, pos.y);
    this.ctx.stroke();

    this.lastPoint = pos;
  }

  endDraw() {
    this.drawing = false;
    this.lastPoint = null;
  }

  /**
   * Get the canvas content as a 28x28 grayscale array (values 0-1).
   * Uses the same preprocessing as MNIST:
   * 1. Find bounding box of content
   * 2. Scale to fit in 20x20
   * 3. Place in 28x28 centered by center of mass
   */
  getImageData() {
    const imageData = this.ctx.getImageData(0, 0, this.canvas.width, this.canvas.height);
    const pixels = imageData.data;
    const w = this.canvas.width;
    const h = this.canvas.height;

    // First, downsample to an intermediate resolution using area averaging
    // to avoid aliasing. We'll use a temporary canvas.
    const tmpSize = 100;
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = tmpSize;
    tmpCanvas.height = tmpSize;
    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(this.canvas, 0, 0, tmpSize, tmpSize);
    const tmpData = tmpCtx.getImageData(0, 0, tmpSize, tmpSize).data;

    // Convert to grayscale float array
    const gray = new Float32Array(tmpSize * tmpSize);
    for (let i = 0; i < tmpSize * tmpSize; i++) {
      gray[i] = tmpData[i * 4] / 255; // R channel
    }

    // Find bounding box
    let minX = tmpSize, maxX = 0, minY = tmpSize, maxY = 0;
    let hasContent = false;

    for (let y = 0; y < tmpSize; y++) {
      for (let x = 0; x < tmpSize; x++) {
        if (gray[y * tmpSize + x] > 0.05) {
          hasContent = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (!hasContent) {
      return new Array(28 * 28).fill(0);
    }

    // Crop to bounding box
    const cropW = maxX - minX + 1;
    const cropH = maxY - minY + 1;

    // Scale to fit in 20x20 while preserving aspect ratio
    const targetSize = 20;
    let scaleW, scaleH;
    if (cropW > cropH) {
      scaleW = targetSize;
      scaleH = Math.max(1, Math.round(cropH * targetSize / cropW));
    } else {
      scaleH = targetSize;
      scaleW = Math.max(1, Math.round(cropW * targetSize / cropH));
    }

    // Resample cropped region to scaled size
    const scaled = new Float32Array(scaleW * scaleH);
    for (let y = 0; y < scaleH; y++) {
      for (let x = 0; x < scaleW; x++) {
        // Map to source coordinates (area sampling)
        const srcX0 = minX + (x / scaleW) * cropW;
        const srcX1 = minX + ((x + 1) / scaleW) * cropW;
        const srcY0 = minY + (y / scaleH) * cropH;
        const srcY1 = minY + ((y + 1) / scaleH) * cropH;

        let sum = 0;
        let count = 0;
        for (let sy = Math.floor(srcY0); sy < Math.ceil(srcY1); sy++) {
          for (let sx = Math.floor(srcX0); sx < Math.ceil(srcX1); sx++) {
            if (sx >= 0 && sx < tmpSize && sy >= 0 && sy < tmpSize) {
              // Weight by overlap
              const overlapX = Math.min(sx + 1, srcX1) - Math.max(sx, srcX0);
              const overlapY = Math.min(sy + 1, srcY1) - Math.max(sy, srcY0);
              const weight = Math.max(0, overlapX) * Math.max(0, overlapY);
              sum += gray[sy * tmpSize + sx] * weight;
              count += weight;
            }
          }
        }
        scaled[y * scaleW + x] = count > 0 ? sum / count : 0;
      }
    }

    // Compute center of mass of the scaled image
    let totalMass = 0, comX = 0, comY = 0;
    for (let y = 0; y < scaleH; y++) {
      for (let x = 0; x < scaleW; x++) {
        const v = scaled[y * scaleW + x];
        totalMass += v;
        comX += x * v;
        comY += y * v;
      }
    }

    if (totalMass > 0) {
      comX /= totalMass;
      comY /= totalMass;
    } else {
      comX = scaleW / 2;
      comY = scaleH / 2;
    }

    // Place in 28x28 canvas, centered so that center of mass is at (14, 14)
    const output = new Array(28 * 28).fill(0);
    const offsetX = Math.round(14 - comX);
    const offsetY = Math.round(14 - comY);

    for (let y = 0; y < scaleH; y++) {
      for (let x = 0; x < scaleW; x++) {
        const tx = x + offsetX;
        const ty = y + offsetY;
        if (tx >= 0 && tx < 28 && ty >= 0 && ty < 28) {
          output[ty * 28 + tx] = scaled[y * scaleW + x];
        }
      }
    }

    return output;
  }
}
