class PCMProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (input.length > 0) {
      const float32Data = input[0];
      const int16Data = new Int16Array(float32Data.length);
      for (let i = 0; i < float32Data.length; i++) {
        let s = Math.max(-1, Math.min(1, float32Data[i]));
        int16Data[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
      }
      this.port.postMessage(int16Data.buffer, [int16Data.buffer]);
    }

    // Silence output to avoid feedback loop if connected to speakers
    // We connect the node to destination to keep it alive, but we don't want to hear the mic input directly.
    if (output.length > 0) {
        for (let channel = 0; channel < output.length; channel++) {
            output[channel].fill(0);
        }
    }

    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
