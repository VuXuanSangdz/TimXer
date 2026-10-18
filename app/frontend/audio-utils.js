/* Tiện ích âm thanh phía trình duyệt:
 *  - decodeToWav(blob): giải mã bất kỳ định dạng trình duyệt hỗ trợ (webm/opus, mp3, m4a, ogg, flac, wav)
 *    -> WAV PCM16 mono 16 kHz. Nhờ vậy backend chỉ cần đọc WAV (không cần ffmpeg).
 *  - drawWaveform(canvas, audioBuffer)
 */
(function (global) {
  const TARGET_SR = 16000;

  async function decodeToWav(blob) {
    const arrayBuf = await blob.arrayBuffer();
    const AC = global.AudioContext || global.webkitAudioContext;
    const ctx = new AC();
    let decoded;
    try {
      decoded = await ctx.decodeAudioData(arrayBuf.slice(0));
    } finally {
      ctx.close && ctx.close();
    }
    // mono
    const n = decoded.length;
    const mono = new Float32Array(n);
    for (let c = 0; c < decoded.numberOfChannels; c++) {
      const d = decoded.getChannelData(c);
      for (let i = 0; i < n; i++) mono[i] += d[i] / decoded.numberOfChannels;
    }
    // resample bằng OfflineAudioContext
    let pcm = mono, sr = decoded.sampleRate;
    if (sr !== TARGET_SR) {
      const frames = Math.ceil(n * TARGET_SR / sr);
      const off = new OfflineAudioContext(1, frames, TARGET_SR);
      const buf = off.createBuffer(1, n, sr);
      buf.copyToChannel(mono, 0);
      const src = off.createBufferSource();
      src.buffer = buf; src.connect(off.destination); src.start();
      const rendered = await off.startRendering();
      pcm = rendered.getChannelData(0); sr = TARGET_SR;
    }
    return { wav: encodeWav(pcm, sr), duration: pcm.length / sr, pcm, sampleRate: sr };
  }

  function encodeWav(samples, sr) {
    const buf = new ArrayBuffer(44 + samples.length * 2);
    const v = new DataView(buf);
    const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
    w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE');
    w(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    w(36, 'data'); v.setUint32(40, samples.length * 2, true);
    let o = 44;
    for (let i = 0; i < samples.length; i++, o += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buf], { type: 'audio/wav' });
  }

  function drawWaveform(canvas, pcm, progress) {
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.clientWidth * devicePixelRatio;
    const H = canvas.height = 48 * devicePixelRatio;
    ctx.clearRect(0, 0, W, H);
    const bars = Math.floor(W / (4 * devicePixelRatio));
    const step = Math.max(1, Math.floor((pcm ? pcm.length : 0) / bars));
    for (let b = 0; b < bars; b++) {
      let amp = 0.08;
      if (pcm) {
        let m = 0;
        for (let i = b * step; i < (b + 1) * step && i < pcm.length; i++) m = Math.max(m, Math.abs(pcm[i]));
        amp = Math.max(0.08, Math.min(1, m * 1.6));
      }
      const h = amp * H * 0.9;
      ctx.fillStyle = (progress != null && b / bars <= progress) ? '#1e3a8a' : '#3b6cf5';
      ctx.fillRect(b * 4 * devicePixelRatio, (H - h) / 2, 2.4 * devicePixelRatio, h);
    }
  }

  global.AudioUtils = { decodeToWav, encodeWav, drawWaveform };
})(window);
