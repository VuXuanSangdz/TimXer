/* TimeXer AI — logic giao diện dự đoán */
(() => {
  const $ = (s) => document.querySelector(s);
  const GROUPS = ['<18', '18-29', '30-49', '50-64', '65+'];
  const LABELS = { '<18': 'Dưới 18 tuổi', '18-29': '18 – 29 tuổi', '30-49': '30 – 49 tuổi', '50-64': '50 – 64 tuổi', '65+': 'Từ 65 tuổi trở lên' };
  const FACES = { '<18': '🧒', '18-29': '🧑', '30-49': '🧔', '50-64': '👩‍🦳', '65+': '🧓' };

  let current = null;      // { wav: Blob, duration, pcm, source: 'mic'|'upload', name }
  let audioEl = null;
  let mediaRecorder = null, chunks = [], recStart = 0, recTimer = null;

  // ---------- điều hướng ----------
  function showView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${name}`));
    document.querySelectorAll('.nav-item[data-view]').forEach(a => a.classList.toggle('active', a.dataset.view === name || (name === 'predict' && a.dataset.view === 'home')));
    if (name === 'predict') document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === 'view-home'));
    if (name === 'history') loadHistory(100, '#history-body-full');
    if (name === 'settings') loadModelInfo();
  }
  document.querySelectorAll('[data-view]').forEach(a => a.addEventListener('click', (e) => { e.preventDefault(); showView(a.dataset.view); location.hash = a.dataset.view; }));
  if (location.hash) showView(location.hash.slice(1) || 'home');

  // ---------- tab ----------
  document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(x => x.classList.toggle('active', x === t));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === `pane-${t.dataset.tab}`));
  }));

  // ---------- ghi âm ----------
  const btnRec = $('#btn-record');
  btnRec.addEventListener('click', async () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') { mediaRecorder.stop(); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: false } });
      chunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        clearInterval(recTimer);
        btnRec.classList.remove('recording');
        $('#mic-title').textContent = 'Đang xử lý bản ghi…';
        const blob = new Blob(chunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        await setAudio(blob, 'mic', 'recording.wav');
        $('#mic-title').textContent = 'Đã ghi xong — nhấn để ghi lại';
      };
      mediaRecorder.start();
      recStart = Date.now();
      btnRec.classList.add('recording');
      $('#mic-title').textContent = 'Đang ghi… nhấn để dừng';
      recTimer = setInterval(() => { $('#mic-sub').textContent = `⏺ ${((Date.now() - recStart) / 1000).toFixed(1)} s`; if (Date.now() - recStart > 15000) mediaRecorder.stop(); }, 100);
    } catch (err) { showError('Không truy cập được micro: ' + err.message); }
  });

  // ---------- upload ----------
  const dz = $('#dropzone'), fi = $('#file-input');
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', (e) => { e.preventDefault(); dz.classList.remove('drag'); if (e.dataTransfer.files[0]) setAudio(e.dataTransfer.files[0], 'upload', e.dataTransfer.files[0].name); });
  fi.addEventListener('change', () => fi.files[0] && setAudio(fi.files[0], 'upload', fi.files[0].name));

  async function setAudio(blob, source, name) {
    hideError();
    try {
      const dec = await AudioUtils.decodeToWav(blob);
      if (dec.duration < 0.8) { showError('Đoạn âm thanh quá ngắn (< 0.8 s). Hãy nói 3–8 giây.'); return; }
      current = { wav: dec.wav, duration: dec.duration, pcm: dec.pcm, source, name: name.replace(/\.[^.]+$/, '') + '.wav' };
      if (audioEl) { audioEl.pause(); URL.revokeObjectURL(audioEl.src); }
      audioEl = new Audio(URL.createObjectURL(dec.wav));
      audioEl.addEventListener('timeupdate', () => { AudioUtils.drawWaveform($('#wave'), current.pcm, audioEl.currentTime / current.duration); $('#time').textContent = `${fmt(audioEl.currentTime)} / ${fmt(current.duration)}`; });
      audioEl.addEventListener('ended', () => $('#btn-play').textContent = '▶');
      AudioUtils.drawWaveform($('#wave'), dec.pcm);
      $('#time').textContent = `0:00 / ${fmt(dec.duration)}`;
      $('#btn-predict').disabled = false;
      $('#btn-play').disabled = false;
    } catch (err) { showError('Không giải mã được âm thanh: ' + err.message); }
  }
  $('#btn-play').addEventListener('click', () => { if (!audioEl) return; if (audioEl.paused) { audioEl.play(); $('#btn-play').textContent = '❚❚'; } else { audioEl.pause(); $('#btn-play').textContent = '▶'; } });
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
  // ISO giờ địa phương (không 'Z') để biến ngoại sinh hour/dow phản ánh đúng thời điểm ghi của người dùng
  const localIso = (d) => new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 19);

  // ---------- dự đoán ----------
  $('#btn-predict').addEventListener('click', async () => {
    if (!current) return;
    const btn = $('#btn-predict'); btn.disabled = true; btn.textContent = '⏳ Đang phân tích…'; hideError();
    try {
      const fd = new FormData();
      fd.append('file', current.wav, current.name);
      fd.append('source', current.source);
      fd.append('recorded_at', localIso(new Date()));
      const r = await fetch('/api/predict', { method: 'POST', body: fd });
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      renderResult(await r.json());
      loadHistory(5, '#history-body');
    } catch (err) { showError('Dự đoán thất bại: ' + err.message); }
    finally { btn.disabled = false; btn.textContent = '✧ Dự đoán độ tuổi'; }
  });

  function renderResult(res) {
    $('#result-age').textContent = LABELS[res.pred_group] || res.pred_group;
    $('#conf-pill').textContent = `◎ Độ tin cậy: ${Math.round(res.confidence * 100)}%`;
    $('#face').textContent = FACES[res.pred_group] || '🙂';
    const list = $('#prob-list'); list.innerHTML = '';
    GROUPS.forEach(g => {
      const p = res.probs[g] || 0;
      const row = document.createElement('div');
      row.className = 'prob-row' + (g === res.pred_group ? ' top' : '');
      row.innerHTML = `<span>${LABELS[g]}</span><div class="bar"><i></i></div><span class="pct">${Math.round(p * 100)}%</span>`;
      list.appendChild(row);
      requestAnimationFrame(() => row.querySelector('i').style.width = `${Math.max(2, p * 100)}%`);
    });
    const f = res.features || {};
    $('#feat-row').innerHTML = [
      f.f0_mean_hz ? `F0 ≈ ${f.f0_mean_hz} Hz` : null,
      f.formant_f1_hz ? `F1/F2 ≈ ${f.formant_f1_hz}/${f.formant_f2_hz} Hz` : null,
      `Giọng ${f.speech_duration_s}s`, `SNR ${f.snr_db} dB`, `${res.latency_ms} ms · ${res.backend}`,
    ].filter(Boolean).map(t => `<span class="chip">${t}</span>`).join('');
  }

  // ---------- lịch sử ----------
  async function loadHistory(limit, sel) {
    const r = await fetch(`/api/history?limit=${limit}`); const rows = await r.json();
    const body = $(sel); body.innerHTML = '';
    const tpl = $('#row-tpl');
    rows.forEach(h => {
      const tr = tpl.content.firstElementChild.cloneNode(true);
      tr.querySelector('.t-time').textContent = new Date(h.timestamp).toLocaleString('vi-VN', { hour12: false });
      tr.querySelector('.t-file').textContent = h.filename;
      const b = tr.querySelector('.badge'); b.textContent = LABELS[h.pred_group] || h.pred_group; b.classList.add('g' + GROUPS.indexOf(h.pred_group));
      tr.querySelector('.t-conf').textContent = `${Math.round(h.confidence * 100)}%`;
      tr.querySelector('.mini-bar i').style.width = `${Math.round(h.confidence * 100)}%`;
      tr.title = GROUPS.map(g => `${g}: ${Math.round((h.probs[g] || 0) * 100)}%`).join(' · ');
      body.appendChild(tr);
    });
    if (!rows.length) body.innerHTML = '<tr><td colspan="6" class="muted center">Chưa có dự đoán nào.</td></tr>';
  }
  $('#btn-clear').addEventListener('click', async () => { if (confirm('Xoá toàn bộ lịch sử?')) { await fetch('/api/history', { method: 'DELETE' }); loadHistory(100, '#history-body-full'); loadHistory(5, '#history-body'); } });

  async function loadModelInfo() {
    const r = await fetch('/api/model'); const m = await r.json();
    const dl = $('#model-info'); dl.innerHTML = '';
    const items = { 'Mô hình': m.run_name || 'timexer_dinov3', 'Backend suy luận': m.backend, 'Backbone': m.dino_model || 'vit_small_patch16_dinov3.lvd1689m', 'Tập test — accuracy': m.test_accuracy != null ? `${(m.test_accuracy * 100).toFixed(1)}%` : '—', 'Tập test — macro F1': m.test_macro_f1 != null ? m.test_macro_f1.toFixed(3) : '—', 'Within ±1 nhóm': m.test_within_one != null ? `${(m.test_within_one * 100).toFixed(1)}%` : '—', 'Số mẫu huấn luyện': m.n_train || '—', 'Phiên bản': m.exported_at || '—' };
    Object.entries(items).forEach(([k, v]) => dl.insertAdjacentHTML('beforeend', `<dt>${k}</dt><dd>${v ?? '—'}</dd>`));
  }

  function showError(msg) { const e = $('#error'); e.textContent = msg; e.hidden = false; }
  function hideError() { $('#error').hidden = true; }

  AudioUtils.drawWaveform($('#wave'), null);
  $('#btn-play').disabled = true;
  loadHistory(5, '#history-body');
})();
