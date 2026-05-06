/* ── JavaScript for Animal Sound MMDB UI ─────────────────────────── */

const FEATURE_GROUPS = [
  "frequency","amplitude","temporal","spectral",
  "waveform","complexity","timbre","brightness","attack","decay"
];

const FEATURE_ICONS = {
  frequency:"🎵", amplitude:"📊", temporal:"⏱️", spectral:"🌈",
  waveform:"〰️", complexity:"🔀", timbre:"🎨", brightness:"✨",
  attack:"⚡", decay:"📉"
};

/* ── DOM refs ─────────────────────────────────────────────────────── */
const fileInput       = document.getElementById('file-input');
const dropZone        = document.getElementById('drop-zone');
const uploadCard      = document.getElementById('upload-card');
const processing      = document.getElementById('processing');
const resultsSection  = document.getElementById('results-section');
const uploadSection   = document.querySelector('.upload-section');
const resultsList     = document.getElementById('results-list');
const featureAccordion= document.getElementById('feature-accordion');
const qFilename       = document.getElementById('q-filename');
const newQueryBtn     = document.getElementById('new-query-btn');
const audioModal      = document.getElementById('audio-modal');
const modalBackdrop   = document.getElementById('modal-backdrop');
const modalClose      = document.getElementById('modal-close');
const modalAudio      = document.getElementById('modal-audio');
const modalTitle      = document.getElementById('modal-title');

/* ── Drag & Drop ─────────────────────────────────────────────────── */
['dragenter','dragover'].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); uploadCard.classList.add('drag-over'); })
);
['dragleave','drop'].forEach(evt =>
  dropZone.addEventListener(evt, e => { e.preventDefault(); uploadCard.classList.remove('drag-over'); })
);
dropZone.addEventListener('drop', e => {
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

/* ── Processing steps animator ───────────────────────────────────── */
function animateSteps() {
  const steps = document.querySelectorAll('.proc-step');
  let i = 0;
  return new Promise(resolve => {
    const interval = setInterval(() => {
      if (i > 0) steps[i-1].classList.remove('active'), steps[i-1].classList.add('done');
      if (i < steps.length) {
        steps[i].classList.add('active');
        i++;
      } else {
        clearInterval(interval);
        resolve();
      }
    }, 600);
  });
}

/* ── Main file handler ───────────────────────────────────────────── */
async function handleFile(file) {
  const allowed = ['.wav','.mp3','.ogg','.flac'];
  const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  if (!allowed.includes(ext)) {
    alert('Unsupported file format. Please use WAV, MP3, OGG, or FLAC.');
    return;
  }

  // Switch to processing state
  dropZone.classList.add('hidden');
  processing.classList.remove('hidden');
  resultsSection.classList.add('hidden');

  // Reset steps
  document.querySelectorAll('.proc-step').forEach(s => {
    s.classList.remove('active','done');
  });

  const stepAnim = animateSteps();

  const formData = new FormData();
  formData.append('file', file);

  let data;
  try {
    const resp = await fetch('/api/retrieve', { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.error || 'Server error');
    }
    data = await resp.json();
  } catch (err) {
    alert('Error: ' + err.message);
    resetUpload();
    return;
  }

  await stepAnim;

  // Render results
  qFilename.textContent = data.query_filename;
  renderResults(data.results);
  renderFeatureAccordion(data.query_features);

  // Show results
  processing.classList.add('hidden');
  dropZone.classList.remove('hidden');
  uploadSection.classList.add('hidden');
  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ── Render result cards ─────────────────────────────────────────── */
function renderResults(results) {
  resultsList.innerHTML = '';
  results.forEach((r, idx) => {
    const isTop = idx === 0;
    const card = document.createElement('div');
    card.className = `result-card${isTop ? ' rank-1' : ''}`;
    card.innerHTML = `
      <div class="rc-rank">${r.rank}</div>
      <div class="rc-info">
        <div class="rc-species">${r.species.replace(/_/g,' ')}</div>
        <div class="rc-filename">${r.filename}</div>
      </div>
      <div class="rc-sim">
        <div class="rc-pct">${r.similarity_pct}%</div>
        <div class="rc-lbl">similarity</div>
      </div>
    `;
    card.addEventListener('click', () => openAudio(r.audio_url, r.filename, r.species));

    // Group similarity bars
    const groups = document.createElement('div');
    groups.className = 'result-groups';
    FEATURE_GROUPS.forEach(g => {
      const sim = r.per_group[g] ?? 0;
      const pct = Math.max(0, Math.min(100, sim * 100));
      groups.innerHTML += `
        <div class="rg-bar-row">
          <span class="rg-name">${FEATURE_ICONS[g]} ${g}</span>
          <div class="rg-bar-bg"><div class="rg-bar" style="width:${pct}%"></div></div>
          <span class="rg-val">${(sim*100).toFixed(1)}%</span>
        </div>`;
    });

    // Wrap card + bars in a container
    const wrap = document.createElement('div');
    wrap.style.borderRadius = '14px';
    wrap.style.overflow = 'hidden';
    wrap.style.marginBottom = '16px';
    wrap.style.border = '1px solid var(--border)';
    wrap.style.background = 'var(--surface)';

    // Remove card's own border/background since wrap provides it
    card.style.border = 'none';
    card.style.borderRadius = '0';
    card.style.marginBottom = '0';
    wrap.appendChild(card);
    wrap.appendChild(groups);
    resultsList.appendChild(wrap);
  });
}

/* ── Render feature accordion ─────────────────────────────────────── */
function renderFeatureAccordion(queryFeatures) {
  featureAccordion.innerHTML = '';
  FEATURE_GROUPS.forEach(group => {
    const vals = queryFeatures[group] || {};
    const item = document.createElement('div');
    item.className = 'feat-item';
    const keys = Object.keys(vals);
    item.innerHTML = `
      <div class="feat-header">
        <span class="feat-hname">${FEATURE_ICONS[group]} ${group.charAt(0).toUpperCase()+group.slice(1)}</span>
        <span class="feat-toggle">▾</span>
      </div>
      <div class="feat-body">
        ${keys.map(k => {
          const v = vals[k];
          const display = Array.isArray(v) ? `[${v.join(', ')}…]` : v;
          return `<div class="feat-kv"><span class="feat-key">${k}</span><span class="feat-val">${display}</span></div>`;
        }).join('')}
      </div>`;
    item.querySelector('.feat-header').addEventListener('click', () => {
      item.classList.toggle('open');
    });
    featureAccordion.appendChild(item);
  });
}

/* ── Audio player modal ───────────────────────────────────────────── */
function openAudio(url, filename, species) {
  modalTitle.textContent = `${species.replace(/_/g,' ')} — ${filename}`;
  modalAudio.src = url;
  modalAudio.play();
  audioModal.classList.remove('hidden');
}
function closeAudio() {
  modalAudio.pause();
  modalAudio.src = '';
  audioModal.classList.add('hidden');
}
modalClose.addEventListener('click', closeAudio);
modalBackdrop.addEventListener('click', closeAudio);

/* ── New query button ────────────────────────────────────────────── */
newQueryBtn.addEventListener('click', resetUpload);
function resetUpload() {
  uploadSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  fileInput.value = '';
  uploadSection.scrollIntoView({ behavior: 'smooth' });
}
