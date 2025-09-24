const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const form = $('#qaForm');
const questionEl = $('#question');
const imageInput = $('#image');
const clearImageBtn = $('#clearImage');
const fileNameEl = $('#fileName');
const preview = $('#imagePreview');
const submitBtn = $('#submitBtn');
const progress = document.querySelector('#progressTop') || document.querySelector('#progress');
const finalAnswerEl = $('#finalAnswer');
const imagesPanel = $('#imagesPanel');
const imagesCountEl = $('#imagesCount');
const imagesGrid = $('#imagesGrid');
const transcriptPanel = $('#transcriptPanel');
const transcriptEl = $('#transcript');
const toolRunsEl = $('#toolRuns');

let _toolCards = [];
let _uploadedSrc = null; 


const IMAGE_TOOL_NAMES = [
  'Point',
  'ZoomInSubfigure',
  'SegmentRegionAroundPoint',
  'DrawHorizontalLineByY',
  'DrawVerticalLineByX',
];


const BADGE_TOOL_NAMES = [...IMAGE_TOOL_NAMES, 'OCR'];

function pretty(obj) {
  try { return JSON.stringify(obj, null, 2); } catch { return String(obj); }
}

function createToolCard(name, args) {
  const idx = _toolCards.length + 1;
  const card = document.createElement('div');

  card.className = 'tool-card collapsed';

  const header = document.createElement('div');
  header.className = 'tool-card-header';
  const isImageTool = IMAGE_TOOL_NAMES.includes(name);
  const showBadgeHeader = BADGE_TOOL_NAMES.includes(name);
  const paramTitle = args && (args.param ?? args.prompt ?? args.phrase ?? null);
  let nameEl;
  if (showBadgeHeader) {

    
    if (paramTitle) {
      const title = document.createElement('div');
      title.className = 'tool-param-title';
      title.textContent = _capFirst(paramTitle);
      header.appendChild(title);
    }
    const badge = document.createElement('span');
    badge.className = 'tool-name-badge';
    badge.textContent = `${name} Tool`;
    header.appendChild(badge);
  } else {
    nameEl = document.createElement('span');
    nameEl.className = 'tool-name';
    nameEl.textContent = `#${idx} ${name}`;
    header.append(nameEl);
  }
  const toggle = document.createElement('button');
  toggle.className = 'tool-toggle';
  toggle.type = 'button';
  toggle.addEventListener('click', () => {
    card.classList.toggle('collapsed');
    card.classList.toggle('open');
  });
  header.append(toggle);

  const body = document.createElement('div');
  let meta = null;
  let out;
  if (isImageTool) {
    body.className = 'tool-card-body image-only';
    out = document.createElement('div');
    out.className = 'tool-image-body';
    body.append(out);
  } else {
    body.className = 'tool-card-body';
    meta = document.createElement('div');
    meta.className = 'tool-section tool-meta';
    meta.innerHTML = `<h4>Call</h4><pre class="tool-args">${args ? pretty(args) : 'No arguments'}</pre>`;
    out = document.createElement('div');
    out.className = 'tool-section tool-output';
    out.innerHTML = '<h4>Result</h4>';
    body.append(meta, out);
  }
  card.append(header, body);
  toolRunsEl.appendChild(card);
  const entry = { card, meta, out, isImageTool };
  _toolCards.push(entry);
  return entry;
}

function setLoading(loading) {
  submitBtn.disabled = loading;
  progress.classList.toggle('hidden', !loading);
}

function renderPreview(file) {
  if (!file) { preview.innerHTML = ''; preview.classList.add('hidden'); return; }
  const reader = new FileReader();
  reader.onload = (e) => {
    _uploadedSrc = e.target.result;
    preview.innerHTML = `<img src="${_uploadedSrc}" alt="Preview">`;
    preview.classList.remove('hidden');
  };
  reader.readAsDataURL(file);
}

imageInput.addEventListener('change', () => {
  const file = imageInput.files?.[0];
  renderPreview(file);
  fileNameEl.textContent = file ? file.name : 'No file chosen';
});

clearImageBtn.addEventListener('click', (e) => {

  e.preventDefault();
  e.stopPropagation();

  questionEl.value = '';
  imageInput.value = '';
  preview.innerHTML = '';
  preview.classList.add('hidden');
  fileNameEl.textContent = 'No file chosen';
  _uploadedSrc = null;
  return false;
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = questionEl.value.trim();
  const file = imageInput.files?.[0];
  if (!question) { alert('Please enter a question.'); return; }
  if (!file) { alert('Please select an image.'); return; }

  setLoading(true);
  finalAnswerEl.textContent = '';
  finalAnswerEl.classList.remove('placeholder');
  imagesGrid.innerHTML = '';
  imagesPanel.classList.add('hidden');
  transcriptEl.textContent = '';
  transcriptPanel.classList.add('hidden');

  try {
    const fd = new FormData();
    fd.append('question', question);
    fd.append('image', file);
    const res = await fetch('/api/ask_stream', { method: 'POST', body: fd });
    if (!res.ok || !res.body) {
      const text = await res.text().catch(() => '');
      throw new Error(text || `Request failed (${res.status})`);
    }

    transcriptPanel.classList.remove('hidden');
    transcriptEl.textContent = '';

    transcriptEl.classList.add('hidden');
    toolRunsEl.innerHTML = '';
    _toolCards = [];
    const imageIds = new Set();

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    const appendTranscript = (line) => {
      transcriptEl.textContent += (transcriptEl.textContent ? '\n' : '') + line;
      transcriptEl.scrollTop = transcriptEl.scrollHeight;
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf('\n')) >= 0) {
        const raw = buf.slice(0, idx).trim();
        buf = buf.slice(idx + 1);
        if (!raw) continue;
        let evt;
        try { evt = JSON.parse(raw); } catch { continue; }
        const type = evt.event;
        if (type === 'start') {

        } else if (type === 'tool_started') {
          const name = evt.tool || 'Tool';

          const args = evt.args || evt.parameters || evt.params || null;
          createToolCard(name, args);
        } else if (type === 'tool_result') {
          const name = evt.tool || 'Tool';
          const summary = typeof evt.result === 'string' ? evt.result : JSON.stringify(evt.result);

          const last = _toolCards[_toolCards.length - 1] || createToolCard(name, null);
          if (evt.image && evt.image.url) {
            const fig = document.createElement('figure');
            fig.className = 'tool-result-figure';
            fig.innerHTML = `<a class="tool-image-link" href="${evt.image.url}" target="_blank" rel="noopener noreferrer"><img src="${evt.image.url}" alt="${evt.image.id || name}"></a>`;
            last.out.appendChild(fig);
          } else if (summary) {

            if (name === 'Point') {
              const m = /<point[^>]*x="([\d.]+)"[^>]*y="([\d.]+)"/i.exec(summary);
              if (m && _uploadedSrc) {
                const x = parseFloat(m[1]);
                const y = parseFloat(m[2]);
                const wrap = document.createElement('div');
                wrap.className = 'annotated-wrapper';
                const toolbar = document.createElement('div');
                toolbar.className = 'annotated-toolbar';
                const rotateBtn = document.createElement('button');
                rotateBtn.type = 'button';
                rotateBtn.textContent = 'Rotate 90°';
                const canvas = document.createElement('canvas');
                canvas.className = 'annotated-canvas';

                let angle = 0; // 0, 90, 180, 270
                const imgEl = new Image();
                imgEl.onload = () => {
                  const draw = () => {

                    const rads = (angle % 360) * Math.PI / 180;
                    const is90 = angle % 180 !== 0;
                    const srcW = imgEl.naturalWidth || imgEl.width;
                    const srcH = imgEl.naturalHeight || imgEl.height;
                    const destW = is90 ? srcH : srcW;
                    const destH = is90 ? srcW : srcH;

                    // Scale down to fit compact card: width and height caps
                    const maxW = 640;  // px
                    const maxH = 360;  // px
                    const scale = Math.min(1, maxW / destW, maxH / destH);

                    canvas.width = Math.round(destW * scale);
                    canvas.height = Math.round(destH * scale);
                    const ctx = canvas.getContext('2d');
                    ctx.save();
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    // Center and rotate
                    ctx.translate(canvas.width / 2, canvas.height / 2);
                    ctx.rotate(rads);
                    const drawW = srcW * scale;
                    const drawH = srcH * scale;
                    ctx.drawImage(imgEl, -drawW / 2, -drawH / 2, drawW, drawH);

                    // Transform point under rotation (around image center)
                    let px = x, py = y;
                    if (angle === 90) { px = srcH - y; py = x; }
                    else if (angle === 180) { px = srcW - x; py = srcH - y; }
                    else if (angle === 270) { px = y; py = srcW - x; }
                    const sx = (px - srcW / 2) * scale;
                    const sy = (py - srcH / 2) * scale;

                    // Draw marker (brand blue crosshair + dot with glow)
                    const R = 8; // radius in px (scaled already)
                    ctx.lineWidth = 2;
                    ctx.strokeStyle = 'rgba(0,153,255,0.95)';
                    ctx.fillStyle = 'rgba(0,153,255,0.95)';
                    ctx.shadowColor = 'rgba(0,153,255,0.35)';
                    ctx.shadowBlur = 6;

                    // Crosshair
                    ctx.beginPath();
                    ctx.moveTo(sx - R, sy); ctx.lineTo(sx + R, sy);
                    ctx.moveTo(sx, sy - R); ctx.lineTo(sx, sy + R);
                    ctx.stroke();

                    // Dot
                    ctx.beginPath();
                    ctx.arc(sx, sy, 3, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                    ctx.restore();
                  };
                  draw();
                  rotateBtn.onclick = () => { angle = (angle + 90) % 360; draw(); };

                  canvas.style.cursor = 'zoom-in';
                  canvas.addEventListener('click', () => { try { window.open(_uploadedSrc, '_blank'); } catch {} });
                };
                imgEl.src = _uploadedSrc;

                toolbar.appendChild(rotateBtn);
                wrap.append(toolbar, canvas);
                last.out.appendChild(wrap);
              } else {
                const pre = document.createElement('pre');
                pre.className = 'tool-result-text';
                pre.textContent = summary;
                last.out.appendChild(pre);
              }
            } else {
              const pre = document.createElement('pre');
              pre.className = 'tool-result-text';
              pre.textContent = summary;
              last.out.appendChild(pre);
            }
          }
          if (evt.image && evt.image.id && evt.image.url) {
            if (!imageIds.has(evt.image.id)) {
              imageIds.add(evt.image.id);
              imagesPanel.classList.remove('hidden');
              const fig = document.createElement('figure');
              fig.innerHTML = `<img src="${evt.image.url}" alt="${evt.image.id}"><figcaption>${evt.image.id}</figcaption>`;
              imagesGrid.appendChild(fig);
              imagesCountEl.textContent = String(imageIds.size);
            }
          }
        } else if (type === 'final_answer') {
          finalAnswerEl.textContent = evt.answer || '(No final answer returned)';
        } else if (type === 'error') {
          finalAnswerEl.textContent = `Error: ${evt.message || 'Unknown error'}`;
        } else if (type === 'done') {

        }
      }
    }
  } catch (err) {
    finalAnswerEl.textContent = `Error: ${err?.message || err}`;
  } finally {
    setLoading(false);
  }
});

function _capFirst(s) {
  if (s == null) return s;
  const str = String(s);
  return str.length ? str[0].toUpperCase() + str.slice(1) : str;
}
