/* Live FSL demo: MediaPipe hand landmarks + in-browser k-NN classifier you train yourself.
   Uses BONES and TIPS from script.js. */
const MP_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
const MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const STORE_KEY = 'fsl-samples-v1';
const K = 5, SAMPLES_PER_GESTURE = 40, HOLD_FRAMES = 15, MAX_DIST = 1.2, MIN_CONF = 0.6;

const $ = id => document.getElementById(id);
const video = $('cam'), canvas = $('cam-overlay'), ctx = canvas.getContext('2d');
const stage = video.parentElement;
const camBtn = $('cam-toggle'), statusEl = $('cam-status'), countEl = $('count');
const resLabel = $('res-label'), resBar = $('res-bar'), resConf = $('res-conf');
const transcriptEl = $('transcript'), speakBox = $('speak');
const labelIn = $('label-in'), recBtn = $('rec'), labelsEl = $('labels'), impFile = $('imp-file');

let landmarker = null, stream = null, running = false, lastTime = -1;
let samples = loadSamples();            // { name: [ [60 numbers], ... ] }
let recording = false, recName = '', recBuf = [];
let candidate = '', candCount = 0, lastCommitted = '', noHand = 0;
let transcript = '', lastTokenLen = 0;

/* ---------- storage ---------- */
function loadSamples() {
  try { return JSON.parse(localStorage.getItem(STORE_KEY)) || {}; } catch { return {}; }
}
function saveSamples() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(samples)); } catch { /* storage unavailable */ }
}
function renderLabels(hit) {
  labelsEl.innerHTML = '';
  Object.keys(samples).forEach(name => {
    const li = document.createElement('li');
    if (name === hit) li.className = 'hit';
    li.append(`${name} (${samples[name].length})`);
    const del = document.createElement('button');
    del.type = 'button'; del.textContent = '×'; del.setAttribute('aria-label', `Delete ${name}`);
    del.onclick = () => { delete samples[name]; saveSamples(); renderLabels(); };
    li.append(del);
    labelsEl.append(li);
  });
}

/* ---------- model + camera ---------- */
async function loadModel() {
  if (landmarker) return;
  statusEl.textContent = 'Loading hand model…';
  const { HandLandmarker, FilesetResolver } = await import(`${MP_BASE}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
  const opts = delegate => ({ baseOptions: { modelAssetPath: MODEL_URL, delegate }, runningMode: 'VIDEO', numHands: 1 });
  try { landmarker = await HandLandmarker.createFromOptions(files, opts('GPU')); }
  catch { landmarker = await HandLandmarker.createFromOptions(files, opts('CPU')); }
}

async function startCamera() {
  camBtn.disabled = true;
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error('insecure');
    await loadModel();
    statusEl.textContent = 'Waiting for camera permission…';
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    stage.classList.add('on');
    running = true;
    camBtn.textContent = 'Stop camera';
    statusEl.textContent = 'Camera on. Show your hand.';
    requestAnimationFrame(loop);
  } catch (err) {
    statusEl.textContent = explain(err);
  } finally { camBtn.disabled = false; }
}

function explain(err) {
  if (err.message === 'insecure') return 'Camera access needs a secure page. Open the site with Live Server (localhost) or over https.';
  if (err.name === 'NotAllowedError') return 'Camera permission was blocked. Allow the camera for this site in your browser settings, then try again.';
  if (err.name === 'NotFoundError') return 'No camera was found on this device.';
  if (err.name === 'NotReadableError') return 'The camera is in use by another app. Close it and try again.';
  return 'Could not start the demo. Check your internet connection (the hand model loads online) and try again.';
}

function stopCamera() {
  running = false; recording = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  stream = null; video.srcObject = null;
  stage.classList.remove('on');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  camBtn.textContent = 'Start camera';
  statusEl.textContent = 'Camera off.';
  countEl.hidden = true; recBtn.disabled = false;
  showResult('', 0, 'Camera off.');
}

/* ---------- features + classifier ---------- */
function features(pts, handedness) {
  const aspect = video.videoWidth / video.videoHeight;
  const w = pts[0], flip = handedness === 'Left' ? -1 : 1;
  const m = pts[9];
  const s = Math.hypot((m.x - w.x) * aspect, m.y - w.y, (m.z - w.z) * aspect) || 1;
  const v = [];
  for (let i = 1; i < 21; i++) {
    v.push(flip * (pts[i].x - w.x) * aspect / s, (pts[i].y - w.y) / s, (pts[i].z - w.z) * aspect / s);
  }
  return v;
}
function dist(a, b) { let t = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; t += d * d; } return Math.sqrt(t); }

function classify(vec) {
  const all = [];
  for (const name in samples) for (const s of samples[name]) all.push({ name, d: dist(vec, s) });
  if (!all.length) return null;
  all.sort((a, b) => a.d - b.d);
  const near = all.slice(0, K);
  if (near[0].d > MAX_DIST) return { name: '', conf: 0 };
  const votes = {}; let total = 0;
  near.forEach(n => { const wt = 1 / (n.d + 0.01); votes[n.name] = (votes[n.name] || 0) + wt; total += wt; });
  const best = Object.keys(votes).sort((a, b) => votes[b] - votes[a])[0];
  return { name: best, conf: votes[best] / total };
}

/* ---------- main loop ---------- */
function loop() {
  if (!running) return;
  if (video.currentTime !== lastTime) {
    lastTime = video.currentTime;
    const res = landmarker.detectForVideo(video, performance.now());
    const hands = res.landmarks || [];
    draw(hands);
    if (hands.length) { noHand = 0; onHand(hands[0], handednessOf(res)); }
    else { noHand++; if (noHand > 8) { candidate = ''; candCount = 0; lastCommitted = ''; } onNoHand(); }
  }
  requestAnimationFrame(loop);
}
function handednessOf(res) {
  const h = res.handedness || res.handednesses;
  return h && h[0] && h[0][0] ? h[0][0].categoryName : 'Right';
}

function onHand(pts, handed) {
  const vec = features(pts, handed);
  if (recording) { recBuf.push(vec); statusEl.textContent = `Recording “${recName}”… ${recBuf.length}/${SAMPLES_PER_GESTURE}`; if (recBuf.length >= SAMPLES_PER_GESTURE) finishRecording(); return; }
  const r = classify(vec);
  if (!r) { showResult('', 0, 'Teach at least one gesture to begin.'); statusEl.textContent = 'Hand detected.'; return; }
  statusEl.textContent = 'Hand detected.';
  if (r.name && r.conf >= MIN_CONF) {
    showResult(r.name, r.conf, `${Math.round(r.conf * 100)}% match`);
    if (r.name === candidate) candCount++; else { candidate = r.name; candCount = 1; }
    if (candCount >= HOLD_FRAMES && candidate !== lastCommitted) { lastCommitted = candidate; commit(candidate); }
  } else {
    showResult('', 0, 'No taught gesture matches.');
    candidate = ''; candCount = 0; lastCommitted = '';
  }
}
function onNoHand() {
  if (recording) { statusEl.textContent = 'Hand lost. Bring it back into view to continue recording.'; return; }
  if (running) { statusEl.textContent = 'Camera on. Show your hand.'; showResult('', 0, Object.keys(samples).length ? 'Show a taught gesture.' : 'Teach at least one gesture to begin.'); }
}
function showResult(name, conf, text) {
  resLabel.textContent = name || '—';
  resBar.style.width = `${Math.round(conf * 100)}%`;
  resConf.textContent = text;
  renderHit(name);
}
let lastHit = null;
function renderHit(name) {
  if (name === lastHit) return; lastHit = name;
  [...labelsEl.children].forEach(li => li.classList.toggle('hit', name && li.firstChild.textContent.startsWith(name + ' (')));
}

/* ---------- transcript ---------- */
function commit(name) {
  const single = name.length === 1;
  if (transcript && !transcript.endsWith(' ') && !(single && lastTokenLen === 1)) transcript += ' ';
  transcript += name; lastTokenLen = name.length;
  transcriptEl.textContent = transcript;
  if (speakBox.checked && 'speechSynthesis' in window) speechSynthesis.speak(new SpeechSynthesisUtterance(name));
}
$('tr-space').onclick = () => { if (transcript && !transcript.endsWith(' ')) { transcript += ' '; lastTokenLen = 0; transcriptEl.textContent = transcript; } };
$('tr-clear').onclick = () => { transcript = ''; lastTokenLen = 0; transcriptEl.textContent = 'Recognized gestures appear here.'; };

/* ---------- teaching ---------- */
recBtn.onclick = () => {
  const name = labelIn.value.trim();
  if (!running) { statusEl.textContent = 'Start the camera first.'; return; }
  if (!name) { statusEl.textContent = 'Type a name for the gesture first.'; labelIn.focus(); return; }
  recBtn.disabled = true; let n = 3;
  statusEl.textContent = `Get your “${name}” sign ready…`;
  countEl.hidden = false; countEl.textContent = n;
  const t = setInterval(() => {
    n--;
    if (n > 0) { countEl.textContent = n; return; }
    clearInterval(t); countEl.hidden = true;
    recName = name; recBuf = []; recording = true;
  }, 1000);
};
function finishRecording() {
  recording = false;
  samples[recName] = (samples[recName] || []).concat(recBuf);
  saveSamples(); renderLabels();
  statusEl.textContent = `Saved “${recName}”. Record it again from a slightly different angle for better accuracy.`;
  recBuf = []; recBtn.disabled = false; lastHit = null;
}

$('exp').onclick = () => {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(samples)], { type: 'application/json' }));
  a.download = 'fsl-gesture-samples.json'; a.click(); URL.revokeObjectURL(a.href);
};
$('imp').onclick = () => impFile.click();
impFile.onchange = async () => {
  try {
    const data = JSON.parse(await impFile.files[0].text());
    for (const k in data) if (Array.isArray(data[k]) && data[k].every(v => Array.isArray(v) && v.length === 60)) samples[k] = (samples[k] || []).concat(data[k]);
    saveSamples(); renderLabels(); statusEl.textContent = 'Imported gestures.';
  } catch { statusEl.textContent = 'That file is not a valid gesture export.'; }
  impFile.value = '';
};
$('wipe').onclick = () => { if (confirm('Delete all taught gestures?')) { samples = {}; saveSamples(); renderLabels(); } };

/* ---------- drawing ---------- */
function draw(hands) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  hands.forEach(pts => {
    ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = 3; ctx.lineCap = 'round';
    BONES.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(pts[a].x * W, pts[a].y * H); ctx.lineTo(pts[b].x * W, pts[b].y * H); ctx.stroke(); });
    pts.forEach((p, i) => {
      ctx.beginPath(); ctx.arc(p.x * W, p.y * H, TIPS.includes(i) ? 7 : 5, 0, Math.PI * 2);
      ctx.fillStyle = TIPS.includes(i) ? '#F2B230' : (i === 0 ? '#C8372D' : '#8FB1F2'); ctx.fill();
    });
  });
}

camBtn.addEventListener('click', () => (running ? stopCamera() : startCamera()));
renderLabels();
