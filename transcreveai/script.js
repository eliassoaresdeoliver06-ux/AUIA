import { pipeline, env } from '@huggingface/transformers';
import { FFmpeg } from '@ffmpeg/ffmpeg';
import { toBlobURL } from '@ffmpeg/util';

const MODEL_ID = 'onnx-community/whisper-small';
const ACCEPTED_EXTENSIONS = new Set(['mp3', 'wav', 'm4a', 'ogg', 'flac', 'mp4', 'mov', 'webm']);
const LANGUAGE_NAMES = { pt: 'portuguese', en: 'english', es: 'spanish', fr: 'french', de: 'german', it: 'italian' };

env.allowLocalModels = false;
env.useBrowserCache = true;

const $ = (selector) => document.querySelector(selector);
const dropZone = $('#drop-zone');
const fileInput = $('#file-input');
const filePanel = $('#file-panel');
const mediaPreview = $('#media-preview');
const transcribeButton = $('#transcribe-button');
const progressPanel = $('#progress-panel');
const resultPanel = $('#result-panel');
const transcript = $('#transcript');
const alertBox = $('#alert');
const comparePanel = $('#compare-panel');
const comparePlayerWrap = $('#compare-player-wrap');
const segmentList = $('#segment-list');
let selectedFile = null;
let mediaUrl = null;
let compareMedia = null;
let transcriber = null;
let transcriberDevice = null;
let ffmpeg = null;
let ffmpegLoading = null;
let segments = [];
let isTranscribing = false;

function correctionKey() {
  return `auia-corrections-${$('#language-select').value}`;
}

function getCorrections() {
  try {
    return JSON.parse(localStorage.getItem(correctionKey()) || '{}');
  } catch {
    return {};
  }
}

function applyLearnedCorrections(text) {
  const corrections = getCorrections();
  return Object.entries(corrections).reduce((result, [wrong, right]) => {
    const escaped = wrong.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const pattern = /\s/.test(wrong) ? escaped : `\\b${escaped}\\b`;
    return result.replace(new RegExp(pattern, 'giu'), right);
  }, text);
}

function showAlert(message) {
  alertBox.querySelector('p').textContent = message;
  alertBox.classList.remove('hidden');
}

function hideAlert() { alertBox.classList.add('hidden'); }

function formatBytes(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return '--:--';
  const total = Math.max(0, Math.round(seconds));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  return hours ? `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}` : `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function formatSrtTime(seconds) {
  const total = Math.max(0, Math.round(seconds * 1000));
  const hours = Math.floor(total / 3600000);
  const minutes = Math.floor((total % 3600000) / 60000);
  const secs = Math.floor((total % 60000) / 1000);
  const milliseconds = total % 1000;
  return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(milliseconds).padStart(3, '0')}`;
}

function setProgress(percent, label, detail) {
  $('#progress-bar').style.width = `${percent}%`;
  $('#progress-percent').textContent = `${Math.round(percent)}%`;
  $('#progress-label').textContent = label;
  if (detail) $('#progress-detail').textContent = detail;
}

function extensionOf(fileName) { return fileName.split('.').pop().toLowerCase(); }

function readDuration(element) {
  return new Promise((resolve) => {
    if (Number.isFinite(element.duration)) return resolve(element.duration);
    element.addEventListener('loadedmetadata', () => resolve(element.duration), { once: true });
    element.addEventListener('error', () => resolve(null), { once: true });
  });
}

async function handleFile(file) {
  hideAlert();
  if (!file) return;
  const extension = extensionOf(file.name);
  if (!ACCEPTED_EXTENSIONS.has(extension)) {
    showAlert('Formato não suportado. Escolha um arquivo MP3, WAV, M4A, OGG, FLAC, MP4, MOV ou WEBM.');
    return;
  }
  if (file.size > 1.5 * 1024 ** 3) {
    showAlert('Este arquivo é muito grande para o processamento local. Tente um arquivo menor que 1,5 GB.');
    return;
  }
  clearResultOnly();
  comparePanel.classList.add('hidden');
  selectedFile = file;
  if (mediaUrl) URL.revokeObjectURL(mediaUrl);
  mediaUrl = URL.createObjectURL(file);
  const isVideo = file.type.startsWith('video') || ['mp4', 'mov', 'webm'].includes(extension);
  const media = document.createElement(isVideo ? 'video' : 'audio');
  media.controls = true;
  media.preload = 'metadata';
  media.src = mediaUrl;
  if (isVideo) media.setAttribute('playsinline', '');
  mediaPreview.replaceChildren(media);
  compareMedia = media.cloneNode(false);
  compareMedia.controls = true;
  compareMedia.preload = 'metadata';
  compareMedia.src = mediaUrl;
  if (isVideo) compareMedia.setAttribute('playsinline', '');
  comparePlayerWrap.replaceChildren(compareMedia);
  $('#file-name').textContent = file.name;
  $('#file-format').textContent = extension.toUpperCase();
  $('#file-size').textContent = formatBytes(file.size);
  $('#file-duration').textContent = 'Calculando...';
  filePanel.classList.remove('hidden');
  const duration = await readDuration(media);
  $('#file-duration').textContent = duration ? formatTime(duration) : 'Indisponível';
}

function clearResultOnly() {
  transcript.value = '';
  segments = [];
  $('#word-count').textContent = '0 palavras';
  $('#success-badge').classList.add('hidden');
}

function resetApp() {
  if (isTranscribing) return;
  selectedFile = null;
  if (mediaUrl) URL.revokeObjectURL(mediaUrl);
  mediaUrl = null;
  mediaPreview.replaceChildren();
  comparePlayerWrap.replaceChildren();
  compareMedia = null;
  comparePanel.classList.add('hidden');
  filePanel.classList.add('hidden');
  progressPanel.classList.add('hidden');
  clearResultOnly();
  hideAlert();
  fileInput.value = '';
}

function enhanceSamples(samples) {
  const normalized = new Float32Array(samples.length);
  let sum = 0;
  let peak = 0;
  for (const sample of samples) {
    sum += sample;
    peak = Math.max(peak, Math.abs(sample));
  }
  const offset = sum / samples.length;
  let energy = 0;
  for (const sample of samples) energy += (sample - offset) ** 2;
  const rms = Math.sqrt(energy / samples.length);
  const gain = rms > 0 ? Math.min(3.5, 0.14 / rms) : 1;
  const peakGain = peak > 0 ? Math.min(gain, 0.95 / peak) : gain;
  for (let index = 0; index < samples.length; index += 1) {
    normalized[index] = Math.max(-1, Math.min(1, (samples[index] - offset) * peakGain));
  }
  return normalized;
}

async function decodeWithFfmpeg(file) {
  if (!ffmpeg) ffmpeg = new FFmpeg();
  if (!ffmpeg.loaded) {
    if (!ffmpegLoading) {
      ffmpegLoading = (async () => {
        const baseURL = 'https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.12.10/dist/esm';
        const ffmpegPackageURL = 'https://cdn.jsdelivr.net/npm/@ffmpeg/ffmpeg@0.12.15/dist/esm';
        const workerSource = await (await fetch(`${ffmpegPackageURL}/worker.js`)).text();
        const workerWithAbsoluteImports = workerSource.replace(/from ["']\.\/(const|errors)\.js["']/g, `from "${ffmpegPackageURL}/$1.js"`);
        const workerURL = URL.createObjectURL(new Blob([workerWithAbsoluteImports], { type: 'text/javascript' }));
        await ffmpeg.load({
          coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
          wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
          classWorkerURL: workerURL,
        });
      })();
    }
    await ffmpegLoading;
  }
  const inputName = `input-${Date.now()}.mp4`;
  const outputName = `audio-${Date.now()}.wav`;
  await ffmpeg.writeFile(inputName, new Uint8Array(await file.arrayBuffer()));
  await ffmpeg.exec(['-i', inputName, '-map', '0:a:0', '-vn', '-sn', '-dn', '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', outputName]);
  const data = await ffmpeg.readFile(outputName);
  await ffmpeg.deleteFile(inputName);
  await ffmpeg.deleteFile(outputName);
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const dataStart = new TextDecoder().decode(data).indexOf('data');
  if (dataStart < 0) throw new Error('FFmpeg não gerou áudio PCM');
  const samples = new Float32Array(Math.floor((data.byteLength - dataStart - 8) / 2));
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getInt16(dataStart + 8 + index * 2, true) / 32768;
  }
  return enhanceSamples(samples);
}

async function decodeAudio(file) {
  const context = new AudioContext();
  try {
    let buffer;
    try {
      buffer = await context.decodeAudioData(await file.arrayBuffer());
    } catch (error) {
      setProgress(12, 'Convertendo áudio...', 'Este MP4 não é compatível com o decodificador nativo; usando FFmpeg WebAssembly local.');
      return decodeWithFfmpeg(file);
    }
    if (!buffer.duration || !buffer.numberOfChannels) throw new Error('Áudio vazio');
    const targetRate = 16000;
    const frameCount = Math.ceil(buffer.duration * targetRate);
    const offline = new OfflineAudioContext(1, frameCount, targetRate);
    const source = offline.createBufferSource();
    const highPass = offline.createBiquadFilter();
    const compressor = offline.createDynamicsCompressor();
    const makeupGain = offline.createGain();
    source.buffer = buffer;
    highPass.type = 'highpass';
    highPass.frequency.value = 75;
    highPass.Q.value = 0.7;
    compressor.threshold.value = -42;
    compressor.knee.value = 24;
    compressor.ratio.value = 4;
    compressor.attack.value = 0.015;
    compressor.release.value = 0.25;
    makeupGain.gain.value = 1.35;
    source.connect(highPass).connect(compressor).connect(makeupGain).connect(offline.destination);
    source.start();
    const rendered = await offline.startRendering();
    return enhanceSamples(rendered.getChannelData(0));
  } finally {
    await context.close();
  }
}

async function loadTranscriber(forceWasm = false) {
  if (transcriber) return transcriber;
  const configurations = [];
  try {
    if (!forceWasm && navigator.gpu && await navigator.gpu.requestAdapter()) configurations.push({ device: 'webgpu', dtype: 'q4' });
  } catch (error) {
    console.warn('WebGPU indisponível, usando WASM.', error);
  }
  configurations.push({ device: 'wasm', dtype: 'q8' });

  let lastError;
  for (const configuration of configurations) {
    try {
      const candidate = await pipeline('automatic-speech-recognition', MODEL_ID, {
        ...configuration,
        progress_callback: (progress) => {
          if (progress.status === 'progress' && Number.isFinite(progress.progress)) {
            const downloadProgress = Math.min(48, progress.progress * .48);
            setProgress(downloadProgress, 'Baixando modelo...', 'O download acontece apenas na primeira execução e fica salvo no cache do navegador.');
          }
        },
      });
      transcriber = candidate;
      transcriberDevice = configuration.device;
      return transcriber;
    } catch (error) {
      lastError = error;
      if (configuration.device === 'webgpu') {
        setProgress(48, 'Preparando modo compatível...', 'A GPU não conseguiu executar o modelo; mudando automaticamente para WASM.');
      }
    }
  }
  throw lastError;
}

function extractSegments(output) {
  if (!output.chunks?.length) return [];
  return output.chunks.filter((chunk) => chunk.text?.trim()).map((chunk) => ({
    start: Number.isFinite(chunk.timestamp?.[0]) ? chunk.timestamp[0] : 0,
    end: Number.isFinite(chunk.timestamp?.[1]) ? chunk.timestamp[1] : (Number.isFinite(chunk.timestamp?.[0]) ? chunk.timestamp[0] + 3 : 3),
    text: chunk.text.trim(),
  }));
}

function removeRepeatedRuns(text) {
  const words = text.split(/(\s+)/);
  const tokens = words.filter((item) => !/^\s+$/.test(item));
  const normalized = tokens.map((item) => item.toLocaleLowerCase('pt-BR').replace(/[^\p{L}\p{N}']+/gu, ''));
  let changed = true;
  while (changed) {
    changed = false;
    for (let size = Math.min(8, Math.floor(normalized.length / 2)); size >= 1; size -= 1) {
      const start = normalized.length - size * 2;
      const first = normalized.slice(start, start + size).join(' ');
      const second = normalized.slice(start + size).join(' ');
      if (start >= 0 && first && first === second) {
        tokens.splice(start + size, size);
        normalized.splice(start + size, size);
        changed = true;
        break;
      }
    }
  }
  return tokens.reduce((result, token, index) => `${result}${index ? ' ' : ''}${token}`, '').replace(/\s+([,.!?;:])/g, '$1');
}

function renderComparison() {
  if (!compareMedia) return;
  const comparisonSegments = segments.length ? segments : [{ start: 0, end: compareMedia.duration || 10, text: transcript.value.trim() }];
  segmentList.replaceChildren();
  comparisonSegments.forEach((segment) => {
    const item = document.createElement('div');
    const button = document.createElement('button');
    const input = document.createElement('input');
    item.className = 'segment-item';
    item.dataset.start = String(segment.start);
    item.dataset.end = String(segment.end);
    button.type = 'button';
    button.className = 'segment-jump';
    button.textContent = formatTime(segment.start);
    input.className = 'segment-text';
    input.type = 'text';
    input.value = applyLearnedCorrections(segment.text);
    input.dataset.original = segment.text;
    button.addEventListener('click', () => {
      compareMedia.currentTime = segment.start;
      compareMedia.play();
      segmentList.querySelectorAll('.segment-item').forEach((item) => item.classList.remove('active'));
      item.classList.add('active');
    });
    item.append(button, input);
    segmentList.append(item);
  });
  compareMedia.ontimeupdate = () => {
    const current = compareMedia.currentTime;
    segmentList.querySelectorAll('.segment-item').forEach((item) => {
      const active = current >= Number(item.dataset.start) && current < Number(item.dataset.end);
      item.classList.toggle('active', active);
      if (active) item.scrollIntoView({ block: 'nearest' });
    });
  };
  comparePanel.classList.remove('hidden');
}

function saveLearning() {
  const corrections = getCorrections();
  let count = 0;
  segmentList.querySelectorAll('.segment-text').forEach((input) => {
    const original = input.dataset.original.trim();
    const corrected = input.value.trim();
    if (original && corrected && original !== corrected) {
      corrections[original] = corrected;
      count += 1;
    }
  });
  localStorage.setItem(correctionKey(), JSON.stringify(corrections));
  segments = segments.map((segment) => ({ ...segment, text: applyLearnedCorrections(segment.text) }));
  transcript.value = applyLearnedCorrections(transcript.value);
  updateWordCount();
  renderComparison();
  $('#learning-status').textContent = count ? `${count} correção(ões) guardada(s) neste navegador.` : 'Nenhuma alteração nova para aprender.';
}

async function runTranscription(audio, options) {
  const model = await loadTranscriber();
  try {
    return await model(audio, options);
  } catch (error) {
    if (transcriberDevice !== 'webgpu') throw error;
    transcriber = null;
    transcriberDevice = null;
    setProgress(48, 'Mudando para CPU...', 'A GPU falhou durante a inferência; tentando o modo WASM compatível.');
    return (await loadTranscriber(true))(audio, options);
  }
}

async function startTranscription() {
  if (!selectedFile || isTranscribing) return;
  isTranscribing = true;
  hideAlert();
  transcribeButton.disabled = true;
  progressPanel.classList.remove('hidden');
  setProgress(2, 'Preparando áudio...', 'O arquivo será decodificado localmente e não será enviado para nenhum servidor.');
  try {
    const audio = await decodeAudio(selectedFile);
    setProgress(50, '🧠 Transcrevendo...', 'A inteligência artificial está trabalhando no seu dispositivo.');
    const language = $('#language-select').value;
    const options = {
      chunk_length_s: 30,
      stride_length_s: 3,
      return_timestamps: true,
      task: 'transcribe',
      condition_on_previous_text: false,
      no_repeat_ngram_size: 3,
      repetition_penalty: 1.08,
      temperature: 0,
      top_k: 0,
      num_beams: 5,
      length_penalty: 1,
    };
    if (language !== 'auto') options.language = LANGUAGE_NAMES[language];
    const output = await runTranscription(audio, options);
    segments = extractSegments(output);
    const rawText = output.text?.trim() || segments.map((segment) => segment.text).join(' ');
    transcript.value = applyLearnedCorrections(removeRepeatedRuns(rawText));
    updateWordCount();
    renderComparison();
    setProgress(100, '✅ Transcrição concluída', 'Revise o texto, edite o que quiser e exporte no formato desejado.');
    $('#success-badge').classList.remove('hidden');
    setTimeout(() => progressPanel.classList.add('hidden'), 1200);
  } catch (error) {
    console.error(error);
    const message = error.name === 'EncodingError'
      ? 'O navegador não conseguiu decodificar este arquivo. Tente convertê-lo para WAV ou MP3.'
      : 'Não foi possível concluir a transcrição. Verifique a memória disponível e tente um arquivo menor.';
    showAlert(message);
    progressPanel.classList.add('hidden');
  } finally {
    isTranscribing = false;
    transcribeButton.disabled = false;
  }
}

function updateWordCount() {
  const words = transcript.value.trim() ? transcript.value.trim().split(/\s+/).length : 0;
  $('#word-count').textContent = `${words} ${words === 1 ? 'palavra' : 'palavras'}`;
}

function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function makeSrt() {
  if (segments.length) return segments.map((segment, index) => `${index + 1}\n${formatSrtTime(segment.start)} --> ${formatSrtTime(segment.end)}\n${segment.text}\n`).join('\n');
  return `1\n00:00:00,000 --> 00:00:10,000\n${transcript.value.trim() || 'Sem texto transcrito.'}\n`;
}

$('#select-button').addEventListener('click', (event) => { event.stopPropagation(); fileInput.click(); });
$('#replace-button').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (event) => handleFile(event.target.files[0]));
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') fileInput.click(); });
['dragenter', 'dragover'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.add('dragging'); }));
['dragleave', 'drop'].forEach((eventName) => dropZone.addEventListener(eventName, (event) => { event.preventDefault(); dropZone.classList.remove('dragging'); }));
dropZone.addEventListener('drop', (event) => handleFile(event.dataTransfer.files[0]));
transcribeButton.addEventListener('click', startTranscription);
transcript.addEventListener('input', updateWordCount);
$('#copy-button').addEventListener('click', async () => {
  if (!transcript.value) return showAlert('Ainda não há texto para copiar.');
  await navigator.clipboard.writeText(transcript.value);
  const label = $('#copy-button span');
  const previous = label.textContent;
  label.textContent = 'Copiado!';
  setTimeout(() => { label.textContent = previous; }, 1600);
});
$('#txt-button').addEventListener('click', () => downloadFile(transcript.value, `${selectedFile?.name.replace(/\.[^.]+$/, '') || 'transcricao'}.txt`, 'text/plain;charset=utf-8'));
$('#srt-button').addEventListener('click', () => downloadFile(makeSrt(), `${selectedFile?.name.replace(/\.[^.]+$/, '') || 'transcricao'}.srt`, 'application/x-subrip;charset=utf-8'));
$('#clear-button').addEventListener('click', resetApp);
$('#save-learning').addEventListener('click', saveLearning);
alertBox.querySelector('button').addEventListener('click', hideAlert);
$('#theme-toggle').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme === 'dark';
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  $('#theme-toggle').textContent = dark ? '☼' : '☾';
  $('#theme-toggle').setAttribute('aria-label', dark ? 'Ativar modo escuro' : 'Ativar modo claro');
});
