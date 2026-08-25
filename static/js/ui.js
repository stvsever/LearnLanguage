// Tiny DOM + UI toolkit: element builder, icon registry, toasts, modals.

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'html') node.innerHTML = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key === 'style' && typeof value === 'object') Object.assign(node.style, value);
    else if (key.startsWith('on') && typeof value === 'function') {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (value === true) node.setAttribute(key, '');
    else node.setAttribute(key, value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function esc(text) {
  const div = document.createElement('div');
  div.textContent = String(text ?? '');
  return div.innerHTML;
}

// -- Icons (Lucide-style strokes, inlined) -----------------------------------
const ICON_PATHS = {
  home: 'M3 10.5 12 3l9 7.5M5 9.5V21h14V9.5M9 21v-6h6v6',
  sparkles: 'M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9L12 3ZM19 16l.9 2.1L22 19l-2.1.9L19 22l-.9-2.1L16 19l2.1-.9L19 16Z',
  refresh: 'M21 12a9 9 0 1 1-2.64-6.36M21 3v6h-6',
  headphones: 'M4 14v4a2 2 0 0 0 2 2h1v-7H6a2 2 0 0 0-2 2Zm16 0a2 2 0 0 0-2-2h-1v7h1a2 2 0 0 0 2-2v-3ZM4 14v-2a8 8 0 0 1 16 0v2',
  mic: 'M12 3a3 3 0 0 1 3 3v5a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3Zm-7 8a7 7 0 0 0 14 0M12 18v3M9 21h6',
  book: 'M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2V5Zm2 14h13M8 7h6M8 11h5',
  chart: 'M4 20V10M10 20V4M16 20v-6M21 20H3',
  settings: 'M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6Zm8.4 3a8.5 8.5 0 0 0-.1-1.2l2-1.5-2-3.5-2.4 1a8.3 8.3 0 0 0-2-1.2L15.5 3h-4l-.4 2.6a8.3 8.3 0 0 0-2 1.2l-2.4-1-2 3.5 2 1.5a8.5 8.5 0 0 0 0 2.4l-2 1.5 2 3.5 2.4-1a8.3 8.3 0 0 0 2 1.2l.4 2.6h4l.4-2.6a8.3 8.3 0 0 0 2-1.2l2.4 1 2-3.5-2-1.5c.1-.4.1-.8.1-1.2Z',
  play: 'M7 5v14l12-7-12-7Z',
  pause: 'M7 5h4v14H7V5Zm6 0h4v14h-4V5Z',
  volume: 'M4 10v4h4l5 4V6l-5 4H4Zm12.5-1a5 5 0 0 1 0 6M19 6.5a9 9 0 0 1 0 11',
  turtle: 'M4 15h16M7 15v3M17 15v3M12 6a6 6 0 0 1 6 6v3H6v-3a6 6 0 0 1 6-6ZM12 6V4',
  check: 'M4 12.5 10 18 20 6',
  x: 'M5 5l14 14M19 5 5 19',
  plus: 'M12 5v14M5 12h14',
  arrowRight: 'M4 12h16m-6-6 6 6-6 6',
  arrowLeft: 'M20 12H4m6-6-6 6 6 6',
  flame: 'M12 3s5 4.5 5 9a5 5 0 0 1-10 0c0-1.5.5-3 1.5-4.5 0 0 .5 2 2 2.5C10 8 10.5 5 12 3Z',
  star: 'M12 3l2.7 5.6 6.3.9-4.5 4.4 1 6.1-5.5-2.9L6.5 20l1-6.1L3 9.5l6.3-.9L12 3Z',
  clock: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 4v5l3.5 2',
  download: 'M12 3v12m-5-5 5 5 5-5M4 21h16',
  upload: 'M12 15V3m-5 5 5-5 5 5M4 21h16',
  trash: 'M4 7h16M9 7V4h6v3m-8 0 1 13h8l1-13',
  eye: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Zm10 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z',
  ear: 'M6 10a6 6 0 1 1 12 0c0 3-2 4-3.5 5.5S13 19 12 21a3.5 3.5 0 0 1-3.5-3M9.5 10a2.5 2.5 0 0 1 5 0',
  keyboard: 'M3 7h18v10H3V7Zm3 3h.01M9.5 10h.01M13 10h.01M16.5 10h.01M6 13.5h.01M9.5 13.5h5M18 13.5h.01M18 10h.01',
  lightbulb: 'M9 18h6M10 21h4M12 3a6 6 0 0 1 4 10.5c-.8.7-1 1.5-1 2.5h-6c0-1-.2-1.8-1-2.5A6 6 0 0 1 12 3Z',
  target: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm0 4a5 5 0 1 0 0 10 5 5 0 0 0 0-10Zm0 4a1 1 0 1 0 0 2 1 1 0 0 0 0-2Z',
  globe: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-9 9h18M12 3c2.5 2.5 3.5 5.5 3.5 9S14.5 18.5 12 21c-2.5-2.5-3.5-5.5-3.5-9S9.5 5.5 12 3Z',
  help: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18Zm-2.5 6.5a2.5 2.5 0 1 1 3.8 2.1c-.8.5-1.3 1-1.3 1.9M12 17h.01',
  zap: 'M13 2 4 14h6l-1 8 9-12h-6l1-8Z',
  shuffle: 'M3 7h4l10 10h4m0 0-3-3m3 3-3 3M3 17h4l2.5-2.5M14 9.5 17 7h4m0 0-3-3m3 3-3 3',
  penLine: 'M4 20h16M4 20v-4L14.5 5.5a2.1 2.1 0 0 1 3 3L7 19l-3 1Z',
  layers: 'M12 3 3 8l9 5 9-5-9-5Zm-9 9 9 5 9-5m-18 4 9 5 9-5',
};

export function icon(name, size = 18) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('width', size);
  svg.setAttribute('height', size);
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.8');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  path.setAttribute('d', ICON_PATHS[name] || ICON_PATHS.help);
  svg.append(path);
  return svg;
}

// -- Toasts ------------------------------------------------------------------
let toastRoot = null;

export function toast(message, kind = 'info', timeout = 3600) {
  if (!toastRoot) {
    toastRoot = el('div', { class: 'toast-root', role: 'status', 'aria-live': 'polite' });
    document.body.append(toastRoot);
  }
  const item = el('div', { class: `toast toast-${kind}` },
    icon(kind === 'error' ? 'x' : kind === 'success' ? 'check' : 'lightbulb', 16),
    el('span', {}, message));
  toastRoot.append(item);
  requestAnimationFrame(() => item.classList.add('show'));
  setTimeout(() => {
    item.classList.remove('show');
    setTimeout(() => item.remove(), 300);
  }, timeout);
}

// -- Modal -------------------------------------------------------------------
export function openModal({ title, content, wide = false, onClose }) {
  const overlay = el('div', { class: 'modal-overlay' });
  const close = () => {
    overlay.classList.remove('show');
    setTimeout(() => overlay.remove(), 200);
    document.removeEventListener('keydown', onKey);
    onClose?.();
  };
  const onKey = (e) => { if (e.key === 'Escape') close(); };
  const panel = el('div', { class: `modal-panel${wide ? ' wide' : ''}`, role: 'dialog', 'aria-modal': 'true' },
    el('header', { class: 'modal-head' },
      el('h2', {}, title),
      el('button', { class: 'btn-icon', 'aria-label': 'Close', onclick: close }, icon('x', 18))),
    el('div', { class: 'modal-body' }, content));
  overlay.append(panel);
  overlay.addEventListener('mousedown', (e) => { if (e.target === overlay) close(); });
  document.addEventListener('keydown', onKey);
  document.body.append(overlay);
  requestAnimationFrame(() => overlay.classList.add('show'));
  return { close, panel };
}

export function confirmDialog(message, { danger = false, confirmLabel = 'Confirm' } = {}) {
  return new Promise((resolve) => {
    const content = el('div', {},
      el('p', { class: 'confirm-text' }, message),
      el('div', { class: 'row gap end', style: { marginTop: '20px' } },
        el('button', { class: 'btn btn-ghost', onclick: () => { modal.close(); resolve(false); } }, 'Cancel'),
        el('button', { class: `btn ${danger ? 'btn-danger' : 'btn-primary'}`, onclick: () => { modal.close(); resolve(true); } }, confirmLabel)));
    const modal = openModal({ title: 'Are you sure?', content, onClose: () => resolve(false) });
  });
}

// -- Formatting --------------------------------------------------------------
export function fmtInt(n) { return new Intl.NumberFormat('en-US').format(Math.round(n || 0)); }

export function fmtDuration(ms) {
  const minutes = Math.round((ms || 0) / 60000);
  if (minutes < 1) return '<1 min';
  if (minutes < 60) return `${minutes} min`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

export function fmtInterval(days) {
  if (days < 1 / 24 / 6) return '<10m';
  if (days < 1) return `${Math.max(1, Math.round(days * 24 * 60))}m`;
  if (days < 30) return `${Math.round(days)}d`;
  if (days < 365) return `${(days / 30.4).toFixed(1).replace('.0', '')}mo`;
  return `${(days / 365).toFixed(1).replace('.0', '')}y`;
}

// -- Staged progress (generation feedback) -----------------------------------
/**
 * Animated step checklist for long-running work. Steps advance on estimated
 * timers up to (but never past) the final step, which completes only when the
 * caller reports the real outcome via finish() or fail().
 *
 * progressSteps(['Contacting model', 'Writing', 'Validating'], [1500, 9000])
 */
export function progressSteps(labels, estimatesMs = []) {
  const rows = labels.map((label, index) => el('div', { class: `pstep${index === 0 ? ' active' : ''}` },
    el('span', { class: 'pstep-marker' },
      el('span', { class: 'pstep-spinner' }),
      icon('check', 12)),
    el('span', { class: 'pstep-label' }, label)));
  const root = el('div', { class: 'progress-steps', role: 'status', 'aria-live': 'polite' }, rows);

  let current = 0;
  const timers = [];
  const setState = (index, state) => {
    rows[index]?.classList.remove('active', 'done', 'failed');
    if (state) rows[index]?.classList.add(state);
  };
  const advance = () => {
    if (current >= labels.length - 1) return; // last step waits for reality
    setState(current, 'done');
    current += 1;
    setState(current, 'active');
  };
  estimatesMs.forEach((ms, i) => {
    timers.push(setTimeout(advance, estimatesMs.slice(0, i + 1).reduce((a, b) => a + b, 0)));
  });

  return {
    root,
    finish() {
      timers.forEach(clearTimeout);
      rows.forEach((_, i) => setState(i, 'done'));
      root.classList.add('all-done');
    },
    fail(message) {
      timers.forEach(clearTimeout);
      setState(current, 'failed');
      if (message) root.append(el('div', { class: 'pstep-error' }, message));
    },
    remove() {
      timers.forEach(clearTimeout);
      root.remove();
    },
  };
}

export function shuffled(array) {
  const copy = [...array];
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

export function sample(array, n) { return shuffled(array).slice(0, n); }
