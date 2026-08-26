// CEFR levels for the interface.
//
// "A2" is a code, not a meaning. Everywhere a level is shown, it is shown with
// its plain-English name, so a learner can tell at a glance whether a topic is
// beneath them, at them, or a reach.
//
// The names come from the server (backend/levels.py) so prompts and interface
// agree, with the same table baked in here as a fallback for when /api/config
// could not be reached.

import { ctx } from './context.js';

export const LEVEL_ORDER = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2'];

const FALLBACK = {
  A1: { name: 'Beginner', blurb: 'First words and fixed phrases: greet, order, ask where things are.' },
  A2: { name: 'Elementary', blurb: 'Everyday exchanges: shopping, travel, describing your routine and your past.' },
  B1: { name: 'Intermediate', blurb: 'Opinions, plans, and stories across past, present, and future.' },
  B2: { name: 'Upper intermediate', blurb: 'Abstract subjects and real argument, with idiom and register under control.' },
  C1: { name: 'Advanced', blurb: 'Nuance, implication, and low-frequency vocabulary in complex syntax.' },
  C2: { name: 'Mastery', blurb: 'Native-like range: subtle register, literary and technical alike.' },
};

/** Every level as {code, name, blurb}, server-provided when available. */
export function allLevels() {
  const fromServer = ctx.config?.levels;
  if (Array.isArray(fromServer) && fromServer.length && typeof fromServer[0] === 'object') {
    return fromServer;
  }
  // Older payloads sent bare codes; keep working with either shape.
  const codes = Array.isArray(fromServer) && fromServer.length ? fromServer : LEVEL_ORDER;
  return codes.map((code) => ({ code, ...(FALLBACK[code] || { name: code, blurb: '' }) }));
}

export function normalizeLevel(code) {
  const raw = String(code || '').trim().toUpperCase();
  return LEVEL_ORDER.includes(raw) ? raw : 'A2';
}

export function levelIndex(code) {
  return LEVEL_ORDER.indexOf(normalizeLevel(code));
}

export function levelMeta(code) {
  const normalized = normalizeLevel(code);
  return allLevels().find((l) => l.code === normalized)
    || { code: normalized, ...FALLBACK[normalized] };
}

export function levelName(code) { return levelMeta(code).name; }
export function levelBlurb(code) { return levelMeta(code).blurb; }

/** "A2 Elementary", the form used wherever there is room for both. */
export function levelLabel(code) {
  const meta = levelMeta(code);
  return `${meta.code} ${meta.name}`;
}

export function levelStep(code, delta) {
  const index = levelIndex(code) + delta;
  return LEVEL_ORDER[Math.max(0, Math.min(LEVEL_ORDER.length - 1, index))];
}

/**
 * How a piece of content sits relative to the learner.
 *
 *   review   below their level, useful for consolidation
 *   at       their level, the default target
 *   stretch  one level up: the i+1 band, still worth doing
 *   ahead    two or more levels up: likely frustrating right now
 */
export function levelRelation(contentLevel, userLevel) {
  const delta = levelIndex(contentLevel) - levelIndex(userLevel);
  if (delta <= -1) return { key: 'review', label: 'Below your level', short: 'review' };
  if (delta === 0) return { key: 'at', label: 'At your level', short: 'your level' };
  if (delta === 1) return { key: 'stretch', label: 'One level up, a good stretch', short: 'stretch' };
  return { key: 'ahead', label: `${delta} levels above you`, short: 'ahead' };
}
