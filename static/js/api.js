// Thin fetch layer over the local JSON API with timeouts and typed errors.
// Generation calls automatically carry the user's model override from Settings.

import { state } from './store.js';

export class ApiError extends Error {
  constructor(status, code, detail) {
    super(detail || code || `HTTP ${status}`);
    this.status = status;
    this.code = code;
  }
}

async function request(path, { method = 'GET', body, timeout = 90000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, {
      method,
      headers: body ? { 'Content-Type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new ApiError(response.status, data.error || 'request_failed', data.detail || response.statusText);
    }
    return data;
  } catch (err) {
    if (err.name === 'AbortError') throw new ApiError(0, 'timeout', 'The request timed out.');
    if (err instanceof ApiError) throw err;
    throw new ApiError(0, 'network', 'Could not reach the local server.');
  } finally {
    clearTimeout(timer);
  }
}

function withModel(payload) {
  return state.settings.model ? { model: state.settings.model, ...payload } : payload;
}

export const api = {
  config: () => request('/api/config'),
  grammar: (language) => request(`/api/grammar?language=${encodeURIComponent(language)}`),
  lesson: (payload) => request('/api/lesson', { method: 'POST', body: withModel(payload), timeout: 180000 }),
  compose: (payload) => request('/api/compose', { method: 'POST', body: withModel(payload), timeout: 300000 }),
  gloss: (payload) => request('/api/gloss', { method: 'POST', body: withModel(payload), timeout: 90000 }),
  tts: (payload) => request('/api/tts', { method: 'POST', body: payload, timeout: 45000 }),
};
