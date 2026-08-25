// Shared runtime context, populated at boot by main.js.
// Views import this instead of importing main.js (avoids circular imports).

export const ctx = {
  config: null,            // /api/config payload
  navigate: () => {},      // (viewName, ...params) => void
  refreshChrome: () => {}, // re-render topbar/sidebar badges
  rerenderView: () => {},  // repaint the current view in place (e.g. after a
                           //   language change made from inside a modal)
};

export function languageProfile(code) {
  return ctx.config?.languages?.find((l) => l.code === code) || null;
}
