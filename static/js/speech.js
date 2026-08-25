// Speech recognition wrapper (Web Speech API) for pronunciation practice.
// Free, on-device/browser-provided; gracefully absent in unsupported browsers.

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;

export function speechSupported() {
  return Boolean(Recognition);
}

/**
 * Listen once and resolve with the best transcript.
 * Returns a controller: { stop(), promise }.
 */
export function listenOnce(locale, { onInterim } = {}) {
  if (!Recognition) {
    return { stop() {}, promise: Promise.reject(new Error('unsupported')) };
  }
  const recognition = new Recognition();
  recognition.lang = locale;
  recognition.interimResults = true;
  recognition.maxAlternatives = 3;
  recognition.continuous = false;

  let settled = false;
  let finalTranscript = '';

  const promise = new Promise((resolve, reject) => {
    recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) finalTranscript += result[0].transcript;
        else interim += result[0].transcript;
      }
      if (interim && onInterim) onInterim(interim);
    };
    recognition.onerror = (event) => {
      if (settled) return;
      settled = true;
      reject(new Error(event.error || 'recognition_error'));
    };
    recognition.onend = () => {
      if (settled) return;
      settled = true;
      if (finalTranscript.trim()) resolve(finalTranscript.trim());
      else reject(new Error('no_speech'));
    };
  });

  try {
    recognition.start();
  } catch (err) {
    return { stop() {}, promise: Promise.reject(err) };
  }
  return {
    stop() { try { recognition.stop(); } catch { /* already stopped */ } },
    promise,
  };
}
