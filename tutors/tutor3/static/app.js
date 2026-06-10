const state = {
  config: null,
  topics: [],
  mode: "vocabulary",
  language: "es",
  voice: "",
  difficulty: "intermediate",
  practiceType: "choice",
  vocabulary: [],
  scenario: null,
  currentIndex: 0,
  questionIndex: 0,
  queue: [],
  notebook: [],
  stats: {
    seen: 0,
    correct: 0,
    attempts: 0,
    generated: 0,
    startedAt: Date.now(),
  },
  activePractice: null,
  audio: new Audio(),
  lastAudioText: "",
};

const els = {};

function $(id) {
  return document.getElementById(id);
}

function cacheElements() {
  [
    "languageStrip",
    "modelStatus",
    "saveSessionBtn",
    "conceptInput",
    "difficultySelect",
    "itemCountInput",
    "topicSelect",
    "voiceSelect",
    "rateSelect",
    "directionSelect",
    "englishToggle",
    "maskToggle",
    "autoQueueToggle",
    "generateVocabularyBtn",
    "topicVocabularyBtn",
    "generateScenarioBtn",
    "statusLine",
    "canvasKicker",
    "canvasTitle",
    "itemCounter",
    "primaryCard",
    "prevItemBtn",
    "nextItemBtn",
    "playCurrentBtn",
    "repeatCurrentBtn",
    "shadowBtn",
    "miniPlayBtn",
    "waveform",
    "audioTime",
    "learningNotes",
    "queueCount",
    "queueList",
    "practiceArea",
    "accuracyLabel",
    "cardsSeen",
    "correctCount",
    "newWords",
    "progressFill",
  ].forEach((id) => {
    els[id] = $(id);
  });
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function normalizeText(value) {
  return String(value ?? "")
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[¿?¡!.,;:()[\]{}"']/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

function shuffle(items) {
  const copy = [...items];
  for (let i = copy.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [copy[i], copy[j]] = [copy[j], copy[i]];
  }
  return copy;
}

function language() {
  return state.config.languages.find((item) => item.code === state.language) ?? state.config.languages[0];
}

function selectedVoice() {
  return state.voice || language().defaultVoice;
}

function setBusy(isBusy, message = "") {
  document.body.classList.toggle("loading", isBusy);
  if (message) {
    els.statusLine.textContent = message;
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || payload.error || "Request failed");
  }
  return payload;
}

function renderWaveform() {
  const bars = Array.from({ length: 54 }, (_, index) => {
    const height = 5 + ((index * 17) % 18);
    return `<span style="height:${height}px"></span>`;
  }).join("");
  els.waveform.innerHTML = bars;
}

function renderConfig() {
  els.modelStatus.textContent = state.config.hasOpenAIKey
    ? `Model ${state.config.model}`
    : "Demo mode, add OPENAI_API_KEY";

  els.languageStrip.innerHTML = state.config.languages
    .map((item) => {
      const active = item.code === state.language ? "active" : "";
      return `
        <button class="language-button ${active}" data-language="${item.code}" type="button">
          <span class="lang-code">${item.code.toUpperCase()}</span>
          <span><strong>${escapeHtml(item.display)}</strong><small>${escapeHtml(item.nativeName)}</small></span>
        </button>
      `;
    })
    .join("");

  els.difficultySelect.innerHTML = state.config.difficulties
    .map((item) => `<option value="${item.id}">${escapeHtml(item.label)} (${escapeHtml(item.cefr)})</option>`)
    .join("");
  els.difficultySelect.value = state.difficulty;

  renderVoiceOptions();
}

function renderVoiceOptions() {
  const lang = language();
  document.documentElement.style.setProperty("--target-font", lang.fontStack);
  els.voiceSelect.innerHTML = lang.voices
    .map((voice) => `<option value="${voice.id}">${escapeHtml(voice.label)}</option>`)
    .join("");
  state.voice = lang.defaultVoice;
  els.voiceSelect.value = state.voice;
}

function renderTopics() {
  els.topicSelect.innerHTML = state.topics
    .map((topic) => `<option value="${topic.id}">${escapeHtml(topic.nameEn)} (${topic.count})</option>`)
    .join("");
}

function modeLabel() {
  if (state.mode === "scenario") return "Scenario Lab";
  if (state.mode === "review") return "Review Queue";
  return "Vocabulary Studio";
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === mode);
  });
  render();
}

function currentVocabularyItem() {
  if (!state.vocabulary.length) return null;
  const safeIndex = Math.max(0, Math.min(state.currentIndex, state.vocabulary.length - 1));
  state.currentIndex = safeIndex;
  return state.vocabulary[safeIndex];
}

function currentScenarioQuestion() {
  if (!state.scenario?.items?.length) return null;
  const safeIndex = Math.max(0, Math.min(state.questionIndex, state.scenario.items.length - 1));
  state.questionIndex = safeIndex;
  return state.scenario.items[safeIndex];
}

function emptyState() {
  return `
    <div class="empty-state">
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path d="M12 3v18M4 8h16M6 16h12"></path>
        <path d="M8 5c0 4 8 4 8 0M8 19c0-4 8-4 8 0"></path>
      </svg>
      <h2>Generate a multilingual study set.</h2>
      <p>Use a custom prompt, a source vocabulary topic, or a scenario. Tutor 3 keeps English support, target script, audio, and retrieval practice in the same workflow.</p>
    </div>
  `;
}

function renderVocabCard() {
  const item = currentVocabularyItem();
  if (!item) {
    return emptyState();
  }
  const showEnglish = els.englishToggle.checked;
  const masked = els.maskToggle.checked ? "masked" : "";
  const tags = (item.tags || []).slice(0, 4).map((tag) => `<span class="tag">${escapeHtml(tag)}</span>`).join("");
  return `
    <div class="vocab-card">
      <div class="target-text ${masked}" style="font-family: var(--target-font)">${escapeHtml(item.target)}</div>
      ${item.pronunciation ? `<div class="native-subline">${escapeHtml(item.pronunciation)}</div>` : ""}
      ${showEnglish ? `<div class="english-line">${escapeHtml(item.english)}</div>` : ""}
      <div class="tag-row">${tags || `<span class="tag">${escapeHtml(language().scriptName)}</span>`}</div>
    </div>
  `;
}

function renderScenarioCard() {
  if (!state.scenario) {
    return emptyState();
  }
  const showEnglish = els.englishToggle.checked;
  const question = currentScenarioQuestion();
  const useEnglish = state.practiceType === "typing";
  const choices = useEnglish ? question.choices_en : question.choices;
  const prompt = useEnglish ? question.question_en : question.question;
  return `
    <div class="scenario-layout">
      <div class="scenario-title">
        <h2>${escapeHtml(state.scenario.scenario_title)}</h2>
        <span class="tag">${escapeHtml(state.scenario.difficulty)}</span>
      </div>
      <div class="passage">
        <p style="font-family: var(--target-font)">${escapeHtml(state.scenario.passage)}</p>
        ${showEnglish ? `<div class="translation-lines">${state.scenario.passage_en_lines.map((line) => `<span>${escapeHtml(line)}</span>`).join("")}</div>` : ""}
      </div>
      <div class="scenario-question">
        <h3>${escapeHtml(prompt)}</h3>
        <div class="choice-grid">
          ${choices
            .map((choice, index) => `<button class="choice-button" data-scenario-choice="${index}" type="button">${String.fromCharCode(65 + index)}. ${escapeHtml(choice)}</button>`)
            .join("")}
        </div>
      </div>
    </div>
  `;
}

function renderReviewCard() {
  if (!state.queue.length) {
    return `
      <div class="empty-state">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v7l4 2"></path><path d="M21 12a9 9 0 1 1-3-6.7"></path><path d="M21 4v5h-5"></path></svg>
        <h2>No review debt yet.</h2>
        <p>Generate vocabulary or scenarios, answer questions, then mark confidence. Hard items automatically stay near the top of the queue.</p>
      </div>
    `;
  }
  return `
    <div class="scenario-layout">
      <div class="scenario-title">
        <h2>Focused review queue</h2>
        <span class="tag">${state.queue.length} due</span>
      </div>
      <div class="choice-grid">
        ${state.queue
          .map((item, index) => `
            <button class="choice-button" data-jump="${item.index}" type="button">
              ${index + 1}. ${escapeHtml(item.target)} <span class="native-subline">${escapeHtml(item.english)}</span>
            </button>
          `)
          .join("")}
      </div>
    </div>
  `;
}

function buildQueue() {
  if (!els.autoQueueToggle.checked) return;
  state.queue = state.vocabulary.map((item, index) => ({
    ...item,
    index,
    difficulty: item.confidence ? Math.max(1, 6 - item.confidence) : 3,
  }));
}

function renderQueue() {
  els.queueCount.textContent = String(state.queue.length);
  if (!state.queue.length) {
    els.queueList.innerHTML = `<div class="queue-item"><span class="queue-number">0</span><div><strong>Queue is empty</strong><small>Hard items appear here after practice.</small></div></div>`;
    return;
  }
  els.queueList.innerHTML = state.queue
    .slice(0, 8)
    .map((item, index) => {
      const label = item.difficulty >= 4 ? "Hard" : item.difficulty >= 3 ? "Medium" : "Easy";
      return `
        <button class="queue-item" data-jump="${item.index}" type="button">
          <span class="queue-number">${index + 1}</span>
          <div><strong>${escapeHtml(item.target)}</strong><small>${escapeHtml(item.english)}</small></div>
          <span class="difficulty-pill">${label}</span>
        </button>
      `;
    })
    .join("");
}

function vocabularyPracticeQuestion() {
  const item = currentVocabularyItem();
  if (!item) return null;
  const directionSetting = els.directionSelect.value;
  const direction = directionSetting === "mixed"
    ? (Math.random() > 0.5 ? "en-to-target" : "target-to-en")
    : directionSetting;
  const ask = direction === "en-to-target" ? item.english : item.target;
  const answer = direction === "en-to-target" ? item.target : item.english;
  const pool = state.vocabulary
    .filter((candidate) => candidate !== item)
    .map((candidate) => (direction === "en-to-target" ? candidate.target : candidate.english));
  const choices = shuffle([answer, ...shuffle(pool).slice(0, 3)]).slice(0, 4);
  return { type: "vocabulary", ask, answer, choices, direction };
}

function scenarioPracticeQuestion() {
  const question = currentScenarioQuestion();
  if (!question) return null;
  const useEnglish = els.directionSelect.value === "target-to-en";
  const choices = useEnglish ? question.choices_en : question.choices;
  const ask = useEnglish ? question.question_en : question.question;
  return {
    type: "scenario",
    ask,
    answer: choices[question.correct_choice],
    choices,
    correctIndex: question.correct_choice,
    explanation: useEnglish ? question.reasoning_note_en : question.reasoning_note,
  };
}

function renderPractice() {
  const question = state.mode === "scenario" ? scenarioPracticeQuestion() : vocabularyPracticeQuestion();
  state.activePractice = question;
  if (!question) {
    els.practiceArea.innerHTML = `
      <div class="practice-prompt">
        <p>No active card.</p>
        <h3>Generate a set to start retrieval practice.</h3>
      </div>
    `;
    return;
  }

  if (state.practiceType === "typing") {
    els.practiceArea.innerHTML = `
      <div class="practice-prompt">
        <p>Type the answer from memory.</p>
        <h3>${escapeHtml(question.ask)}</h3>
        <div class="typing-row">
          <input id="typingAnswer" type="text" autocomplete="off" />
          <button class="primary-button" id="checkTypingBtn" type="button">Check</button>
        </div>
        <div class="feedback" id="practiceFeedback"></div>
      </div>
    `;
    return;
  }

  els.practiceArea.innerHTML = `
    <div class="practice-prompt">
      <p>Choose the best answer.</p>
      <h3>${escapeHtml(question.ask)}</h3>
      <div class="choice-grid">
        ${question.choices
          .map((choice, index) => `<button class="choice-button" data-choice="${index}" type="button">${String.fromCharCode(65 + index)}. ${escapeHtml(choice)}</button>`)
          .join("")}
      </div>
      <div class="feedback" id="practiceFeedback"></div>
    </div>
  `;
}

function updateCounters() {
  const total = state.mode === "scenario" ? (state.scenario?.items?.length || 0) : state.vocabulary.length;
  const index = state.mode === "scenario" ? state.questionIndex : state.currentIndex;
  els.itemCounter.textContent = total ? `${index + 1} / ${total}` : "0 / 0";
  els.cardsSeen.textContent = String(state.stats.seen);
  els.correctCount.textContent = String(state.stats.correct);
  els.newWords.textContent = String(state.vocabulary.length);
  const accuracy = state.stats.attempts ? Math.round((state.stats.correct / state.stats.attempts) * 100) : 0;
  els.accuracyLabel.textContent = `${accuracy}%`;
  const progress = total ? Math.round(((index + 1) / total) * 100) : 0;
  els.progressFill.style.width = `${progress}%`;
}

function renderNotes() {
  const current = currentVocabularyItem();
  const script = language().scriptSample;
  const note = current?.note || "Use hide target first, answer from memory, then reveal and play audio.";
  els.learningNotes.innerHTML = `
    <div class="note-card"><strong>Script guard</strong><span>${escapeHtml(script)}</span></div>
    <div class="note-card"><strong>Learning cue</strong><span>${escapeHtml(note)}</span></div>
    <div class="note-card"><strong>Daily loop</strong><span>Generate, listen, shadow, recall, rate confidence, repeat hard items.</span></div>
  `;
}

function render() {
  els.canvasKicker.textContent = modeLabel();
  if (state.mode === "scenario") {
    els.canvasTitle.textContent = state.scenario
      ? "Read the passage, infer details, and test comprehension."
      : "Generate compact scenarios with aligned English support.";
    els.primaryCard.innerHTML = renderScenarioCard();
  } else if (state.mode === "review") {
    els.canvasTitle.textContent = "Spend the next minutes on the items that need retrieval.";
    els.primaryCard.innerHTML = renderReviewCard();
  } else {
    els.canvasTitle.textContent = state.vocabulary.length
      ? "Active recall with script, sound, and meaning aligned."
      : "Daily recall, listening, and comprehension in one place.";
    els.primaryCard.innerHTML = renderVocabCard();
  }
  buildQueue();
  renderQueue();
  renderPractice();
  renderNotes();
  updateCounters();
}

function currentAudioText() {
  if (state.mode === "scenario" && state.scenario) {
    return state.scenario.passage;
  }
  const item = currentVocabularyItem();
  return item?.target || "";
}

async function playText(text) {
  if (!text.trim()) return;
  els.audioTime.textContent = "loading";
  const payload = await api("/api/tts", {
    method: "POST",
    body: {
      text,
      language: state.language,
      voice: selectedVoice(),
      rate: els.rateSelect.value,
    },
  });
  state.lastAudioText = text;
  state.audio.src = payload.url;
  try {
    await state.audio.play();
    els.audioTime.textContent = payload.durationSeconds ? `0:00 / ${formatTime(payload.durationSeconds)}` : "playing";
  } catch (error) {
    els.audioTime.textContent = payload.durationSeconds ? `ready / ${formatTime(payload.durationSeconds)}` : "ready";
    els.statusLine.textContent = "Audio is ready. Click play again if the browser blocked immediate playback.";
  }
}

function formatTime(seconds) {
  const safe = Math.max(0, Math.floor(seconds || 0));
  const minutes = Math.floor(safe / 60);
  return `${minutes}:${String(safe % 60).padStart(2, "0")}`;
}

function advance(delta) {
  if (state.mode === "scenario") {
    const total = state.scenario?.items?.length || 0;
    if (total) state.questionIndex = (state.questionIndex + delta + total) % total;
  } else {
    const total = state.vocabulary.length;
    if (total) state.currentIndex = (state.currentIndex + delta + total) % total;
  }
  render();
}

async function generateVocabulary(useTopic = false) {
  setBusy(true, useTopic ? "Loading topic vocabulary..." : "Generating vocabulary with AI...");
  try {
    const body = {
      language: state.language,
      count: Number(els.itemCountInput.value || 12),
      difficulty: els.difficultySelect.value,
    };
    const payload = useTopic
      ? await api("/api/vocabulary/topic", { method: "POST", body: { ...body, topicId: els.topicSelect.value } })
      : await api("/api/vocabulary/generate", { method: "POST", body: { ...body, concept: els.conceptInput.value.trim() } });
    state.vocabulary = payload.items || [];
    state.currentIndex = 0;
    state.mode = "vocabulary";
    state.stats.generated += state.vocabulary.length;
    els.statusLine.textContent = `${state.vocabulary.length} items loaded from ${payload.source}.`;
    setMode("vocabulary");
  } catch (error) {
    els.statusLine.textContent = `Vocabulary failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

async function generateScenario() {
  setBusy(true, "Generating a scenario and inference questions...");
  try {
    const payload = await api("/api/scenario/generate", {
      method: "POST",
      body: {
        topic: els.conceptInput.value.trim(),
        language: state.language,
        difficulty: els.difficultySelect.value,
      },
    });
    state.scenario = payload;
    state.questionIndex = 0;
    state.mode = "scenario";
    els.statusLine.textContent = `Scenario loaded from ${payload.source}.`;
    setMode("scenario");
  } catch (error) {
    els.statusLine.textContent = `Scenario failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

function markPractice(correct, detail = "") {
  state.stats.attempts += 1;
  state.stats.seen += 1;
  if (correct) state.stats.correct += 1;
  const feedback = $("practiceFeedback");
  if (feedback) {
    feedback.className = `feedback ${correct ? "good" : "bad"}`;
    feedback.textContent = detail || (correct ? "Correct. Move to the next item or rate confidence." : "Not yet. Listen once, reveal support, and try again.");
  }
  updateCounters();
}

function handleChoice(index) {
  const question = state.activePractice;
  if (!question) return;
  const selected = question.choices[index];
  const correct = question.type === "scenario" ? index === question.correctIndex : normalizeText(selected) === normalizeText(question.answer);
  document.querySelectorAll("[data-choice]").forEach((button) => {
    const buttonIndex = Number(button.dataset.choice);
    const buttonCorrect = question.type === "scenario"
      ? buttonIndex === question.correctIndex
      : normalizeText(question.choices[buttonIndex]) === normalizeText(question.answer);
    button.classList.toggle("correct", buttonCorrect);
    button.classList.toggle("wrong", buttonIndex === index && !buttonCorrect);
  });
  markPractice(correct, correct ? `Correct. ${question.explanation || ""}` : `Answer: ${question.answer}`);
}

function handleScenarioChoice(index) {
  if (!state.scenario) return;
  const question = currentScenarioQuestion();
  document.querySelectorAll("[data-scenario-choice]").forEach((button) => {
    const buttonIndex = Number(button.dataset.scenarioChoice);
    button.classList.toggle("correct", buttonIndex === question.correct_choice);
    button.classList.toggle("wrong", buttonIndex === index && index !== question.correct_choice);
  });
  markPractice(index === question.correct_choice, index === question.correct_choice ? question.reasoning_note_en : `Answer: ${question.choices_en[question.correct_choice]}`);
}

function handleTyping() {
  const question = state.activePractice;
  const input = $("typingAnswer");
  if (!question || !input) return;
  const correct = normalizeText(input.value) === normalizeText(question.answer);
  markPractice(correct, correct ? "Correct typed recall." : `Answer: ${question.answer}`);
}

function rateConfidence(score) {
  const item = currentVocabularyItem();
  if (item) {
    item.confidence = Number(score);
    if (score <= 3 && !state.queue.some((queued) => queued.index === state.currentIndex)) {
      state.queue.unshift({ ...item, index: state.currentIndex, difficulty: 6 - score });
    }
  }
  if (Number(score) >= 4) {
    advance(1);
  } else {
    render();
  }
}

async function saveSession() {
  setBusy(true, "Saving session locally...");
  try {
    const payload = await api("/api/session/save", {
      method: "POST",
      body: {
        language: state.language,
        difficulty: els.difficultySelect.value,
        stats: state.stats,
        vocabulary: state.vocabulary,
        scenario: state.scenario,
        queue: state.queue,
      },
    });
    els.statusLine.textContent = `Session saved as ${payload.file}.`;
  } catch (error) {
    els.statusLine.textContent = `Save failed: ${error.message}`;
  } finally {
    setBusy(false);
  }
}

function bindEvents() {
  els.languageStrip.addEventListener("click", (event) => {
    const button = event.target.closest("[data-language]");
    if (!button) return;
    state.language = button.dataset.language;
    renderConfig();
    render();
  });

  document.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  });

  document.querySelectorAll("[data-practice]").forEach((button) => {
    button.addEventListener("click", () => {
      state.practiceType = button.dataset.practice;
      document.querySelectorAll("[data-practice]").forEach((item) => item.classList.toggle("active", item === button));
      renderPractice();
    });
  });

  els.voiceSelect.addEventListener("change", () => {
    state.voice = els.voiceSelect.value;
  });
  els.difficultySelect.addEventListener("change", () => {
    state.difficulty = els.difficultySelect.value;
  });
  els.englishToggle.addEventListener("change", render);
  els.maskToggle.addEventListener("change", render);
  els.autoQueueToggle.addEventListener("change", render);
  els.directionSelect.addEventListener("change", renderPractice);
  els.generateVocabularyBtn.addEventListener("click", () => generateVocabulary(false));
  els.topicVocabularyBtn.addEventListener("click", () => generateVocabulary(true));
  els.generateScenarioBtn.addEventListener("click", generateScenario);
  els.prevItemBtn.addEventListener("click", () => advance(-1));
  els.nextItemBtn.addEventListener("click", () => advance(1));
  els.playCurrentBtn.addEventListener("click", () => playText(currentAudioText()));
  els.repeatCurrentBtn.addEventListener("click", () => playText(state.lastAudioText || currentAudioText()));
  els.miniPlayBtn.addEventListener("click", () => playText(currentAudioText()));
  els.shadowBtn.addEventListener("click", async () => {
    await playText(currentAudioText());
    els.statusLine.textContent = "Shadowing: listen once, repeat aloud, then rate confidence.";
  });
  els.saveSessionBtn.addEventListener("click", saveSession);

  els.practiceArea.addEventListener("click", (event) => {
    const choice = event.target.closest("[data-choice]");
    if (choice) handleChoice(Number(choice.dataset.choice));
    const check = event.target.closest("#checkTypingBtn");
    if (check) handleTyping();
  });

  els.primaryCard.addEventListener("click", (event) => {
    const scenarioChoice = event.target.closest("[data-scenario-choice]");
    if (scenarioChoice) handleScenarioChoice(Number(scenarioChoice.dataset.scenarioChoice));
    const jump = event.target.closest("[data-jump]");
    if (jump) {
      state.mode = "vocabulary";
      state.currentIndex = Number(jump.dataset.jump);
      setMode("vocabulary");
    }
  });

  els.queueList.addEventListener("click", (event) => {
    const jump = event.target.closest("[data-jump]");
    if (!jump) return;
    state.currentIndex = Number(jump.dataset.jump);
    state.mode = "vocabulary";
    setMode("vocabulary");
  });

  document.querySelector(".confidence-grid").addEventListener("click", (event) => {
    const button = event.target.closest("[data-confidence]");
    if (button) rateConfidence(Number(button.dataset.confidence));
  });

  state.audio.addEventListener("timeupdate", () => {
    const current = formatTime(state.audio.currentTime);
    const duration = Number.isFinite(state.audio.duration) ? ` / ${formatTime(state.audio.duration)}` : "";
    els.audioTime.textContent = `${current}${duration}`;
  });
}

async function init() {
  cacheElements();
  renderWaveform();
  const [config, topicsPayload] = await Promise.all([
    api("/api/config"),
    api("/api/topics"),
  ]);
  state.config = config;
  state.topics = topicsPayload.topics || [];
  renderConfig();
  renderTopics();
  bindEvents();
  render();
}

init().catch((error) => {
  console.error(error);
  document.body.innerHTML = `<main class="empty-state"><h2>Tutor 3 could not start.</h2><p>${escapeHtml(error.message)}</p></main>`;
});
