// Spaced-repetition scheduler based on FSRS-4.5 (Free Spaced Repetition Scheduler).
//
// Each card carries a memory state {S: stability in days, D: difficulty 1..10}.
// Retrievability decays as R(t) = (1 + t/(9S))^-1; the next interval is chosen
// so that R at review time equals the configured target retention (default 0.9)
// - the "desirable difficulty" dial of the whole system.
//
// New cards pass through short learning steps (1m, 10m) before graduating,
// and lapsed cards through a single 10m relearning step.

export const Rating = { AGAIN: 1, HARD: 2, GOOD: 3, EASY: 4 };

// FSRS-4.5 default parameters.
export const W = [
  0.4872, 1.4003, 3.7145, 13.8206, 5.1618, 1.2298, 0.8975, 0.031,
  1.6474, 0.1367, 1.0461, 2.1072, 0.0793, 0.3246, 1.587, 0.2272, 2.8755,
];

const LEARNING_STEPS_MIN = [1, 10]; // minutes
const RELEARNING_STEPS_MIN = [10];
const MIN_INTERVAL_DAYS = 1;
const MAX_INTERVAL_DAYS = 3650;
const DAY_MS = 86400000;

export function retrievability(elapsedDays, stability) {
  if (stability <= 0) return 0;
  return Math.pow(1 + elapsedDays / (9 * stability), -1);
}

export function intervalForRetention(stability, targetRetention) {
  const r = Math.min(0.99, Math.max(0.7, targetRetention || 0.9));
  return 9 * stability * (1 / r - 1);
}

function initStability(rating) {
  return Math.max(0.1, W[rating - 1]);
}

function initDifficulty(rating) {
  return clampD(W[4] - (rating - 3) * W[5]);
}

function clampD(d) { return Math.min(10, Math.max(1, d)); }

function nextDifficulty(d, rating) {
  const updated = d - W[6] * (rating - 3);
  return clampD(W[7] * initDifficulty(Rating.GOOD) + (1 - W[7]) * updated);
}

function nextRecallStability(d, s, r, rating) {
  const hardPenalty = rating === Rating.HARD ? W[15] : 1;
  const easyBonus = rating === Rating.EASY ? W[16] : 1;
  return s * (1 + Math.exp(W[8]) * (11 - d) * Math.pow(s, -W[9]) *
    (Math.exp(W[10] * (1 - r)) - 1) * hardPenalty * easyBonus);
}

function nextForgetStability(d, s, r) {
  const sf = W[11] * Math.pow(d, -W[12]) * (Math.pow(s + 1, W[13]) - 1) * Math.exp(W[14] * (1 - r));
  return Math.min(sf, s); // a lapse can never increase stability
}

function fuzz(intervalDays) {
  if (intervalDays < 2.5) return intervalDays;
  const spread = intervalDays * 0.05;
  return intervalDays + (Math.random() * 2 - 1) * spread;
}

export function newSrsState(now = Date.now()) {
  return {
    S: 0, D: 0,
    due: now,
    lastReview: null,
    reps: 0,
    lapses: 0,
    step: 0, // index within learning/relearning steps
  };
}

function elapsedDays(card, now) {
  if (!card.srs.lastReview) return 0;
  return Math.max(0, (now - card.srs.lastReview) / DAY_MS);
}

function clampInterval(days) {
  return Math.min(MAX_INTERVAL_DAYS, Math.max(MIN_INTERVAL_DAYS, days));
}

/** Preview the interval each rating would produce, for button labels. */
export function previewIntervals(card, now = Date.now(), targetRetention = 0.9) {
  const preview = {};
  for (const rating of [1, 2, 3, 4]) {
    const clone = {
      state: card.state,
      srs: { ...card.srs },
    };
    schedule(clone, rating, now, { targetRetention, applyFuzz: false });
    preview[rating] = (clone.srs.due - now) / DAY_MS;
  }
  return preview;
}

/**
 * Apply a rating to a card, mutating card.state and card.srs.
 * States: 'new' -> 'learning' -> 'review'; lapse: 'review' -> 'relearning' -> 'review'.
 */
export function schedule(card, rating, now = Date.now(), opts = {}) {
  const targetRetention = opts.targetRetention ?? 0.9;
  const applyFuzz = opts.applyFuzz !== false;
  const srs = card.srs;

  const graduate = (gradRating) => {
    if (!srs.S) {
      srs.S = initStability(gradRating);
      srs.D = initDifficulty(gradRating);
    }
    card.state = 'review';
    srs.step = 0;
    let interval = clampInterval(intervalForRetention(srs.S, targetRetention));
    if (applyFuzz) interval = clampInterval(fuzz(interval));
    srs.due = now + interval * DAY_MS;
  };

  if (card.state === 'new' || card.state === 'learning') {
    card.state = 'learning';
    if (rating === Rating.AGAIN) {
      srs.step = 0;
      srs.due = now + LEARNING_STEPS_MIN[0] * 60000;
    } else if (rating === Rating.EASY) {
      graduate(Rating.EASY);
    } else if (rating === Rating.HARD) {
      srs.due = now + LEARNING_STEPS_MIN[Math.min(srs.step, LEARNING_STEPS_MIN.length - 1)] * 60000;
    } else {
      srs.step += 1;
      if (srs.step >= LEARNING_STEPS_MIN.length) {
        graduate(Rating.GOOD);
      } else {
        srs.due = now + LEARNING_STEPS_MIN[srs.step] * 60000;
      }
    }
  } else if (card.state === 'relearning') {
    if (rating === Rating.AGAIN) {
      srs.step = 0;
      srs.due = now + RELEARNING_STEPS_MIN[0] * 60000;
    } else {
      card.state = 'review';
      srs.step = 0;
      let interval = clampInterval(intervalForRetention(srs.S, targetRetention));
      if (applyFuzz) interval = clampInterval(fuzz(interval));
      srs.due = now + interval * DAY_MS;
    }
  } else { // review
    const elapsed = elapsedDays(card, now);
    const r = retrievability(elapsed, srs.S);
    if (rating === Rating.AGAIN) {
      srs.lapses += 1;
      srs.S = Math.max(0.1, nextForgetStability(srs.D, srs.S, r));
      srs.D = nextDifficulty(srs.D, rating);
      card.state = 'relearning';
      srs.step = 0;
      srs.due = now + RELEARNING_STEPS_MIN[0] * 60000;
    } else {
      srs.S = nextRecallStability(srs.D, srs.S, r, rating);
      srs.D = nextDifficulty(srs.D, rating);
      let interval = clampInterval(intervalForRetention(srs.S, targetRetention));
      if (rating === Rating.HARD) interval = Math.max(MIN_INTERVAL_DAYS, interval * 0.8);
      if (applyFuzz) interval = clampInterval(fuzz(interval));
      srs.due = now + interval * DAY_MS;
    }
  }

  srs.reps += 1;
  srs.lastReview = now;
  return card;
}

/** A card counts as "mature" once its stability carries it three weeks or more. */
export function isMature(card) {
  return card.state === 'review' && card.srs.S >= 21;
}

export function isDue(card, now = Date.now()) {
  return card.state !== 'new' && !card.suspended && card.srs.due <= now;
}
