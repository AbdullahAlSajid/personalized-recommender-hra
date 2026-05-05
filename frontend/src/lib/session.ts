// Fetch a single text by ID
export async function getTextById(text_id: string) {
  const res = await fetch(`${BACKEND}/session/text/${text_id}`, {
    credentials: 'include',
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error('Kunne ikke hente teksten.');
  }
  return await res.json();
}

export type TextQuestionOption = {
  option_id: string;
  body: string;
  sanity_answer_key?: string | null;
  is_correct?: boolean | null;
  display_order?: number | null;
};

export type TextQuestion = {
  question_id: string;
  body: string;
  question_type: string;
  display_order?: number | null;
  options: TextQuestionOption[];
};

export async function getTextQuestionsById(text_id: string): Promise<TextQuestion[]> {
  const res = await fetch(`${BACKEND}/session/text/${text_id}/questions`, {
    credentials: 'include',
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error('Kunne ikke hente spørsmålene.');
  }
  const data = await res.json();
  return Array.isArray(data?.questions) ? (data.questions as TextQuestion[]) : [];
}

//const BACKEND = 'http://localhost:8000';
type ViteImportMeta = ImportMeta & {
  env: {
    VITE_BACKEND_URL: string;
  };
};
const BACKEND = (import.meta as ViteImportMeta).env.VITE_BACKEND_URL;

export const BACKEND_URL = BACKEND;

export type BroadTopic = {
  id: number;
  name: string;
};

export type SessionRecommendation = {
  text_id: string;
  title: string;
  broad_topics: string[];
  first_image_url?: string | null;
  preview_text?: string | null;
  final_difficulty: number;
  composite_score: number;
  score_topic: number;
  score_difficulty: number;
};


// The session_id lives exclusively in an HttpOnly cookie managed by the
// backend. JS never reads or stores it. All requests use credentials: 'include'
// so the browser sends the cookie automatically.

/**
 * Validate the shared access code. No DB write, no cookie.
 * Returns a one-time token to be passed to startSession() after consent.
 * The passcode never needs to be sent again after this call.
 */
export async function validatePasscode(passcode: string): Promise<string> {
  const res = await fetch(`${BACKEND}/sessions/validate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ passcode }),
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || 'Ugyldig kode. Prøv igjen.');
  }
  const data = await res.json();
  return data.token as string;
}

/**
 * Check whether the browser's cookie maps to an active session.
 * Returns false on any network error so the caller falls back gracefully.
 */
export async function checkSessionStatus(): Promise<boolean> {
  try {
    const res = await fetch(`${BACKEND}/sessions/status`, {
      credentials: 'include',
    });
    if (!res.ok) return false;
    const data = await res.json();
    return data.active === true;
  } catch {
    return false;
  }
}

/**
 * Create a DB session row and receive the HttpOnly cookie.
 * Passcode is validated server-side — /start rejects without a correct passcode.
 * Idempotent: backend returns existing session if cookie is still valid.
 * Only call after the user gives consent.
 */
/**
 * Consume the one-time token and create a DB session row.
 * The token is valid for 5 minutes and deleted on first use.
 * Sets the HttpOnly session cookie on success.
 */
export class TokenExpiredError extends Error {
  constructor() { super('token_expired'); }
}

export class SessionExpiredError extends Error {
  constructor() { super('session_expired'); }
}

function isSessionExpiry(status: number, detail: string): boolean {
  return (
    status === 401 ||
    (status === 400 && detail === 'Session already ended.') ||
    (status === 404 && detail === 'Session not found.')
  );
}

export async function startSession(token: string, consentGiven: boolean): Promise<void> {
  const res = await fetch(`${BACKEND}/sessions/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ token, consent_given: consentGiven }),
  });
  if (!res.ok) {
    if (res.status === 401) throw new TokenExpiredError();
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || 'Kunne ikke starte økt.');
  }
}

/**
 * Mark the session as ended and clear the cookie.
 * Idempotent. Safe to call from multiple tabs simultaneously.
 */
export async function endSession(): Promise<void> {
  try {
    await fetch(`${BACKEND}/sessions/end`, {
      method: 'POST',
      credentials: 'include',
    });
  } catch {
    // Swallow network errors.
  }
}

export async function getBroadTopics(): Promise<BroadTopic[]> {
  const res = await fetch(`${BACKEND}/session/topics`, {
    credentials: 'include',
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error(detail || 'Kunne ikke hente interesser.');
  }

  const data = await res.json();
  return Array.isArray(data?.topics) ? (data.topics as BroadTopic[]) : [];
}

export async function saveInterests(interests: string[]): Promise<void> {
  const res = await fetch(`${BACKEND}/session/interests`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ interests }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error(detail || 'Kunne ikke lagre interesser.');
  }
}

export async function getSessionRecommendations(): Promise<SessionRecommendation[]> {
  const res = await fetch(`${BACKEND}/session/recommendations`, {
    credentials: 'include',
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error(detail || 'Kunne ikke hente anbefalinger.');
  }

  const data = await res.json();
  return Array.isArray(data?.texts) ? (data.texts as SessionRecommendation[]) : [];
}

export async function refreshSessionRecommendations(
  shownTextIds: string[],
): Promise<SessionRecommendation[]> {
  const res = await fetch(`${BACKEND}/session/refresh`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ shown_text_ids: shownTextIds }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error(detail || 'Kunne ikke oppdatere anbefalingene.');
  }

  const data = await res.json();
  return Array.isArray(data?.texts) ? (data.texts as SessionRecommendation[]) : [];
}

export type SessionEventPayload = {
  event_type: string;
  slate_id?: number | null;
  text_id?: string | null;
  metadata?: Record<string, unknown> | null;
};

export async function logSessionEvent(payload: SessionEventPayload): Promise<void> {
  const res = await fetch(`${BACKEND}/session/events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error(detail || 'Kunne ikke logge hendelsen.');
  }
}


// ── Session feedback (end-of-session) ───────────────────

export type SessionFeedbackText = {
  text_id: string;
  title: string;
  first_image_url?: string | null;
};

export async function getSessionFeedbackTexts(): Promise<SessionFeedbackText[]> {
  const res = await fetch(`${BACKEND}/session/feedback/texts`, {
    credentials: 'include',
  });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error(detail || 'Kunne ikke hente økt-tekstene.');
  }
  const data = await res.json();
  return Array.isArray(data?.texts) ? (data.texts as SessionFeedbackText[]) : [];
}


// ── Persist answers ─────────────────────────────────────

export async function submitReadingAnswers(payload: {
  text_id: string;
  answers: Record<string, unknown>;
}): Promise<void> {
  const res = await fetch(`${BACKEND}/session/reading/submit`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error(detail || 'Kunne ikke lagre svarene.');
  }
}

export async function submitSessionFeedback(payload: {
  q1: number;
  q2: string;
  favorite_text_id?: string | null;
  favorite_why?: string | null;
}): Promise<void> {
  const res = await fetch(`${BACKEND}/session/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    const detail = (data.detail as string) || '';
    if (isSessionExpiry(res.status, detail)) throw new SessionExpiredError();
    throw new Error(detail || 'Kunne ikke lagre tilbakemeldingen.');
  }
}
