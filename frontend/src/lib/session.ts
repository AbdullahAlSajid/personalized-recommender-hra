
//const BACKEND = 'http://localhost:8000';
const BACKEND = import.meta.env.VITE_BACKEND_URL as string;


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

export async function startSession(token: string): Promise<void> {
  const res = await fetch(`${BACKEND}/sessions/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    credentials: 'include',
    body: JSON.stringify({ token }),
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
