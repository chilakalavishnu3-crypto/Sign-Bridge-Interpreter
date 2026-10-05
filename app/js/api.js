/** HTTP helpers with timeouts and bounded retries for idempotent GETs. */

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function request(path, { method = "GET", body, timeout = 8000 } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeout);
  try {
    const res = await fetch(path, {
      method,
      signal: ctrl.signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch { /* non-JSON error */ }
      throw new ApiError(typeof detail === "string" ? detail : "Request failed", res.status);
    }
    return res;
  } catch (err) {
    if (err.name === "AbortError") throw new ApiError("Request timed out", 0);
    if (err instanceof ApiError) throw err;
    throw new ApiError("Server unreachable", 0);
  } finally {
    clearTimeout(timer);
  }
}

export async function getJSON(path, { retries = 2 } = {}) {
  for (let attempt = 0; ; attempt++) {
    try {
      return await (await request(path)).json();
    } catch (err) {
      const retryable = err.status === 0 || err.status >= 500;
      if (!retryable || attempt >= retries) throw err;
      await new Promise((r) => setTimeout(r, 400 * 2 ** attempt));
    }
  }
}

export async function getBlob(path, opts) {
  return (await request(path, opts)).blob();
}

export const api = {
  health: () => getJSON("/api/health", { retries: 0 }),
  vocabulary: () => getJSON("/api/vocabulary"),
  ttsUrl: (text, lang) => `/api/tts?${new URLSearchParams({ text, lang })}`,
};
