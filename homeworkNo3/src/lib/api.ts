/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

// Memory fallback storage for iframe sandbox environments without localStorage permission
const memoryStorage: Record<string, string> = {};
export const safeStorage = {
  getItem: (key: string): string | null => {
    try {
      return localStorage.getItem(key);
    } catch {
      return memoryStorage[key] || null;
    }
  },
  setItem: (key: string, value: string): void => {
    try {
      localStorage.setItem(key, value);
    } catch {
      memoryStorage[key] = value;
    }
  },
  removeItem: (key: string): void => {
    try {
      localStorage.removeItem(key);
    } catch {
      delete memoryStorage[key];
    }
  }
};

export async function secureFetch(input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
  let finalInit: RequestInit = init || {};
  try {
    const token = safeStorage.getItem('orion_token');
    if (token) {
      const headers = new Headers(finalInit.headers || {});
      if (!headers.has('Authorization')) {
        headers.set('Authorization', `Bearer ${token}`);
      }
      finalInit = { ...finalInit, headers };
    }
  } catch (err) {
    console.warn("Secure fetch setup failed:", err);
  }
  return window.fetch(input, finalInit);
}
