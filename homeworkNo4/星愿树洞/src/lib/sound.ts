/**
 * Celestial Sound Synthesizer
 * Uses Web Audio API to synthesize ambient soundscapes and chimes
 */

let audioCtx: AudioContext | null = null;
let isSoundEnabled = true;

const getAudioContext = () => {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
  }
  if (audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
  return audioCtx;
};

export const setSoundEnabled = (enabled: boolean) => {
  isSoundEnabled = enabled;
  localStorage.setItem('celestial-sounds', enabled ? 'true' : 'false');
};

export const getSoundEnabled = (): boolean => {
  const saved = localStorage.getItem('celestial-sounds');
  if (saved === null) return true;
  return saved === 'true';
};

// Beautiful high-frequency resonant celestial bell/chime
export const playCelestialChime = () => {
  if (!isSoundEnabled) return;
  try {
    const ctx = getAudioContext();
    const now = ctx.currentTime;

    // Harmonic chime uses multiple frequencies
    const baseFreq = 880; // A5 note
    const harmonics = [1, 1.5, 2, 2.5, 3]; // Beautiful major cord feel
    const volumes = [0.12, 0.06, 0.04, 0.02, 0.01];

    harmonics.forEach((h, i) => {
      const osc = ctx.createOscillator();
      const gainNode = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(baseFreq * h, now);
      
      // Slight pitch drift for that organic cosmic space aura
      osc.frequency.exponentialRampToValueAtTime(baseFreq * h * 0.995, now + 2);

      gainNode.gain.setValueAtTime(volumes[i], now);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 1.8 + i * 0.2);

      osc.connect(gainNode);
      gainNode.connect(ctx.destination);

      osc.start(now);
      osc.stop(now + 2.5);
    });

  } catch (err) {
    console.error("Audio Synthesis Error:", err);
  }
};

// Warm organic tick sound for switching tabs
export const playTabTick = () => {
  if (!isSoundEnabled) return;
  try {
    const ctx = getAudioContext();
    const now = ctx.currentTime;

    const osc = ctx.createOscillator();
    const gainNode = ctx.createGain();

    osc.type = 'triangle';
    osc.frequency.setValueAtTime(320, now);
    osc.frequency.exponentialRampToValueAtTime(150, now + 0.08);

    gainNode.gain.setValueAtTime(0.04, now);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 0.08);

    osc.connect(gainNode);
    gainNode.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 0.1);
  } catch (err) {
    // Ignore context blocked
  }
};

// Gentle soft wind/ocean sweep when user submits secrets
export const playCosmicSwell = () => {
  if (!isSoundEnabled) return;
  try {
    const ctx = getAudioContext();
    const now = ctx.currentTime;

    const osc = ctx.createOscillator();
    const gainNode = ctx.createGain();

    osc.type = 'sine';
    // Low frequency rumbling envelope
    osc.frequency.setValueAtTime(120, now);
    osc.frequency.exponentialRampToValueAtTime(220, now + 1.2);
    osc.frequency.exponentialRampToValueAtTime(60, now + 2.5);

    gainNode.gain.setValueAtTime(0.001, now);
    gainNode.gain.linearRampToValueAtTime(0.1, now + 1.0);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, now + 2.5);

    osc.connect(gainNode);
    gainNode.connect(ctx.destination);

    osc.start(now);
    osc.stop(now + 2.6);
  } catch (err) {
    // Ignore
  }
};
