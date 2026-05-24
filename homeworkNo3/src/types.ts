/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export type EmotionType = 'joyful' | 'melancholy' | 'calm' | 'anxious' | 'grateful';

export interface EmotionMeta {
  type: EmotionType;
  label: string;
  color: string;
  bgGlow: string;
  emoji: string;
}

export const EMOTIONS: Record<EmotionType, EmotionMeta> = {
  joyful: {
    type: 'joyful',
    label: '欣喜',
    color: 'text-amber-400 border-amber-500/30 bg-amber-500/10 hover:bg-amber-500/20 shadow-amber-500/5',
    bgGlow: 'from-amber-500/20 to-transparent',
    emoji: '☀️'
  },
  melancholy: {
    type: 'melancholy',
    label: '忧郁',
    color: 'text-sky-400 border-sky-500/30 bg-sky-500/10 hover:bg-sky-500/20 shadow-sky-500/5',
    bgGlow: 'from-sky-500/20 to-transparent',
    emoji: '🌊'
  },
  calm: {
    type: 'calm',
    label: '平静',
    color: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10 hover:bg-emerald-500/20 shadow-emerald-500/5',
    bgGlow: 'from-emerald-500/20 to-transparent',
    emoji: '🍃'
  },
  anxious: {
    type: 'anxious',
    label: '焦虑',
    color: 'text-rose-400 border-rose-500/30 bg-rose-500/10 hover:bg-rose-500/20 shadow-rose-500/5',
    bgGlow: 'from-rose-500/20 to-transparent',
    emoji: '🔥'
  },
  grateful: {
    type: 'grateful',
    label: '感恩',
    color: 'text-violet-400 border-violet-500/30 bg-violet-500/10 hover:bg-violet-500/20 shadow-violet-500/5',
    bgGlow: 'from-violet-500/20 to-transparent',
    emoji: '✨'
  }
};

export interface Wish {
  id: string;
  content: string;
  emotion: EmotionType;
  timestamp: string; // ISO string
  orionReply?: string;
  isRead?: boolean;
}

export interface TodoItem {
  id: string;
  text: string;
  completed: boolean;
  createdAt: string; // MM/DD
  completedAt?: string; // MM/DD
}

export interface ChatMessage {
  id: string;
  sender: 'user' | 'orion';
  text: string;
  timestamp: string;
}

export interface EmotionScore {
  date: string; // MM/DD
  value: number; // 0 - 100
  emotion: EmotionType;
}

export interface AnalyticsData {
  trend: EmotionScore[];
  daysCount: number;
  daysDiffPercent: number;
  totalWishes: number;
  targetWishes: number;
  avgEmotionScore: number;
  avgEmotionLabel: string;
  wordsCount: number;
}

export interface AchievementBadge {
  id: string;
  name: string;
  description: string;
  unlocked: boolean;
  iconName: string; // Lucide icon name mapping
}
