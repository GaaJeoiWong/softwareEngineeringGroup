/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState } from 'react';
import { Sparkles, Activity, MessageSquareHeart, HeartHandshake, LogOut, Volume2, VolumeX } from 'lucide-react';
import { getSoundEnabled, setSoundEnabled } from '../lib/sound';

interface HeaderProps {
  activeTab: 'delivery' | 'chat' | 'emotion';
  setActiveTab: (tab: 'delivery' | 'chat' | 'emotion') => void;
  user: { uid: string; username: string; nickname: string; mbti: string } | null;
  onLogout: () => void;
}

export function Header({ activeTab, setActiveTab, user, onLogout }: HeaderProps) {
  const [soundsOn, setSoundsOn] = useState(getSoundEnabled());

  const handleToggleSound = () => {
    const nextOn = !soundsOn;
    setSoundsOn(nextOn);
    setSoundEnabled(nextOn);
  };

  const todayStr = new Date().toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  return (
    <header className="fixed top-4 left-4 right-4 md:left-6 md:right-6 z-[100] border border-white/10 bg-[#0E0E10]/60 backdrop-blur-xl px-6 py-3.5 flex items-center justify-between rounded-xl md:rounded-full shadow-[0_8px_32px_0_rgba(0,0,0,0.4)] transition-all ease-out duration-300">
      {/* Brand Logo */}
      <div className="flex items-center space-x-3">
        <div className="relative">
          <div className="absolute -inset-1 rounded-full bg-indigo-500/20 blur-sm animate-pulse"></div>
          <div className="relative w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center text-white border border-white/10 font-bold text-sm">
            星
          </div>
        </div>
        <div className="flex flex-col">
          <div className="flex items-center space-x-2">
            <span className="font-sans font-bold text-sm tracking-tight text-slate-100">
              星愿树洞
            </span>
            {user && (
              <div className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 bg-indigo-500/10 rounded text-[10px] text-indigo-300 border border-indigo-500/20">
                <span className="font-mono tracking-wider">{user.mbti} 星象</span>
              </div>
            )}
          </div>
          <span className="text-[8px] font-sans text-slate-500 tracking-wider">
            PROJECT: CANV_CELEST_SOUL • AES-256
          </span>
        </div>
      </div>

      {/* Decorative center node in place of tab bar for branding balance */}
      <div className="hidden md:flex items-center gap-2 px-3 py-1 bg-white/5 border border-white/5 rounded-full text-[10px] text-indigo-200">
        <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse" />
        <span className="font-mono tracking-wider">CELESTIAL REALM CONNECTED</span>
      </div>

      {/* Date & User Info */}
      <div className="flex items-center space-x-4">
        {/* Date */}
        <div className="hidden lg:flex flex-col items-end text-right">
          <span className="text-[11px] font-medium text-slate-300 font-mono">
            {todayStr}
          </span>
          <span className="text-[9px] text-slate-500 font-mono">
            E2EE ACTIVE (AES)
          </span>
        </div>

        {/* User Badge */}
        {user && (
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2 bg-[#0A0A0B] border border-white/10 px-3 py-1 rounded-lg">
              <div className="relative">
                <span className="absolute bottom-0 right-0 block h-1.5 w-1.5 rounded-full bg-emerald-500 ring-1 ring-slate-950 animate-pulse" />
                <div className="w-5 h-5 rounded bg-indigo-950 flex items-center justify-center font-mono font-bold text-indigo-300 text-[10px] border border-indigo-500/20">
                  {user.nickname ? user.nickname.slice(0, 1).toUpperCase() : 'U'}
                </div>
              </div>
              <span className="text-[10px] text-slate-300 font-sans font-light hidden sm:inline max-w-[100px] truncate">
                {user.nickname}
              </span>
            </div>

            <button
              onClick={handleToggleSound}
              title={soundsOn ? "静音星原音效" : "开启星原音效"}
              className={`p-1.5 rounded border transition-colors cursor-pointer text-xs flex items-center justify-center ${
                soundsOn 
                  ? 'border-indigo-500/20 bg-indigo-500/5 text-indigo-300 hover:bg-indigo-500/10' 
                  : 'border-white/5 bg-transparent text-slate-500 hover:text-slate-300 hover:bg-white/5'
              }`}
            >
              {soundsOn ? <Volume2 size={13} className="animate-pulse" /> : <VolumeX size={13} />}
            </button>

            <button
              onClick={onLogout}
              title="退出登录离开树洞"
              className="p-1 px-2.5 rounded border border-white/5 bg-[#0E0E10] text-[#FF4466]/75 hover:text-rose-400 hover:bg-rose-500/5 transition-colors cursor-pointer text-xs flex items-center gap-1.5 font-light"
            >
              <LogOut size={12} />
              <span className="hidden md:inline text-[10px]">登出</span>
            </button>
          </div>
        )}
      </div>
    </header>
  );
}
