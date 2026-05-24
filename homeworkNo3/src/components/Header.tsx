/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { Sparkles, Activity, MessageSquareHeart, HeartHandshake, LogOut } from 'lucide-react';

interface HeaderProps {
  activeTab: 'delivery' | 'chat' | 'emotion';
  setActiveTab: (tab: 'delivery' | 'chat' | 'emotion') => void;
  user: { uid: string; username: string; nickname: string; mbti: string } | null;
  onLogout: () => void;
}

export function Header({ activeTab, setActiveTab, user, onLogout }: HeaderProps) {
  const todayStr = new Date().toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  });

  return (
    <header className="border-b border-white/10 bg-[#0E0E10]/95 backdrop-blur-md px-6 py-4 flex items-center justify-between sticky top-0 z-50">
      {/* Brand Logo */}
      <div className="flex items-center space-x-3">
        <div className="relative">
          <div className="absolute -inset-1 rounded-full bg-indigo-500/20 blur-sm animate-pulse"></div>
          <div className="relative w-8 h-8 rounded bg-indigo-600 flex items-center justify-center text-white border border-white/10 font-bold text-sm">
            星
          </div>
        </div>
        <div className="flex flex-col">
          <div className="flex items-center space-x-2">
            <span className="font-sans font-bold text-sm md:text-base tracking-tight text-slate-100">
              星愿树洞
            </span>
            {user && (
              <div className="hidden sm:flex items-center gap-1.5 px-2 py-0.5 bg-indigo-500/10 rounded text-[10px] text-indigo-300 border border-indigo-500/20">
                <span className="font-mono tracking-wider">{user.mbti} 星象</span>
              </div>
            )}
          </div>
          <span className="text-[9px] font-sans text-slate-500 tracking-wider">
            PROJECT: CANV_CELEST_SOUL • AES-256
          </span>
        </div>
      </div>

      {/* Navigation Tabs */}
      <nav id="nav-tabs" className="flex items-center space-x-1 p-1 rounded-lg bg-[#0A0A0B] border border-white/10">
        <button
          id="btn-tab-delivery"
          onClick={() => setActiveTab('delivery')}
          className={`flex items-center space-x-1.5 px-4 py-1.5 text-xs font-medium rounded transition-all duration-300 ${
            activeTab === 'delivery'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/10 font-semibold'
              : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
          }`}
        >
          <HeartHandshake size={13} />
          <span>投递心声</span>
        </button>

        <button
          id="btn-tab-chat"
          onClick={() => setActiveTab('chat')}
          className={`flex items-center space-x-1.5 px-4 py-1.5 text-xs font-medium rounded transition-all duration-300 ${
            activeTab === 'chat'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/10 font-semibold'
              : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
          }`}
        >
          <MessageSquareHeart size={13} />
          <span>智能回信</span>
        </button>

        <button
          id="btn-tab-emotion"
          onClick={() => setActiveTab('emotion')}
          className={`flex items-center space-x-1.5 px-4 py-1.5 text-xs font-medium rounded transition-all duration-300 ${
            activeTab === 'emotion'
              ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-500/10 font-semibold'
              : 'text-slate-400 hover:text-slate-100 hover:bg-white/5'
          }`}
        >
          <Activity size={13} />
          <span>情绪中心</span>
        </button>
      </nav>

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
