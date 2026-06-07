/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { Shield, Sparkles, User, Lock, ArrowRight, UserCheck } from 'lucide-react';
import { secureFetch } from '../lib/api';
import { CelestialParticles } from './CelestialParticles';
import { CustomCursor } from './CustomCursor';

interface AuthScreenProps {
  onAuthSuccess: (token: string, user: { uid: string; username: string; nickname: string; mbti: string }) => void;
}

export function AuthScreen({ onAuthSuccess }: AuthScreenProps) {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [nickname, setNickname] = useState('');
  const [mbti, setMbti] = useState('INFP'); // Mystical defaults
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const mbtiOptions = [
    'INFP', 'INFJ', 'ENFP', 'ENFJ', 
    'INTJ', 'INTP', 'ENTJ', 'ENTP',
    'ISFP', 'ISFJ', 'ESFP', 'ESFJ',
    'ISTP', 'ISTJ', 'ESTP', 'ESTJ'
  ];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    const finalUrl = isLogin ? '/api/auth/login' : '/api/auth/register';

    try {
      const resp = await secureFetch(finalUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username,
          password,
          nickname: nickname || undefined,
          mbti: mbti || undefined
        })
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "网络星核连接异常，请稍后再试");
      }

      // Success!
      onAuthSuccess(data.token, data.user);
    } catch (err: any) {
      setError(err.message || "请求失败，请检查填写内容");
    } finally {
      setLoading(false);
    }
  };

  // One-click secure anonymous access (Perfect for high privacy focus!)
  const handleAnonymousRegister = async () => {
    setError('');
    setLoading(true);

    // Generate highly random credentials automatically
    const anonId = Math.floor(Math.random() * 900000 + 100000);
    const anonUser = `anon_${anonId}`;
    const anonPass = `pass_${Math.random().toString(36).substring(2, 12)}_${Math.random().toString(36).substring(2, 12)}`;
    const anonNick = `星河旅人 #${anonId}`;
    const randomMbti = mbtiOptions[Math.floor(Math.random() * mbtiOptions.length)];

    try {
      const resp = await secureFetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: anonUser,
          password: anonPass,
          nickname: anonNick,
          mbti: randomMbti
        })
      });

      const data = await resp.json();
      if (!resp.ok) {
        throw new Error(data.error || "无法建立匿名连接");
      }

      onAuthSuccess(data.token, data.user);
    } catch (err: any) {
      setError(err.message || "匿名账户创建失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-slate-200 flex flex-col items-center justify-center font-sans relative overflow-hidden px-4">
      {/* High Inertia Magnetic Celestial Cursor */}
      <CustomCursor />

      {/* Interactive Celestial Particles Background */}
      <CelestialParticles intensity="medium" themeColor="indigo" />

      {/* Dynamic Star background */}
      <div 
        className="absolute inset-0 z-0 pointer-events-none opacity-40" 
        style={{
          backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.04) 1px, transparent 1px)',
          backgroundSize: '24px 24px'
        }}
      />
      <div className="absolute top-1/3 left-1/4 w-96 h-96 rounded-full bg-indigo-500/5 blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/3 right-1/4 w-96 h-96 rounded-full bg-violet-500/5 blur-3xl pointer-events-none" />

      <div className="w-full max-w-sm z-10">
        <div className="text-center mb-6">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-full border border-white/10 bg-white/5 backdrop-blur-md mb-3 text-indigo-400">
            <Sparkles className="h-6 w-6" id="auth-sparkles-icon" />
          </div>
          <h1 className="text-2xl font-light tracking-wider text-white">星愿树洞</h1>
          <p className="text-xs text-slate-500 mt-1.5 font-mono">SECURE & ANONYMOUS HEAVEN</p>
        </div>

        {/* Security Notice Card */}
        <div className="mb-5 p-3 rounded-lg border border-indigo-500/10 bg-indigo-500/5 text-slate-400 text-[11px] leading-relaxed flex items-start gap-2.5">
          <Shield className="h-4 w-4 text-indigo-400 shrink-0 mt-0.5" id="auth-shield-notice" />
          <div>
            <span className="text-slate-200 font-medium">极致隐私保护</span>：所有倾诉心声均在服务端经 AES-256 加密落库，且支持完全免除个人信息的匿名随机注册。
          </div>
        </div>

        <div className="bg-[#121215]/80 border border-white/10 rounded-xl p-5 shadow-2xl backdrop-blur-md">
          <div className="flex justify-around border-b border-white/5 pb-3 mb-4">
            <button 
              type="button" 
              onClick={() => { setIsLogin(true); setError(''); }}
              className={`text-sm tracking-wide transition-colors ${isLogin ? 'text-indigo-400 font-light border-b border-indigo-500 pb-1.5' : 'text-slate-400 hover:text-white pb-1.5'}`}
            >
              登入树洞
            </button>
            <button 
              type="button" 
              onClick={() => { setIsLogin(false); setError(''); }}
              className={`text-sm tracking-wide transition-colors ${!isLogin ? 'text-indigo-400 font-light border-b border-indigo-500 pb-1.5' : 'text-slate-400 hover:text-white pb-1.5'}`}
            >
              创建心洞
            </button>
          </div>

          {error && (
            <div className="mb-4 text-xs font-mono text-rose-400 bg-rose-500/5 border border-rose-500/10 px-3 py-2 rounded">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-[10px] text-slate-400 tracking-wider uppercase font-mono mb-1.5">倾听密匙 (账号)</label>
              <div className="relative">
                <span className="absolute left-3 top-2.5 text-slate-500">
                  <User className="h-4 w-4" id="auth-field-user" />
                </span>
                <input 
                  type="text" 
                  required
                  placeholder="输入你独特的匿名代号"
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  className="w-full bg-black/40 border border-white/5 rounded-lg pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 transition-all font-mono"
                />
              </div>
            </div>

            <div>
              <label className="block text-[10px] text-slate-400 tracking-wider uppercase font-mono mb-1.5">守护指纹 (密码)</label>
              <div className="relative">
                <span className="absolute left-3 top-2.5 text-slate-500">
                  <Lock className="h-4 w-4" id="auth-field-lock" />
                </span>
                <input 
                  type="password" 
                  required
                  placeholder="保护你心灵家园的密码"
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  className="w-full bg-black/40 border border-white/5 rounded-lg pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 transition-all font-mono"
                />
              </div>
            </div>

            {!isLogin && (
              <>
                <div>
                  <label className="block text-[10px] text-slate-400 tracking-wider uppercase font-mono mb-1.5">树洞匿昵 (昵称)</label>
                  <div className="relative">
                    <span className="absolute left-3 top-2.5 text-slate-500">
                      <UserCheck className="h-4 w-4" id="auth-field-nick" />
                    </span>
                    <input 
                      type="text" 
                      placeholder="选填：在树洞中显示的称呼"
                      value={nickname}
                      onChange={e => setNickname(e.target.value)}
                      className="w-full bg-black/40 border border-white/5 rounded-lg pl-9 pr-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500 transition-all"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-[10px] text-slate-400 tracking-wider uppercase font-mono mb-1.5">宇宙星盘特征 (MBTI)</label>
                  <select 
                    value={mbti}
                    onChange={e => setMbti(e.target.value)}
                    className="w-full bg-black/40 border border-white/5 rounded-lg px-3 py-2 text-xs text-indigo-300 focus:outline-none focus:border-indigo-500 transition-all font-mono appearance-none"
                    style={{
                      backgroundImage: 'linear-gradient(45deg, transparent 50%, #4f46e5 50%), linear-gradient(135deg, #4f46e5 50%, transparent 50%)',
                      backgroundPosition: 'calc(100% - 20px) calc(1em + 2px), calc(100% - 15px) calc(1em + 2px)',
                      backgroundSize: '5px 5px, 5px 5px',
                      backgroundRepeat: 'no-repeat'
                    }}
                  >
                    {mbtiOptions.map(opt => (
                      <option key={opt} value={opt} className="bg-[#121215] text-slate-300 font-mono">{opt}</option>
                    ))}
                  </select>
                </div>
              </>
            )}

            <button 
              type="submit"
              disabled={loading}
              className="w-full mt-2 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 disabled:opacity-50 text-white rounded-lg py-2 text-xs font-light tracking-wider transition-all flex items-center justify-center gap-1.5 cursor-pointer shadow-lg shadow-indigo-900/10"
            >
              {loading ? (
                <span className="h-3.5 w-3.5 rounded-full border border-white/30 border-t-white animate-spin"></span>
              ) : (
                <>
                  <span>{isLogin ? '点亮并进入树洞' : '签署心灵协议并注册'}</span>
                  <ArrowRight className="h-3.5 w-3.5 text-white/70" id="auth-submit-arrow" />
                </>
              )}
            </button>
          </form>

          {/* Prompt 1-Click Anonymous Option */}
          <div className="relative my-4 flex items-center justify-center">
            <span className="absolute w-full border-t border-white/5"></span>
            <span className="relative z-10 px-2 bg-[#121215] text-[10px] text-slate-500 font-mono">或</span>
          </div>

          <button
            type="button"
            disabled={loading}
            onClick={handleAnonymousRegister}
            className="w-full bg-slate-900/80 hover:bg-slate-800 disabled:opacity-50 border border-indigo-500/30 text-indigo-300 hover:text-indigo-200 rounded-lg py-2 text-xs font-light tracking-wide transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
          >
            <span>✨ 一键即安，免注匿名登录直达</span>
          </button>
        </div>

        <div className="text-center mt-6 text-[10px] text-slate-600 font-mono select-none">
          SECURE CHANNEL • PBKDF2 • STATE ISOLATED • AES-256
        </div>
      </div>
    </div>
  );
}
