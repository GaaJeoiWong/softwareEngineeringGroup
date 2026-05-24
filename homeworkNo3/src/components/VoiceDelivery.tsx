/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { ShieldCheck, MessageSquareHeart, Compass, HelpCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { EMOTIONS, EmotionType, Wish } from '../types';
import { motion, AnimatePresence } from 'motion/react';
import { secureFetch } from '../lib/api';

interface VoiceDeliveryProps {
  onWishDelivered: (wish: Wish) => void;
}

export function VoiceDelivery({ onWishDelivered }: VoiceDeliveryProps) {
  const [content, setContent] = useState('');
  const [selectedEmotion, setSelectedEmotion] = useState<EmotionType>('calm');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [apiIndicator, setApiIndicator] = useState<'working' | 'offline'>('working');

  // Sparkles animations parameters
  const [submissionProgress, setSubmissionProgress] = useState(false);
  const [lastOrionReply, setLastOrionReply] = useState<Wish | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!content.trim()) {
      setErrorMsg('请在纸笺上写下您此刻的心声流露。');
      return;
    }
    setErrorMsg(null);
    setIsSubmitting(true);
    setSubmissionProgress(true);

    try {
      const resp = await secureFetch('/api/wishes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content, emotion: selectedEmotion }),
      });

      if (!resp.ok) {
        throw new Error('伺服器开往星海的航道有些阻碍，请稍后重试。');
      }

      const data = await resp.json();
      const newWish: Wish = data.wish;
      
      // Delay slightly for the magical stellar submission animation
      setTimeout(() => {
        onWishDelivered(newWish);
        setLastOrionReply(newWish);
        setContent('');
        setIsSubmitting(false);
        setSubmissionProgress(false);
      }, 2500);

    } catch (err: any) {
      console.error(err);
      setErrorMsg(err.message || '网络连接遇到异常，请确认网络状态后重试。');
      setIsSubmitting(false);
      setSubmissionProgress(false);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto space-y-10 pt-6 px-4">
      
      {/* Visual Header */}
      <div className="text-center space-y-3">
        <h1 className="font-sans font-bold text-3xl md:text-4xl tracking-tight text-white/95">
          在这里，留下你的心声
        </h1>
        <p className="text-xs md:text-sm text-slate-400 font-sans tracking-wide max-w-lg mx-auto leading-relaxed">
          无论它是闪烁的愿望，还是沉重的秘密，星河都会悉心收藏。
        </p>
      </div>

      <AnimatePresence mode="wait">
        {!lastOrionReply ? (
          <motion.div
            key="input-form"
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -15 }}
            className="relative"
          >
            {/* Celestial Delivery Card */}
            <form 
              onSubmit={handleSubmit}
              className="relative p-6 md:p-8 rounded-xl border border-white/10 bg-[#0E0E10] shadow-2xl overflow-hidden"
            >
              {/* Submission Star dust overlay */}
              {submissionProgress && (
                <div className="absolute inset-0 z-20 bg-black/90 flex flex-col items-center justify-center space-y-4">
                  <div className="relative w-20 h-20">
                    <div className="absolute inset-0 rounded-full border border-indigo-500/10 animate-ping"></div>
                    <div className="absolute inset-2 rounded-full border border-indigo-400/20 animate-spin"></div>
                    <div className="absolute inset-4 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center">
                      <Compass className="text-amber-200 animate-spin-slow" size={24} />
                    </div>
                  </div>
                  <div className="text-center">
                    <p className="text-xs font-mono tracking-widest text-[#9d9da6] animate-pulse">
                      DELIVERING HEART STREAM TO GALAXY...
                    </p>
                    <p className="text-xs text-indigo-400 mt-1 font-sans">
                      信件正转化为星尘流光，正在飞往守护者 Orion...
                    </p>
                  </div>
                </div>
              )}

              {/* Textarea for feelings */}
              <div className="relative">
                <textarea
                  id="textarea-heart"
                  rows={6}
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  placeholder="此刻你在想什么？不论是日记、碎碎念、愿望还是不能对旁人倾吐的秘密，都可以在这里写下。Orion 将用最温柔的星语倾听你在夜晚的呼吸..."
                  disabled={isSubmitting}
                  className="w-full text-sm md:text-base leading-relaxed text-slate-200 bg-[#0A0A0B] border border-white/10 focus:border-indigo-500 hover:border-white/15 rounded-lg px-5 py-4 placeholder-slate-600 focus:outline-none transition-all resize-none shadow-inner"
                />
                <span className="absolute bottom-3 right-4 font-mono text-[10px] text-zinc-600">
                  {content.length} 字
                </span>
              </div>

              {/* Emotions selector */}
              <div className="mt-6 space-y-3">
                <label className="text-xs font-semibold text-slate-400 tracking-wider flex items-center space-x-1.5">
                  <span>选择当前情绪状态:</span>
                </label>
                <div id="emotion-buttons" className="flex flex-wrap gap-2.5">
                  {(Object.keys(EMOTIONS) as EmotionType[]).map((emoKey) => {
                    const emo = EMOTIONS[emoKey];
                    const isSelected = selectedEmotion === emoKey;
                    return (
                      <button
                        key={emoKey}
                        type="button"
                        id={`btn-emotion-${emoKey}`}
                        onClick={() => setSelectedEmotion(emoKey)}
                        disabled={isSubmitting}
                        className={`flex items-center space-x-1.5 px-3.5 py-1.5 rounded-lg text-xs font-medium border transition-all duration-300 transform active:scale-95 ${
                          isSelected
                            ? `${emo.color} scale-105 border-opacity-100 ring-1 ring-offset-2 ring-offset-[#0A0A0B] ring-indigo-500/20`
                            : 'text-zinc-500 border-white/5 hover:border-white/15 hover:text-zinc-400 bg-[#0A0A0B] bg-opacity-40'
                        }`}
                      >
                        <span className="text-sm">{emo.emoji}</span>
                        <span>{emo.label}</span>
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* Action Submit */}
              <div className="mt-8 flex flex-col sm:flex-row items-center justify-between gap-4 border-t border-white/10 pt-5">
                {/* Cloud and Error Status */}
                <div className="flex items-center space-x-2 text-xs text-slate-400">
                  {errorMsg ? (
                    <div className="flex items-center space-x-1.5 text-rose-400">
                      <AlertCircle size={14} className="shrink-0" />
                      <span>{errorMsg}</span>
                    </div>
                  ) : (
                    <div className="flex items-center space-x-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                      <span className="font-sans hover:text-indigo-400">云端服务正常运行中</span>
                    </div>
                  )}
                </div>

                <button
                  type="submit"
                  id="btn-deliver"
                  disabled={isSubmitting}
                  className="w-full sm:w-auto flex items-center justify-center space-x-2 px-8 py-2.5 rounded-lg text-xs font-semibold tracking-wider text-white bg-indigo-600 hover:bg-indigo-700 hover:shadow-lg transition-all duration-300 disabled:opacity-50 cursor-pointer"
                >
                  <MessageSquareHeart size={14} className="text-white" />
                  <span>投递到星河</span>
                </button>
              </div>
            </form>
          </motion.div>
        ) : (
          <motion.div
            key="orion-reply"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="p-6 md:p-8 rounded-xl border border-white/10 bg-[#0E0E10] shadow-2xl max-w-2xl mx-auto space-y-6"
          >
            {/* Header of Reply letter */}
            <div className="flex items-center justify-between border-b border-white/10 pb-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 rounded bg-[#1A1A1E] flex items-center justify-center text-indigo-400 font-bold border border-white/10 text-sm">
                  O
                </div>
                <div>
                  <h3 className="text-xs font-semibold text-slate-100 font-sans tracking-wide">
                    Orion (智慧守护者)
                  </h3>
                  <p className="text-[10px] text-indigo-400 font-mono">
                    守护信函已由星空封印
                  </p>
                </div>
              </div>
              <span className="text-[10px] font-mono p-1 rounded border border-emerald-500/20 bg-emerald-500/5 text-emerald-400 font-medium">
                ● 投递成功
              </span>
            </div>

            {/* Letter Envelope Contents */}
            <div className="space-y-4">
              <div className="bg-[#0A0A0B] rounded-lg p-4 border border-white/10 relative">
                <p className="text-[10px] text-zinc-500 uppercase font-mono tracking-widest absolute top-2 right-3">
                  YOUR SECRET
                </p>
                <p className="text-xs md:text-sm text-slate-400 italic line-clamp-3">
                  &quot;{lastOrionReply.content}&quot;
                </p>
              </div>

              <div className="space-y-3 pt-2">
                <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-widest">
                  Orion 的星愿回音:
                </h4>
                <p className="text-sm md:text-base leading-relaxed text-indigo-100/90 whitespace-pre-line font-sans pl-1.5 border-l-2 border-indigo-500/40">
                  {lastOrionReply.orionReply}
                </p>
              </div>
            </div>

            {/* Re-deliver buttons instructions */}
            <div className="flex flex-col sm:flex-row items-center gap-3 pt-4 border-t border-white/10 justify-end">
              <p className="text-[10px] text-zinc-500 text-center sm:text-right mr-auto">
                这封信笺已永久存档至您的[智能回信]历史库中。
              </p>
              
              <button
                type="button"
                id="btn-retry-deliver"
                onClick={() => setLastOrionReply(null)}
                className="w-full sm:w-auto px-5 py-2 rounded-lg text-xs font-medium text-slate-300 border border-white/10 hover:border-indigo-500 hover:text-white hover:bg-white/5 transition-all text-center cursor-pointer"
              >
                继续倾诉
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Feature Guarantee Cards block */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-6">
        <div className="p-5 rounded-lg border border-white/10 bg-[#0E0E10] flex flex-col space-y-2 hover:border-white/20 transition-all">
          <div className="w-8 h-8 rounded bg-[#1A1A1E] border border-[#ff007f]/10 flex items-center justify-center text-pink-400">
            <ShieldCheck size={16} />
          </div>
          <h3 className="text-xs font-semibold text-slate-200">绝对私密</h3>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            您的倾诉被完全安全密送，仅用户本地与AI倾听可见，尊重及守护每一份私隐与信任。
          </p>
        </div>

        <div className="p-5 rounded-lg border border-white/10 bg-[#0E0E10] flex flex-col space-y-2 hover:border-white/20 transition-all">
          <div className="w-8 h-8 rounded bg-[#1A1A1E] border border-indigo-500/10 flex items-center justify-center text-indigo-400">
            <MessageSquareHeart size={16} />
          </div>
          <h3 className="text-xs font-semibold text-slate-200">智能治愈</h3>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            基于人工智能强大的温情模型算法，给您真诚、充满文学温度的善意回复与理解共鸣。
          </p>
        </div>

        <div className="p-5 rounded-lg border border-white/10 bg-[#0E0E10] flex flex-col space-y-2 hover:border-white/20 transition-all">
          <div className="w-8 h-8 rounded bg-[#1A1A1E] border border-amber-500/10 flex items-center justify-center text-amber-400">
            <Compass size={16} />
          </div>
          <h3 className="text-xs font-semibold text-slate-200">时光胶囊</h3>
          <p className="text-[11px] text-slate-500 leading-relaxed">
            每一段心声都将被打上情绪之印，永久流存入情绪波谱谱图，可在历史中随时寻找痕迹。
          </p>
        </div>
      </div>
    </div>
  );
}
