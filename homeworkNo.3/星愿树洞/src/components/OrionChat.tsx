/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef } from 'react';
import { Phone, Info, Smile, Send, MessageSquareHeart, Trash2, ArrowLeft, Loader2 } from 'lucide-react';
import { CelestialLoading } from './CelestialLoading';
import { Wish, ChatMessage } from '../types';
import { motion, AnimatePresence } from 'motion/react';
import { secureFetch } from '../lib/api';

interface OrionChatProps {
  wishes: Wish[];
}

export function OrionChat({ wishes }: OrionChatProps) {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [selectedWishId, setSelectedWishId] = useState<string | null>(null);
  const [inputText, setInputText] = useState('');
  const [isSending, setIsSending] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [showPhoneModal, setShowPhoneModal] = useState(false);
  const [showInfoModal, setShowInfoModal] = useState(false);

  const chatListRef = useRef<HTMLDivElement>(null);

  // Fetch interactive chats on component load or selected wish change
  useEffect(() => {
    fetchChats();
  }, []);

  const fetchChats = async () => {
    setLoadingHistory(true);
    try {
      const resp = await secureFetch('/api/chat');
      if (resp.ok) {
        const data = await resp.json();
        setChatMessages(data);
      }
    } catch (err) {
      console.error("Error fetching chats:", err);
    } finally {
      setLoadingHistory(false);
    }
  };

  // Scroll to bottom helper directly on host container to avoid layout shifting
  useEffect(() => {
    if (chatListRef.current) {
      chatListRef.current.scrollTop = chatListRef.current.scrollHeight;
    }
  }, [chatMessages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim() || isSending) return;

    const messageToSend = inputText;
    setInputText('');
    setIsSending(true);

    // Dynamic locally appended User message
    const tempUserMsg: ChatMessage = {
      id: `temp-${Date.now()}`,
      sender: 'user',
      text: messageToSend,
      timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
    };
    setChatMessages(prev => [...prev, tempUserMsg]);

    try {
      const resp = await secureFetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageToSend })
      });

      if (resp.ok) {
        const data = await resp.json();
        setChatMessages(data.chats);
      } else {
        throw new Error();
      }
    } catch (err) {
      console.error(err);
      // Fallback Orion reply on connection problem
      const errorReply: ChatMessage = {
        id: `err-${Date.now()}`,
        sender: 'orion',
        text: '星空航道有些拥堵，Orion 的声音无法及时传来... 但请相信我一直都在星原的这一段，静静陪伴着你。',
        timestamp: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
      };
      setChatMessages(prev => [...prev, errorReply]);
    } finally {
      setIsSending(false);
    }
  };

  const handleClearChat = async () => {
    if (!window.confirm("确定要清空当前的心灵对话历史吗？（此操作不可逆）")) return;
    try {
      const resp = await secureFetch('/api/chat', { method: 'DELETE' });
      if (resp.ok) {
        const data = await resp.json();
        setChatMessages(data.chats);
      }
    } catch (err) {
      console.error("Error clearing chats:", err);
    }
  };

  // Automatically load wish contents in chat space if user selects a past wish
  const handleSelectWish = (wish: Wish) => {
    setSelectedWishId(wish.id);
    
    // Convert selected wish context into chat format to inject directly!
    const wishChatContext: ChatMessage[] = [
      {
        id: `wish-ctx-1-${wish.id}`,
        sender: 'user',
        text: `【我的倾诉】：\n${wish.content}`,
        timestamp: new Date(wish.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
      },
      {
        id: `wish-ctx-2-${wish.id}`,
        sender: 'orion',
        text: `【星愿回音】：\n${wish.orionReply || '收到你的心声。我在这里听着风吹过树梢，星轨在夜空舒展。我会替你保管这一份情绪。'}`,
        timestamp: new Date(wish.timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
      }
    ];

    setChatMessages(wishChatContext);
  };

  return (
    <div className="w-full max-w-6xl mx-auto flex flex-col gap-6 pt-6 px-4">
      {/* Title */}
      <div className="mb-2 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="font-sans font-bold text-2xl md:text-3xl tracking-tight text-white flex items-center space-x-2">
            <span>星愿回音与倾诉历史</span>
          </h1>
          <p className="text-xs md:text-sm text-slate-400 font-sans tracking-wide mt-1 leading-relaxed">
            回顾你投递的每一束情绪，感受来自 Orion 的真挚星空回音与温柔守护。
          </p>
        </div>
      </div>

      <div className="w-full h-[78vh] flex rounded-xl border border-white/10 bg-[#0E0E10] shadow-2xl overflow-hidden relative">
      
      {/* Sidebar: Past Wishes / Secrets */}
      <aside className="w-1/3 max-w-[280px] border-r border-white/10 bg-[#0E0E10] flex-col shrink-0 hidden md:flex">
        {/* Sidebar Header */}
        <div className="p-4 border-b border-white/10 bg-[#0A0A0B]/50">
          <h3 className="text-xs font-semibold text-slate-300 tracking-wider flex items-center space-x-1.5 uppercase">
            <MessageSquareHeart size={14} className="text-[#a49df5]" />
            <span>心愿倾诉历史</span>
          </h3>
          <p className="text-[10px] text-zinc-500 mt-1">
            点击倾诉条目即可同步载入对应密简
          </p>
        </div>

        {/* Sidebar List */}
        <div id="sidebar-wishes-list" className="flex-1 overflow-y-auto p-2 space-y-1.5 scrollbar-thin">
          {wishes.length === 0 ? (
            <div className="text-center py-12 text-zinc-600 space-y-2">
              <p className="text-xs italic">尚无倾诉信件</p>
              <p className="text-[10px]">去[投递心声]写第一笔吧</p>
            </div>
          ) : (
            wishes.map((w) => {
              const dateStr = new Date(w.timestamp).toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' });
              const isSelected = selectedWishId === w.id;
              
              return (
                <button
                  key={w.id}
                  onClick={() => handleSelectWish(w)}
                  id={`btn-sidebar-wish-${w.id}`}
                  className={`w-full text-left p-3 rounded-lg transition-all border duration-300 select-none cursor-pointer ${
                    isSelected
                      ? 'bg-indigo-600/10 border-indigo-500/40 text-white'
                      : 'border-transparent bg-transparent hover:bg-white/[0.02] text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-[10px] uppercase font-mono tracking-widest px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/15">
                      {w.emotion === 'anxious' && '🔥 焦虑'}
                      {w.emotion === 'melancholy' && '🌊 忧郁'}
                      {w.emotion === 'calm' && '🍃 平静'}
                      {w.emotion === 'joyful' && '☀️ 欣喜'}
                      {w.emotion === 'grateful' && '✨ 感恩'}
                    </span>
                    <span className="text-[10px] font-mono text-zinc-500 font-medium">
                      {dateStr}
                    </span>
                  </div>
                  <p className="text-xs line-clamp-2 leading-relaxed opacity-80">
                    {w.content}
                  </p>
                </button>
              );
            })
          )}
        </div>

        {/* Sidebar utility footer */}
        <div className="p-3 bg-[#0A0A0B] border-t border-white/10 text-center flex items-center justify-between">
          <button
            onClick={fetchChats}
            className="text-[10px] font-mono text-indigo-400 hover:text-indigo-300 transition"
          >
            获取对话原卷
          </button>
          <button
            onClick={handleClearChat}
            className="text-[10px] text-zinc-500 hover:text-rose-400 transition flex items-center space-x-1"
          >
            <Trash2 size={10} />
            <span>清空大厅</span>
          </button>
        </div>
      </aside>

      {/* Main Chat space */}
      <section className="flex-1 flex flex-col bg-[#121214] relative">
        {/* Header toolbar */}
        <div className="px-5 py-4 border-b border-white/10 flex items-center justify-between bg-[#0E0E10]">
          <div className="flex items-center space-x-3">
            {/* Back button for mobile widgets */}
            {selectedWishId && (
              <button 
                onClick={() => { setSelectedWishId(null); fetchChats(); }}
                className="md:hidden p-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white"
              >
                <ArrowLeft size={14} />
              </button>
            )}
            <div className="relative">
              <div className="h-3 w-3 rounded-full bg-emerald-500 ring-4 ring-emerald-500/10 animate-pulse absolute -right-0.5 -bottom-0.5" />
              <div className="w-8 h-8 rounded bg-[#1A1A1E] flex items-center justify-center border border-white/10 text-indigo-400 font-bold text-xs">
                O
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <span className="font-sans font-bold text-xs md:text-sm text-slate-100">
                  Orion <span className="font-normal text-slate-400">{selectedWishId ? '(专注特定私章)' : '(正在专注倾听)'}</span>
                </span>
              </div>
              <p className="text-[9px] text-zinc-500 font-mono tracking-wider">
                COMPASSIONATE AI SPIRIT LISTENER
              </p>
            </div>
          </div>

          {/* Quick info toolbar */}
          <div className="flex items-center space-x-2.5">
            <button
              onClick={() => setShowPhoneModal(true)}
              className="p-2 rounded border border-white/10 bg-[#1A1A1E] hover:bg-white/5 text-slate-400 hover:text-indigo-300 transition duration-300 cursor-pointer"
            >
              <Phone size={13} />
            </button>
            <button
              onClick={() => setShowInfoModal(true)}
              className="p-2 rounded border border-white/10 bg-[#1A1A1E] hover:bg-white/5 text-slate-400 hover:text-indigo-300 transition duration-300 cursor-pointer"
            >
              <Info size={13} />
            </button>
          </div>
        </div>

        {/* Conversation List area */}
        <div ref={chatListRef} className="flex-1 overflow-y-auto p-5 space-y-6 scrollbar-thin">
          {loadingHistory ? (
            <div className="flex flex-col items-center justify-center h-full">
              <CelestialLoading 
                label="RETRIEVING MEMORY SCROLLS..." 
                sublabel="正在穿梭于星系树洞，寻回记忆深处的往来信件记录..." 
                className="py-12"
              />
            </div>
          ) : (
            <>
              {chatMessages.map((msg) => {
                const isOrion = msg.sender === 'orion';
                return (
                  <div
                    key={msg.id}
                    className={`flex items-start max-w-[85%] space-x-3.5 ${
                      isOrion ? 'mr-auto text-left' : 'ml-auto flex-row-reverse text-right space-x-reverse'
                    }`}
                  >
                    {/* Character avatar */}
                    <div className={`w-8 h-8 rounded shrink-0 flex items-center justify-center font-bold border text-xs select-none ${
                      isOrion 
                        ? 'bg-[#1A1A1E] border-white/10 text-indigo-400' 
                        : 'bg-[#1A1A1E] border-white/10 text-violet-300'
                    }`}>
                      {isOrion ? 'O' : 'U'}
                    </div>

                    {/* Chat Bubble card */}
                    <div className="space-y-1.5 flex flex-col">
                      <div className={`px-4.5 py-3.5 rounded-lg text-xs md:text-sm leading-relaxed whitespace-pre-wrap ${
                        isOrion 
                          ? 'bg-[#0E0E10] text-slate-200 border border-white/10 font-sans shadow shadow-indigo-500/5'
                          : 'bg-indigo-600 text-white shadow shadow-indigo-600/10'
                      }`}>
                        {msg.text}
                      </div>
                      <span className="text-[9px] font-mono text-zinc-600 px-1 select-none">
                        {msg.timestamp}
                      </span>
                    </div>
                  </div>
                );
              })}
              
              {isSending && (
                <div className="flex items-start max-w-[85%] space-x-3.5 mr-auto">
                  <div className="w-8 h-8 rounded-full bg-slate-900 border border-indigo-500/10 flex items-center justify-center text-indigo-400 font-bold text-xs animate-pulse">
                    O
                  </div>
                  <div className="px-4 py-3 rounded-2xl bg-[#141525] border border-indigo-950/20 flex items-center space-x-1.5">
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '0ms' }} />
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '150ms' }} />
                    <span className="h-1.5 w-1.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '300ms' }} />
                  </div>
                </div>
              )}
            </>
          )}
        </div>

        {/* Input area */}
        <form
          onSubmit={handleSendMessage}
          className="p-4 bg-[#0E0E10] border-t border-white/10 flex items-center space-x-3"
        >
          <div className="flex-1 relative flex items-center rounded-lg bg-[#0A0A0B] border border-white/10 focus-within:border-indigo-500 transition">
            <button
              type="button"
              className="p-3 text-slate-500 hover:text-slate-300 transition shrink-0 cursor-pointer"
            >
              <Smile size={15} />
            </button>
            <input
              type="text"
              id="input-chat-message"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder={selectedWishId ? "在这章私语下继续探讨..." : "继续和 Orion 交流..."}
              disabled={isSending}
              autoComplete="off"
              className="flex-1 bg-transparent py-3.5 px-1.5 text-xs md:text-sm text-slate-100 placeholder-slate-600 focus:outline-none min-w-0"
            />
            <span className="hidden sm:inline font-mono text-[9px] text-zinc-600 px-4 select-none shrink-0 font-sans">
              {inputText.length}字
            </span>
          </div>

          <button
            type="submit"
            id="btn-send-chat"
            disabled={!inputText.trim() || isSending}
            className="p-3.5 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 hover:shadow-lg disabled:opacity-40 disabled:hover:bg-indigo-600 transition duration-300 flex items-center justify-center shrink-0 cursor-pointer"
          >
            <Send size={14} />
          </button>
        </form>

        {/* Phone Modal */}
        {showPhoneModal && (
          <div className="absolute inset-0 z-50 bg-[#0A0A0B]/90 backdrop-blur-sm flex items-center justify-center p-6 text-center animate-fade-in">
            <div className="p-6 rounded-lg border border-white/10 bg-[#0E0E10] max-w-sm space-y-4 shadow-2xl">
              <div className="w-12 h-12 rounded bg-indigo-600/10 border border-indigo-600/20 flex items-center justify-center text-indigo-400 mx-auto animate-bounce">
                <Phone size={20} />
              </div>
              <h3 className="text-sm font-bold text-white">正在发起心灵语音热线...</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Orion 的「极星密语」高保真全息音频通道目前正在架构中（Beta 2.0 专属）。音频合成需依赖服务端流式合成环境，目前暂支持纯文本纸笺传递。
              </p>
              <button
                onClick={() => setShowPhoneModal(false)}
                className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold cursor-pointer"
              >
                好的，继续文字交流
              </button>
            </div>
          </div>
        )}

        {/* Info Modal */}
        {showInfoModal && (
          <div className="absolute inset-0 z-50 bg-[#0A0A0B]/90 backdrop-blur-sm flex items-center justify-center p-6 text-center animate-fade-in">
            <div className="p-6 rounded-lg border border-white/10 bg-[#0E0E10] max-w-sm space-y-4 shadow-2xl">
              <div className="w-12 h-12 rounded bg-indigo-600/10 border border-indigo-600/20 flex items-center justify-center text-purple-400 mx-auto">
                <Info size={20} />
              </div>
              <h3 className="text-sm font-bold text-white">与 AI Orion 心灵对话的秘诀</h3>
              <p className="text-xs text-slate-400 leading-relaxed text-left space-y-2">
                • Orion 会根据您书写和流露出的具体情感予以不同视角的感同身受和共鸣。<br />
                • 随时能在左侧倾诉历史里阅读和重新分析历史笺，将历史笺作为深层语境探求。<br />
                • 所有对话全在内存及本地，绝对私密，做您随叫随到的温情知己。
              </p>
              <button
                onClick={() => setShowInfoModal(false)}
                className="w-full py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-semibold cursor-pointer"
              >
                明白了
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  </div>
);
}
