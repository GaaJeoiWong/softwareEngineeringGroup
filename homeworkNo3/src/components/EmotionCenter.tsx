/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { 
  Plus, Check, Trash2, Trophy, Eye, ShieldAlert, BadgeInfo, Calendar, 
  Send, Moon, Sparkles, Leaf, PenTool, Heart, Navigation, Compass 
} from 'lucide-react';
import { TodoItem, AchievementBadge, AnalyticsData } from '../types';
import { AnimatePresence } from 'motion/react';
import { secureFetch } from '../lib/api';

// Dynamic Lucide mapping for Badge icons
const BADGE_ICONS: Record<string, React.ComponentType<any>> = {
  Send: Send,
  Moon: Moon,
  Sparkles: Sparkles,
  Leaf: Leaf,
  PenTool: PenTool,
  Heart: Heart,
  Navigation: Navigation,
  Compass: Compass,
};

interface EmotionCenterProps {
  onStateUpdate: () => void;
}

export function EmotionCenter({ onStateUpdate }: EmotionCenterProps) {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [badges, setBadges] = useState<AchievementBadge[]>([]);
  const [todos, setTodos] = useState<TodoItem[]>([]);
  const [newTodoText, setNewTodoText] = useState('');
  const [loading, setLoading] = useState(true);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [activeDateTab, setActiveDateTab] = useState<'week' | 'month'>('week');
  const [hoveredTrendPoint, setHoveredTrendPoint] = useState<any | null>(null);
  const [showResetModal, setShowResetModal] = useState(false);
  const [resetSuccess, setResetSuccess] = useState(false);

  const handleResetAll = async () => {
    try {
      const resp = await secureFetch('/api/profile/reset', { method: 'POST' });
      if (resp.ok) {
        setResetSuccess(true);
        setTimeout(() => {
          setResetSuccess(false);
          setShowResetModal(false);
        }, 1500);
        fetchCoreData();
        onStateUpdate(); // trigger global states reload
      }
    } catch (err) {
      console.error("Error resetting profile:", err);
    }
  };

  useEffect(() => {
    fetchCoreData();
  }, []);

  const fetchCoreData = async () => {
    setLoading(true);
    try {
      const analyticsResp = await secureFetch('/api/analytics');
      const todosResp = await secureFetch('/api/todos');
      
      if (analyticsResp.ok && todosResp.ok) {
        const analyticsData = await analyticsResp.json();
        const todosData = await todosResp.json();
        
        setAnalytics(analyticsData.analytics);
        setBadges(analyticsData.badges);
        setTodos(todosData);
      } else {
        throw new Error('无法连接情绪分析核心。');
      }
    } catch (err) {
      console.error(err);
      setErrorMsg('同步情绪档案受阻，请检查网络后尝试重连。');
    } finally {
      setLoading(false);
    }
  };

  const handleAddTodo = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTodoText.trim()) return;

    try {
      const resp = await secureFetch('/api/todos', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: newTodoText })
      });
      if (resp.ok) {
         setNewTodoText('');
         fetchCoreData();
         onStateUpdate(); // trigger global states reload
      }
    } catch (err) {
      console.error("Error adding todo:", err);
    }
  };

  const handleToggleTodo = async (id: string, currentCompleted: boolean) => {
    try {
      const resp = await secureFetch(`/api/todos/${id}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ completed: !currentCompleted })
      });
      if (resp.ok) {
        fetchCoreData();
        onStateUpdate();
      }
    } catch (err) {
      console.error("Error toggling todo:", err);
    }
  };

  const handleDeleteTodo = async (id: string) => {
    try {
      const resp = await secureFetch(`/api/todos/${id}`, {
        method: 'DELETE'
      });
      if (resp.ok) {
        fetchCoreData();
        onStateUpdate();
      }
    } catch (err) {
      console.error("Error deleting todo:", err);
    }
  };

  // Helper to build a responsive, gorgeous SVG Wave Chart
  const renderWaveChart = () => {
    if (!analytics || !analytics.trend || analytics.trend.length === 0) {
      return (
        <div className="relative w-full h-[180px] flex flex-col items-center justify-center border border-white/5 bg-[#0A0A0B]/60 rounded-xl p-6 text-center select-none">
          <Moon className="text-indigo-400 animate-pulse mb-2" size={24} />
          <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest flex items-center gap-1.5 justify-center">
            <span>暂无心境心语轨迹</span>
            <span className="text-[9px] font-mono text-zinc-600 lowercase font-normal">NO EMOTIONAL TREND</span>
          </h4>
          <p className="text-[10px] text-slate-500 mt-1.5 max-w-xs leading-normal">
            去“投递心声”倾诉心愿，或在“智能回信”与 Orion 精密互动，情绪就会交织绘制出静谧的心灵曲线。
          </p>
        </div>
      );
    }

    const trend = analytics.trend;
    const width = 500;
    const height = 180;
    const padding = 25;

    // Scale helpers
    const xScale = (index: number) => {
      return padding + (index * (width - padding * 2)) / (trend.length - 1);
    };

    const yScale = (value: number) => {
      // value is 0 - 100, scale to height minus padding
      return height - padding - (value * (height - padding * 2)) / 100;
    };

    // Build SVG Path points for line and area under line
    let points = trend.map((point, index) => {
      return { x: xScale(index), y: yScale(point.value), raw: point };
    });

    // Make smooth curve path with cubic bezier if possible, or simple smooth line
    // Here we'll build a smooth bezier wave representing the quiet calm galactic line!
    let pathD = '';
    let areaD = '';

    if (points.length > 0) {
      // Start path
      pathD = `M ${points[0].x} ${points[0].y}`;
      areaD = `M ${points[0].x} ${height - padding} L ${points[0].x} ${points[0].y}`;

      for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[i];
        const p1 = points[i + 1];
        // Control points
        const cpX1 = p0.x + (p1.x - p0.x) / 2;
        const cpY1 = p0.y;
        const cpX2 = p0.x + (p1.x - p0.x) / 2;
        const cpY2 = p1.y;

        pathD += ` C ${cpX1} ${cpY1}, ${cpX2} ${cpY2}, ${p1.x} ${p1.y}`;
        areaD += ` C ${cpX1} ${cpY1}, ${cpX2} ${cpY2}, ${p1.x} ${p1.y}`;
      }

      areaD += ` L ${points[points.length - 1].x} ${height - padding} Z`;
    }

    return (
      <div className="relative w-full h-auto">
        {/* Date Tabs inside chart Header */}
        <div className="flex justify-between items-center mb-4">
          <div className="flex flex-col">
            <h4 className="text-[11px] font-semibold text-slate-400 uppercase tracking-widest flex items-center space-x-1">
              <span>情绪指数波动 (7天)</span>
              <span className="text-[9px] font-mono text-indigo-400 lowercase font-normal">EMOTIONAL TREND ANALYSIS</span>
            </h4>
          </div>
          <div className="flex items-center space-x-1.5 p-0.5 rounded-md bg-slate-950/60 border border-indigo-950/50">
            <button 
              onClick={() => setActiveDateTab('week')}
              className={`text-[9px] px-2.5 py-1 rounded font-medium transition ${activeDateTab === 'week' ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/20' : 'text-zinc-500 hover:text-zinc-300'}`}
            >
              周
            </button>
            <button 
              onClick={() => setActiveDateTab('month')}
              className={`text-[9px] px-2.5 py-1 rounded font-medium transition ${activeDateTab === 'month' ? 'bg-indigo-600/30 text-indigo-300 border border-indigo-500/20' : 'text-zinc-500 hover:text-zinc-300'}`}
            >
              月
            </button>
          </div>
        </div>

        {/* Real Chart SVG */}
        <div id="wrapper-chart-svg" className="w-full relative bg-[#070912]/45 rounded-xl border border-indigo-950/20 p-2 overflow-hidden shadow-inner">
          <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto overflow-visible select-none">
            {/* Definitions for Gradients */}
            <defs>
              <linearGradient id="chartGlow" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#4338ca" stopOpacity="0.45" />
                <stop offset="50%" stopColor="#6366f1" stopOpacity="0.15" />
                <stop offset="100%" stopColor="#818cf8" stopOpacity="0" />
              </linearGradient>
            </defs>

            {/* Grid references */}
            <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} stroke="#1e1b4b" strokeWidth="0.8" strokeDasharray="3 3" />
            <line x1={padding} y1={yScale(50)} x2={width - padding} y2={yScale(50)} stroke="#1e1b4b" strokeWidth="0.6" strokeDasharray="3 3" />
            <line x1={padding} y1={yScale(100)} x2={width - padding} y2={yScale(100)} stroke="#1e1b4b" strokeWidth="0.8" strokeDasharray="3 3" />

            {/* Left reference Y legends */}
            <text x={padding - 5} y={yScale(100) + 3} fill="#4b5563" fontSize="7" fontFamily="monospace" textAnchor="end">100</text>
            <text x={padding - 5} y={yScale(50) + 3} fill="#4b5563" fontSize="7" fontFamily="monospace" textAnchor="end">50</text>
            <text x={padding - 5} y={yScale(0) + 3} fill="#4b5563" fontSize="7" fontFamily="monospace" textAnchor="end">0</text>

            <AnimatePresence>
              {/* Highlight Gradient Area Fill */}
              {areaD && (
                <path d={areaD} fill="url(#chartGlow)" className="transition-all duration-700 ease-out" />
              )}

              {/* Glowing Wave Line */}
              {pathD && (
                <path d={pathD} fill="none" stroke="#6366f1" strokeWidth="2.5" strokeLinecap="round" className="transition-all duration-700 ease-out" />
              )}

              {/* Interactive Points circles */}
              {points.map((p, i) => {
                const isHovered = hoveredTrendPoint?.index === i;
                const emojiMap: Record<string, string> = {
                  joyful: '☀️',
                  grateful: '✨',
                  calm: '🍃',
                  melancholy: '🌊',
                  anxious: '🔥'
                };
                
                return (
                  <g key={i}>
                    {/* Background interactive larger touch point */}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r="12"
                      fill="transparent"
                      className="cursor-pointer"
                      onMouseEnter={() => setHoveredTrendPoint({ x: p.x, y: p.y, index: i, ...p.raw })}
                      onMouseLeave={() => setHoveredTrendPoint(null)}
                    />
                    
                    {/* Visually glowing points */}
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={isHovered ? "5" : "3.5"}
                      fill={isHovered ? "#fff" : "#6366f1"}
                      stroke="#4f46e5"
                      strokeWidth={isHovered ? "2.5" : "1.5"}
                      className="transition-all"
                    />

                    {/* X bottom dates */}
                    <text
                      x={p.x}
                      y={height - 8}
                      fill="#94a3b8"
                      fontSize="7.5"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {p.raw.date}
                    </text>
                  </g>
                );
              })}

              {/* Interactive Tooltip over the Canvas inside SVG for perfect responsive scaling */}
              {hoveredTrendPoint && (
                <foreignObject
                  x={Math.min(Math.max(hoveredTrendPoint.x - 64, 5), width - 133)}
                  y={Math.max(hoveredTrendPoint.y - 55, 5)}
                  width="128"
                  height="45"
                  className="pointer-events-none overflow-visible"
                >
                  <div className="px-2 py-1 rounded-lg border border-white/10 bg-[#0E0E10]/95 backdrop-blur-md font-sans shadow-xl w-32 flex flex-col items-center justify-center">
                    <span className="text-[7.5px] font-mono text-zinc-500 uppercase leading-normal">{hoveredTrendPoint.date} 心境</span>
                    <span className="text-[10px] font-bold text-white mt-0.5 flex items-center space-x-1 leading-normal">
                      <span>{hoveredTrendPoint.value}%</span>
                      <span>
                        {hoveredTrendPoint.emotion === 'anxious' && '🔥 焦虑'}
                        {hoveredTrendPoint.emotion === 'melancholy' && '🌊 忧郁'}
                        {hoveredTrendPoint.emotion === 'calm' && '🍃 平静'}
                        {hoveredTrendPoint.emotion === 'joyful' && '☀️ 欣喜'}
                        {hoveredTrendPoint.emotion === 'grateful' && '✨ 感恩'}
                      </span>
                    </span>
                  </div>
                </foreignObject>
              )}
            </AnimatePresence>
          </svg>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="w-full flex flex-col items-center justify-center py-20 space-y-3">
        <span className="h-6 w-6 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin"></span>
        <p className="text-xs text-slate-500 font-mono">LOADING ASTRO THERMAL PROFILE...</p>
      </div>
    );
  }

  if (errorMsg) {
    return (
      <div className="w-full max-w-xl mx-auto p-6 text-center border border-white/10 bg-[#0E0E10] rounded-lg space-y-4 my-10">
        <ShieldAlert className="text-rose-400 mx-auto" size={28} />
        <h3 className="text-sm font-bold text-white">同步数据流失败</h3>
        <p className="text-xs text-slate-500 leading-relaxed">{errorMsg}</p>
        <button 
          onClick={fetchCoreData}
          className="px-5 py-1.5 bg-indigo-600 hover:bg-indigo-500 rounded-lg text-xs font-semibold text-white"
        >
          重新同步
        </button>
      </div>
    );
  }

  return (
    <div className="w-full max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6 pt-6 px-4">
      
      {/* Title */}
      <div className="lg:col-span-3 mb-2 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="font-sans font-bold text-2xl md:text-3xl tracking-tight text-white flex items-center space-x-2">
            <span>个人情绪档案</span>
          </h1>
          <p className="text-xs md:text-sm text-slate-400 font-sans tracking-wide mt-1 leading-relaxed">
            在这里回顾你的心路历程，每一颗星星都记录着成长。
          </p>
        </div>
        <button
          onClick={() => setShowResetModal(true)}
          id="btn-master-reset"
          className="self-start sm:self-center flex items-center space-x-1.5 px-4 py-2 border border-rose-500/20 hover:border-rose-500/40 hover:bg-rose-950/20 text-rose-400 hover:text-rose-300 text-xs font-semibold rounded-lg transition-all duration-300 cursor-pointer"
        >
          <Trash2 size={13} />
          <span>清空档案与历史</span>
        </button>
      </div>

      {/* Custom Reset Modal */}
      {showResetModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-sm border border-white/10 bg-[#0E0E10] p-6 rounded-xl shadow-2xl space-y-4 relative">
            {resetSuccess ? (
              <div className="text-center py-6 space-y-3">
                <div className="w-12 h-12 rounded bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 mx-auto">
                  <Check size={20} />
                </div>
                <h3 className="text-sm font-bold text-white">重置成功</h3>
                <p className="text-xs text-slate-500">所有情绪历史已永久清空，重置星空。</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="w-12 h-12 rounded bg-rose-500/10 border border-rose-500/20 flex items-center justify-center text-rose-400 mx-auto animate-bounce">
                  <Trash2 size={20} />
                </div>
                <div className="text-center space-y-1">
                  <h3 className="text-sm font-bold text-white">确定要清空全部记录吗？</h3>
                  <p className="text-xs text-slate-500 leading-relaxed">
                    此操作将永久抹去您的所有情绪档案、心愿投递星空、星愿待办清单，以及与 Orion 的所有智能对话历史，该过程不可逆。
                  </p>
                </div>
                <div className="flex items-center gap-3 pt-2">
                  <button
                    onClick={() => setShowResetModal(false)}
                    className="flex-1 px-4 py-2 rounded-lg border border-white/10 hover:bg-white/5 text-xs font-semibold text-slate-400 hover:text-white transition cursor-pointer"
                  >
                    取消
                  </button>
                  <button
                    onClick={handleResetAll}
                    id="btn-confirm-reset"
                    className="flex-1 px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-700 text-white text-xs font-semibold hover:shadow-lg transition cursor-pointer"
                  >
                    确认清空
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Column Left & Middle: Analytics Trend & Badges grid */}
      <div className="lg:col-span-2 space-y-6">
        
        {/* SVG Wave chart block */}
        <div className="p-6 rounded-xl border border-white/10 bg-[#0E0E10] shadow-2xl">
          {renderWaveChart()}
        </div>

        {/* Statistics highlights grids */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="p-4 rounded-lg border border-white/10 bg-[#0E0E10]">
            <span className="text-[10px] text-zinc-500 font-medium uppercase font-sans tracking-wide">累计倾诉天数</span>
            <div className="flex items-baseline space-x-1.5 mt-2">
              <span className="text-2xl font-bold font-mono text-indigo-100">{analytics?.daysCount || 128}</span>
              <span className="text-xs text-zinc-400">天</span>
              <span className="text-[9px] font-mono text-emerald-400">+{analytics?.daysDiffPercent || 12}%</span>
            </div>
          </div>

          <div className="p-4 rounded-lg border border-white/10 bg-[#0E0E10]">
            <span className="text-[10px] text-zinc-500 font-medium uppercase font-sans tracking-wide">点亮星愿总数</span>
            <div className="flex items-baseline space-x-1 mt-2">
              <span className="text-2xl font-bold font-mono text-indigo-100">{analytics?.totalWishes || 42}</span>
              <span className="text-xs text-zinc-400">/ {analytics?.targetWishes || 50} 颗</span>
            </div>
          </div>

          <div className="p-4 rounded-lg border border-white/10 bg-[#0E0E10]">
            <span className="text-[10px] text-zinc-500 font-medium uppercase font-sans tracking-wide">本周平均情绪值</span>
            <div className="flex items-baseline space-x-1.5 mt-2">
              <span className="text-2xl font-bold font-mono text-indigo-100">{analytics?.avgEmotionScore || 78}</span>
              <span className="text-xs text-emerald-400">{analytics?.avgEmotionLabel || '平静稳定'}</span>
            </div>
          </div>

          <div className="p-4 rounded-lg border border-white/10 bg-[#0E0E10]">
            <span className="text-[10px] text-zinc-500 font-medium uppercase font-sans tracking-wide">共计文字倾诉</span>
            <div className="flex items-baseline space-x-1 mt-2">
              <span className="text-2xl font-bold font-mono text-indigo-100">
                {analytics?.wordsCount ? (analytics.wordsCount >= 10000 ? `${(analytics.wordsCount/10000).toFixed(1)}w` : analytics.wordsCount) : '4.2w'}
              </span>
              <span className="text-xs text-zinc-400">字</span>
            </div>
          </div>
        </div>

        {/* Achievement Badges Block */}
        <div id="wrapper-badges" className="p-5 rounded-xl border border-white/10 bg-[#0E0E10] space-y-4 shadow-2xl">
          <div className="flex justify-between items-center">
            <div>
              <h4 className="text-xs font-semibold text-slate-200 flex items-center space-x-1.5">
                <Trophy size={14} className="text-amber-400" />
                <span>成就徽章</span>
              </h4>
              <p className="text-[10px] text-zinc-500 mt-0.5">每一个节点都是治愈的印记</p>
            </div>
            <span className="text-[10px] font-mono text-zinc-600">
              已解锁 ({(badges.filter((b) => b.unlocked).length)}/{(badges.length)})
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {badges.map((b) => {
              const IconComp = BADGE_ICONS[b.iconName] || Trophy;
              return (
                <div 
                  key={b.id} 
                  id={`badge-card-${b.id}`}
                  className={`p-3 rounded-lg border flex flex-col items-center justify-center text-center space-y-1.5 transition select-none group relative ${
                    b.unlocked 
                      ? 'bg-[#0A0A0B] border-white/10 text-slate-100 hover:border-white/20' 
                      : 'bg-transparent border-white/5 text-zinc-700 opacity-40'
                  }`}
                >
                  <div className={`p-2.5 rounded-full ${
                    b.unlocked 
                      ? 'bg-indigo-600/10 border border-indigo-500/20 text-[#968cf4]' 
                      : 'bg-zinc-950 text-zinc-700 border border-transparent'
                  }`}>
                    <IconComp size={16} />
                  </div>
                  <span className="text-[10px] font-semibold">{b.name}</span>
                  
                  {/* Floating description on touch/hover */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-1.5 opacity-0 group-hover:opacity-100 bg-[#0E0E10] border border-white/10 px-2 py-1 rounded text-[8px] text-slate-300 pointer-events-none transition duration-200 z-50 w-28 leading-snug">
                    {b.description}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

      </div>

      {/* Column Right: Checklist (星愿清单) */}
      <div className="lg:col-span-1">
        <div id="wrapper-星愿清单" className="p-5 rounded-xl border border-white/10 bg-[#0E0E10] flex flex-col h-full min-h-[400px] space-y-4 shadow-2xl">
          <div className="flex justify-between items-center border-b border-white/10 pb-3">
            <h3 className="text-xs font-semibold text-slate-200 flex items-center space-x-1.5 tracking-wider uppercase">
              <Calendar size={13} className="text-[#a49df5]" />
              <span>星愿清单 (Active)</span>
            </h3>
          </div>

          {/* New Todo input form inline */}
          <form onSubmit={handleAddTodo} className="flex space-x-2">
            <input
              type="text"
              id="input-new-wishlist"
              value={newTodoText}
              onChange={(e) => setNewTodoText(e.target.value)}
              placeholder="添加你想要点亮的小心愿..."
              className="flex-1 bg-[#0A0A0B] border border-white/10 focus:border-indigo-500 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-600 focus:outline-none min-w-0"
            />
            <button
              type="submit"
              id="btn-add-todo"
              className="p-2 rounded-lg bg-indigo-600 text-white hover:bg-indigo-500 transition cursor-pointer"
            >
              <Plus size={14} />
            </button>
          </form>

          {/* Checklist scrollable area */}
          <div className="flex-1 overflow-y-auto space-y-2.5 pr-1 scrollbar-thin">
            {todos.length === 0 ? (
              <div className="text-center py-16 text-zinc-600">
                <p className="text-xs italic">暂时空无一字</p>
                <p className="text-[10px] mt-1">添加些疗愈自己的日常期盼吧</p>
              </div>
            ) : (
               todos.map((todo) => (
                <div 
                  key={todo.id}
                  id={`todo-row-${todo.id}`}
                  className={`p-3 rounded-lg border flex items-center justify-between transition group ${
                    todo.completed 
                      ? 'bg-emerald-950/5 border-emerald-500/10 text-emerald-400/70' 
                      : 'bg-[#0A0A0B] border-white/5 hover:border-white/10 text-slate-300'
                  }`}
                >
                  <div className="flex items-center space-x-2.5 flex-1 min-w-0">
                    <button
                      type="button"
                      id={`btn-todo-checkbox-${todo.id}`}
                      onClick={() => handleToggleTodo(todo.id, todo.completed)}
                      className={`w-4 h-4 rounded flex items-center justify-center border transition shrink-0 cursor-pointer ${
                        todo.completed 
                          ? 'bg-emerald-500 border-emerald-500 text-slate-950' 
                          : 'border-white/10 hover:border-white/20 bg-[#0A0A0B]'
                      }`}
                    >
                      {todo.completed && <Check size={10} strokeWidth={3} />}
                    </button>
                    <div className="flex flex-col min-w-0">
                      <span className={`text-xs truncate ${todo.completed ? 'line-through opacity-60' : ''}`}>
                        {todo.text}
                      </span>
                      <span className="text-[9px] text-zinc-600 font-mono mt-0.5 select-none">
                        {todo.completed ? `已完成 于 ${todo.completedAt}` : `创建于 ${todo.createdAt}`}
                      </span>
                    </div>
                  </div>

                  <button
                    type="button"
                    id={`btn-todo-delete-${todo.id}`}
                    onClick={() => handleDeleteTodo(todo.id)}
                    className="p-1 rounded text-zinc-600 hover:text-rose-400 opacity-0 group-hover:opacity-100 transition cursor-pointer"
                  >
                    <Trash2 size={11} />
                  </button>
                </div>
              ))
            )}
          </div>

          {/* Motivational advice */}
          <div className="p-3 bg-[#0A0A0B] border border-white/10 rounded-lg flex items-start space-x-2">
            <BadgeInfo size={13} className="text-medium text-indigo-400 shrink-0 mt-0.5" />
            <p className="text-[10px] text-slate-500 leading-normal font-sans">
              完成这些行动可以增强平均情绪平稳性，有助于逐渐点亮“星河织梦人”勋章。
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}
