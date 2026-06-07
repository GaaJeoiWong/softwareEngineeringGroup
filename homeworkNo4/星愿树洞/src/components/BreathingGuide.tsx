import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { Leaf, Info, Play, Pause, RefreshCw } from 'lucide-react';

type BreathPhase = 'inhale' | 'hold' | 'exhale' | 'idle';

export function BreathingGuide() {
  const [phase, setPhase] = useState<BreathPhase>('idle');
  const [secondsLeft, setSecondsLeft] = useState(4);
  const [isActive, setIsActive] = useState(false);
  const [completedCycles, setCompletedCycles] = useState(0);

  useEffect(() => {
    if (!isActive) {
      setPhase('idle');
      return;
    }

    let timer: NodeJS.Timeout;
    
    const runPhase = () => {
      if (phase === 'idle' || phase === 'exhale') {
        setPhase('inhale');
        setSecondsLeft(4);
      } else if (phase === 'inhale') {
        setPhase('hold');
        setSecondsLeft(4);
      } else if (phase === 'hold') {
        setPhase('exhale');
        setSecondsLeft(4);
        setCompletedCycles((c) => c + 1);
      }
    };

    if (secondsLeft > 0) {
      timer = setTimeout(() => {
        setSecondsLeft((s) => s - 1);
      }, 1000);
    } else {
      runPhase();
    }

    return () => clearTimeout(timer);
  }, [secondsLeft, isActive, phase]);

  const handleToggle = () => {
    if (!isActive) {
      setIsActive(true);
      setPhase('inhale');
      setSecondsLeft(4);
    } else {
      setIsActive(false);
      setPhase('idle');
    }
  };

  const handleReset = () => {
    setIsActive(false);
    setPhase('idle');
    setSecondsLeft(4);
    setCompletedCycles(0);
  };

  // Label visual mapping
  const getPhaseConfig = () => {
    switch (phase) {
      case 'inhale':
        return {
          title: '吸气 • INHALE',
          desc: '随着光圈扩张缓慢吸气，感受星河之息入胸膛',
          color: 'from-emerald-400 to-teal-500',
          scale: 1.5,
          duration: 4,
        };
      case 'hold':
        return {
          title: '屏息 • HOLD',
          desc: '静止思绪，将温暖的情感凝练心底',
          color: 'from-indigo-400 to-purple-500',
          scale: 1.5,
          duration: 4,
        };
      case 'exhale':
        return {
          title: '呼气 • EXHALE',
          desc: '随着光圈收缩徐徐呼气，释放心海所有的焦虑与疲惫',
          color: 'from-indigo-500 to-blue-600',
          scale: 1.0,
          duration: 4,
        };
      default:
        return {
          title: '心境平静引导 • ZENTIME',
          desc: '书写心声前，试着做 1 分钟深呼吸，能够舒缓情绪',
          color: 'from-[#1A1A1E] to-[#141525]',
          scale: 1.1,
          duration: 0,
        };
    }
  };

  const config = getPhaseConfig();

  return (
    <div className="p-6 rounded-xl border border-white/10 bg-[#0E0E10] shadow-xl flex flex-col md:flex-row items-center gap-6 relative overflow-hidden select-none">
      
      {/* Decorative pulse glow background */}
      <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 blur-3xl pointer-events-none rounded-full" />
      
      {/* Breathing Guide Interactive Visual Circle node */}
      <div className="relative w-36 h-36 flex items-center justify-center shrink-0">
        <AnimatePresence mode="popLayout">
          {/* Wave Ripple Circles */}
          {isActive && phase !== 'idle' && (
            <>
              <motion.div
                key={`ripple-1-${phase}`}
                initial={{ scale: 0.8, opacity: 0.5 }}
                animate={{ 
                  scale: phase === 'inhale' ? 2.0 : phase === 'exhale' ? 1.0 : 1.5, 
                  opacity: 0 
                }}
                transition={{ duration: 4, ease: 'easeOut', repeat: Infinity }}
                className="absolute inset-0 rounded-full bg-indigo-500/10 pointer-events-none"
              />
              <motion.div
                key={`ripple-2-${phase}`}
                initial={{ scale: 0.8, opacity: 0.3 }}
                animate={{ 
                  scale: phase === 'inhale' ? 1.8 : phase === 'exhale' ? 0.9 : 1.3, 
                  opacity: 0 
                }}
                transition={{ duration: 4, ease: 'easeOut', repeat: Infinity, delay: 1.5 }}
                className="absolute inset-0 rounded-full bg-indigo-400/5 pointer-events-none"
              />
            </>
          )}
        </AnimatePresence>

        {/* Central Breathing Orb */}
        <motion.div
          animate={{
            scale: config.scale,
          }}
          transition={{
            duration: config.duration || 1,
            ease: "easeInOut"
          }}
          className={`w-20 h-20 rounded-full bg-gradient-to-tr ${config.color} flex flex-col items-center justify-center border border-white/20 shadow-lg relative z-10 transition-all`}
        >
          {isActive ? (
            <div className="text-center">
              <span className="text-xl font-bold font-mono text-white tracking-widest">{secondsLeft}</span>
              <span className="text-[7px] font-mono block text-white/70 uppercase leading-none mt-0.5">secs</span>
            </div>
          ) : (
            <Leaf size={22} className="text-[#a5b4fc] animate-pulse" />
          )}
        </motion.div>
      </div>

      {/* Guide Typography Info */}
      <div className="flex-1 space-y-3.5 text-center md:text-left z-10">
        <div>
          <div className="flex items-center justify-center md:justify-start gap-2">
            <span className="text-xs uppercase font-mono tracking-widest text-[#a5b4fc] font-semibold bg-indigo-500/10 px-2.5 py-0.5 rounded-full border border-indigo-500/20">
              {config.title}
            </span>
            {completedCycles > 0 && (
              <span className="text-[10px] font-mono text-emerald-400">
                己循环 {completedCycles} 次
              </span>
            )}
          </div>
          <p className="text-xs md:text-sm text-slate-300 mt-2 font-sans leading-relaxed min-h-[40px]">
            {config.desc}
          </p>
        </div>

        {/* Action button controls */}
        <div className="flex flex-wrap items-center justify-center md:justify-start gap-2 pt-1">
          <button
            onClick={handleToggle}
            className={`flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition active:scale-95 cursor-pointer border ${
              isActive
                ? 'bg-rose-500/10 hover:bg-rose-500/15 border-rose-500/20 text-rose-400'
                : 'bg-indigo-600 hover:bg-indigo-500 border-indigo-500 text-white shadow shadow-indigo-600/10'
            }`}
          >
            {isActive ? (
              <>
                <Pause size={12} />
                <span>暂停冥想</span>
              </>
            ) : (
              <>
                <Play size={12} />
                <span>开启冥想呼吸</span>
              </>
            )}
          </button>

          {isActive && (
            <button
              onClick={handleReset}
              className="p-1.5 rounded-lg bg-[#0A0A0B] border border-white/5 text-slate-400 hover:text-slate-200 transition active:scale-95 cursor-pointer flex items-center justify-center"
              title="重置"
            >
              <RefreshCw size={12} />
            </button>
          )}
        </div>
      </div>

    </div>
  );
}
