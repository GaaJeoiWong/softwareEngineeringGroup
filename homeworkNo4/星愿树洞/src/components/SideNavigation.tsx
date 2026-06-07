import { motion } from 'motion/react';
import { HeartHandshake, MessageSquareHeart, Activity } from 'lucide-react';
import { playTabTick } from '../lib/sound';

interface SideNavigationProps {
  activeTab: 'delivery' | 'chat' | 'emotion';
  setActiveTab: (tab: 'delivery' | 'chat' | 'emotion') => void;
}

export function SideNavigation({ activeTab, setActiveTab }: SideNavigationProps) {
  const navItems = [
    {
      id: 'delivery' as const,
      label: '投递心声',
      sub: 'SEND WISH',
      icon: HeartHandshake,
      color: 'from-amber-400 to-orange-500',
      activeColor: 'rgba(245, 158, 11, 0.15)',
    },
    {
      id: 'chat' as const,
      label: '智能回信',
      sub: 'CELESTIAL CHAT',
      icon: MessageSquareHeart,
      color: 'from-indigo-400 to-indigo-600',
      activeColor: 'rgba(99, 102, 241, 0.15)',
    },
    {
      id: 'emotion' as const,
      label: '情绪中心',
      sub: 'SOUL SPECTRUM',
      icon: Activity,
      color: 'from-violet-400 to-purple-600',
      activeColor: 'rgba(139, 92, 246, 0.15)',
    },
  ];

  const handleNavClick = (tab: 'delivery' | 'chat' | 'emotion') => {
    setActiveTab(tab);
    playTabTick();
  };

  return (
    <div className="fixed right-6 top-1/2 -translate-y-1/2 z-50 flex flex-col items-center gap-4 bg-black/40 backdrop-blur-xl border border-white/10 p-4 rounded-2xl shadow-[0_8px_32px_rgba(0,0,0,0.5)] select-none">
      {/* Decorative vertical connection node track */}
      <div className="absolute top-8 bottom-8 left-1/2 -translate-x-1/2 w-[1px] bg-white/5 pointer-events-none" />

      {navItems.map((item) => {
        const Icon = item.icon;
        const isActive = activeTab === item.id;

        return (
          <button
            key={item.id}
            onClick={() => handleNavClick(item.id)}
            className="group relative flex items-center justify-center w-12 h-12 rounded-xl transition-all duration-300 focus:outline-none"
            title={item.label}
          >
            {/* Active sliding glow backdrop node */}
            {isActive && (
              <motion.div
                layoutId="sideActiveBg"
                className="absolute inset-0 rounded-xl bg-indigo-500/10 border border-indigo-500/30 shadow-[0_0_15px_rgba(99,102,241,0.25)]"
                transition={{ type: 'spring', stiffness: 300, damping: 25 }}
              />
            )}

            {/* Glowing dot representing current state */}
            {isActive && (
              <motion.div
                layoutId="sideDotCore"
                className="absolute -left-1 w-1.5 h-1.5 rounded-full bg-indigo-400 shadow-[0_0_8px_#818cf8]"
                transition={{ type: 'spring', stiffness: 350, damping: 20 }}
              />
            )}

            {/* Icon visual container */}
            <div
              className={`relative z-10 p-2 rounded-lg transition-all duration-300 ${
                isActive
                  ? 'text-indigo-300'
                  : 'text-slate-500 group-hover:text-slate-300 group-hover:scale-110'
              }`}
            >
              <Icon size={20} className={isActive ? "drop-shadow-[0_0_6px_rgba(129,140,248,0.7)]" : ""} />
            </div>

            {/* Premium sliding typography label on hover */}
            <div className="absolute right-14 pointer-events-none opacity-0 group-hover:opacity-100 translate-x-3 group-hover:translate-x-0 transition-all duration-300 ease-out flex flex-col items-end pr-2">
              <div className="bg-[#0E0E10]/95 border border-white/10 backdrop-blur-md px-3 py-1.5 rounded-lg whitespace-nowrap shadow-xl">
                <span className="text-xs font-semibold text-slate-100 font-sans block leading-none">
                  {item.label}
                </span>
                <span className="text-[8px] font-mono text-indigo-400 tracking-wider block mt-0.5 leading-none">
                  {item.sub}
                </span>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
