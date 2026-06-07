import { motion } from 'motion/react';

interface CelestialLoadingProps {
  label?: string;
  sublabel?: string;
  className?: string;
}

export function CelestialLoading({ 
  label = "LOADING CELESTIAL HEART SPACE...", 
  sublabel = "正在连通星盘，编织深邃夜空...",
  className = "py-20"
}: CelestialLoadingProps) {
  return (
    <div className={`flex flex-col items-center justify-center text-center select-none space-y-6 relative overflow-hidden w-full ${className}`}>
      {/* Dynamic multi-layered glowing orbits */}
      <div className="relative w-24 h-24 flex items-center justify-center">
        {/* Outer slow breathing ring */}
        <motion.div 
          className="absolute inset-0 rounded-full border border-indigo-500/20 shadow-[0_0_30px_rgba(99,102,241,0.1)]"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.3, 0.7, 0.3],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />

        {/* Middle spinning dashed path */}
        <motion.div 
          className="absolute w-20 h-20 rounded-full border-2 border-dashed border-indigo-400/30"
          animate={{ rotate: 360 }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "linear"
          }}
        />

        {/* Counter rotating inner ring */}
        <motion.div 
          className="absolute w-14 h-14 rounded-full border border-[#fb7185]/20 border-t-[#a78bfa]/50"
          animate={{ rotate: -360 }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: "linear"
          }}
        />

        {/* Central glowing core node */}
        <motion.div 
          className="absolute w-6 h-6 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-600 flex items-center justify-center shadow-[0_0_20px_rgba(99,102,241,0.6)]"
          animate={{
            scale: [1, 1.15, 1],
            boxShadow: [
              "0 0 15px rgba(99,102,241,0.4)",
              "0 0 35px rgba(139,92,246,0.8)",
              "0 0 15px rgba(99,102,241,0.4)"
            ]
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        >
          <div className="w-2.5 h-2.5 rounded-full bg-white" />
        </motion.div>

        {/* Orbiting star items */}
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="absolute w-1.5 h-1.5 rounded-full bg-[#fde68a] shadow-[0_0_6px_#fde68a]"
            animate={{
              x: [
                Math.cos((i * Math.PI * 2) / 3) * 36,
                Math.cos((i * Math.PI * 2) / 3 + Math.PI * 2) * 36,
              ],
              y: [
                Math.sin((i * Math.PI * 2) / 3) * 36,
                Math.sin((i * Math.PI * 2) / 3 + Math.PI * 2) * 36,
              ],
            }}
            transition={{
              duration: 6 + i,
              repeat: Infinity,
              ease: "linear",
            }}
          />
        ))}
      </div>

      {/* Narrative Label animations */}
      <div className="space-y-1.5 px-6">
        <motion.h3 
          className="text-[11px] font-mono tracking-[0.25em] text-indigo-300 font-semibold uppercase"
          animate={{ opacity: [0.4, 1, 0.4] }}
          transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
        >
          {label}
        </motion.h3>
        <p className="text-[10px] text-slate-500 max-w-xs mx-auto leading-relaxed">
          {sublabel}
        </p>
      </div>
    </div>
  );
}
