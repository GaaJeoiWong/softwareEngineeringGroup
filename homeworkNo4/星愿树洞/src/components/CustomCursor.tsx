import { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'motion/react';

interface Particle {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  size: number;
  color: string;
  alpha: number;
}

export function CustomCursor() {
  const [coords, setCoords] = useState({ x: 0, y: 0 });
  const [isHovering, setIsHovering] = useState(false);
  const [isClicked, setIsClicked] = useState(false);
  const [particles, setParticles] = useState<Particle[]>([]);
  const [isMobile, setIsMobile] = useState(true);

  // Magnetic targeting state
  const [magneticTarget, setMagneticTarget] = useState<DOMRect | null>(null);
  const [magneticClass, setMagneticClass] = useState<string>('');

  // Refs for animation loops & interpolation (Inertia)
  const dotRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const ringRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const mouseRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  
  const cursorDotVisualRef = useRef<HTMLDivElement | null>(null);
  const cursorRingVisualRef = useRef<HTMLDivElement | null>(null);

  // Check if devices support touch / hover
  useEffect(() => {
    const checkDevice = () => {
      const hasTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
      setIsMobile(hasTouch);
    };
    checkDevice();
    window.addEventListener('resize', checkDevice);
    return () => window.removeEventListener('resize', checkDevice);
  }, []);

  // Tracking Mouse Coordinates & Hover delegation
  useEffect(() => {
    if (isMobile) return;

    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current.x = e.clientX;
      mouseRef.current.y = e.clientY;
      setCoords({ x: e.clientX, y: e.clientY });
    };

    const handleMouseOver = (e: MouseEvent) => {
      // Look for clickable elements
      const target = e.target as HTMLElement | null;
      if (!target) return;

      const clickable = target.closest('button, a, [role="button"], input[type="submit"], [onClick]');
      
      if (clickable) {
        setIsHovering(true);
        // Get bounds for magnetic snap
        const rect = clickable.getBoundingClientRect();
        setMagneticTarget(rect);
        
        // Custom interactive scaling or wrapper detection
        if (clickable.tagName === 'BUTTON') {
          setMagneticClass('button-hover');
        } else {
          setMagneticClass('link-hover');
        }
      } else {
        setIsHovering(false);
        setMagneticTarget(null);
        setMagneticClass('');
      }
    };

    // Global Click Listener for particle burst
    const handleMouseClick = (e: MouseEvent) => {
      setIsClicked(true);
      setTimeout(() => setIsClicked(false), 150);

      const count = 12; // Number of particles
      const newParticles: Particle[] = [];
      const colors = ['#818cf8', '#a78bfa', '#fbcfe8', '#67e8f9', '#a7f3d0', '#fde68a'];

      for (let i = 0; i < count; i++) {
        const angle = Math.random() * Math.PI * 2;
        const speed = Math.random() * 4 + 2;
        newParticles.push({
          id: Date.now() + Math.random(),
          x: e.clientX,
          y: e.clientY,
          vx: Math.cos(angle) * speed,
          vy: Math.sin(angle) * speed,
          size: Math.random() * 4 + 2,
          color: colors[Math.floor(Math.random() * colors.length)],
          alpha: 1.0,
        });
      }

      setParticles((prev) => [...prev, ...newParticles]);
    };

    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    document.addEventListener('mouseover', handleMouseOver, { passive: true });
    window.addEventListener('click', handleMouseClick, { passive: true });

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseover', handleMouseOver);
      window.removeEventListener('click', handleMouseClick);
    };
  }, [isMobile]);

  // Inertial interpolation loop (Smooth cursor tracking)
  useEffect(() => {
    if (isMobile) return;

    let animFrameId: number;

    const updateCursor = () => {
      const dot = dotRef.current;
      const ring = ringRef.current;
      const mouse = mouseRef.current;

      // Inertia parameters - increased values to reduce cursor latency and inertia for a crisper, hand-to-eye alignment feeling
      const dotEase = 0.45; 
      const ringEase = 0.18; // Responsive outer ring lag filter

      // Check if inside magnetic snap threshold
      let targetRingX = mouse.x;
      let targetRingY = mouse.y;

      if (magneticTarget) {
        const centerOfTargetX = magneticTarget.left + magneticTarget.width / 2;
        const centerOfTargetY = magneticTarget.top + magneticTarget.height / 2;
        
        // Calculate vector from mouse to target center
        const dx = centerOfTargetX - mouse.x;
        const dy = centerOfTargetY - mouse.y;
        
        // Snaps strongly towards center when close
        targetRingX = mouse.x + dx * 0.45;
        targetRingY = mouse.y + dy * 0.45;
      }

      dot.x += (mouse.x - dot.x) * dotEase;
      dot.y += (mouse.y - dot.y) * dotEase;

      ring.x += (targetRingX - ring.x) * ringEase;
      ring.y += (targetRingY - ring.y) * ringEase;

      // Update Visual elements directly for high-performance screen rendering via refs
      if (cursorDotVisualRef.current) {
        cursorDotVisualRef.current.style.transform = `translate3d(${dot.x}px, ${dot.y}px, 0) translate(-50%, -50%)`;
      }

      if (cursorRingVisualRef.current) {
        if (magneticTarget) {
          // If magnetic, scale the ring to warp/hug around the outer dimensions
          const padWidth = magneticTarget.width + 16;
          const padHeight = magneticTarget.height + 12;
          cursorRingVisualRef.current.style.transform = `translate3d(${ring.x}px, ${ring.y}px, 0) translate(-50%, -50%)`;
          cursorRingVisualRef.current.style.width = `${padWidth}px`;
          cursorRingVisualRef.current.style.height = `${padHeight}px`;
          cursorRingVisualRef.current.style.borderRadius = '8px';
          cursorRingVisualRef.current.style.borderColor = 'rgba(129, 140, 248, 0.45)';
          cursorRingVisualRef.current.style.backgroundColor = 'rgba(129, 140, 248, 0.08)';
        } else {
          // Normal elegant circular ring
          cursorRingVisualRef.current.style.transform = `translate3d(${ring.x}px, ${ring.y}px, 0) translate(-50%, -50%)`;
          cursorRingVisualRef.current.style.width = isHovering ? '36px' : '20px';
          cursorRingVisualRef.current.style.height = isHovering ? '36px' : '20px';
          cursorRingVisualRef.current.style.borderRadius = '50%';
          cursorRingVisualRef.current.style.borderColor = isHovering ? 'rgba(139, 92, 246, 0.6)' : 'rgba(255, 255, 255, 0.35)';
          cursorRingVisualRef.current.style.backgroundColor = 'transparent';
        }
      }

      animFrameId = requestAnimationFrame(updateCursor);
    };

    updateCursor();

    return () => cancelAnimationFrame(animFrameId);
  }, [isMobile, magneticTarget, isHovering]);

  // Click Particles physics render update
  useEffect(() => {
    if (particles.length === 0) return;

    let frameId: number;

    const updateParticles = () => {
      setParticles((prev) =>
        prev
          .map((p) => ({
            ...p,
            x: p.x + p.vx,
            y: p.y + p.vy,
            vy: p.vy + 0.1, // Soft particle gravity
            alpha: p.alpha - 0.025, // fade out
          }))
          .filter((p) => p.alpha > 0)
      );
      frameId = requestAnimationFrame(updateParticles);
    };

    frameId = requestAnimationFrame(updateParticles);
    return () => cancelAnimationFrame(frameId);
  }, [particles]);

  if (isMobile) return null;

  return (
    <>
      {/* Hide standard cursor only on fine-pointer devices */}
      <style>{`
        @media (pointer: fine) {
          html, body, button, a, [role="button"], input, select, textarea {
            cursor: none !important;
          }
        }
      `}</style>

      {/* Main Cursor Dot */}
      <div
        ref={cursorDotVisualRef}
        className="fixed top-0 left-0 w-2 h-2 rounded-full bg-indigo-400 pointer-events-none z-[9999] shadow-[0_0_8px_rgba(129,140,248,0.8)] mix-blend-screen transition-all duration-75"
        style={{ transform: 'translate3d(-100px, -100px, 0)' }}
      />

      {/* Dynamic Cursor Ring (with high inertial smooth lag) */}
      <div
        ref={cursorRingVisualRef}
        className="fixed top-0 left-0 pointer-events-none z-[9998] border border-white/35 transition-all duration-300 ease-[cubic-bezier(0.19,1,0.22,1)]"
        style={{
          width: '20px',
          height: '20px',
          borderRadius: '50%',
          transform: 'translate3d(-100px, -100px, 0)',
          boxShadow: isHovering ? '0 0 12px rgba(139, 92, 246, 0.15)' : 'none',
        }}
      />

      {/* Burst Particles canvas overlay */}
      <div className="fixed inset-0 pointer-events-none z-[10000] overflow-hidden">
        {particles.map((p) => (
          <div
            key={p.id}
            className="absolute rounded-full"
            style={{
              left: p.x,
              top: p.y,
              width: `${p.size}px`,
              height: `${p.size}px`,
              backgroundColor: p.color,
              opacity: p.alpha,
              transform: 'translate(-50%, -50%)',
              boxShadow: `0 0 ${p.size * 2}px ${p.color}`,
              transition: 'opacity 30ms linear',
            }}
          />
        ))}
      </div>
    </>
  );
}
