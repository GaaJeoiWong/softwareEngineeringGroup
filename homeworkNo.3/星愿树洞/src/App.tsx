/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect, useRef } from 'react';
import { Header } from './components/Header';
import { SideNavigation } from './components/SideNavigation';
import { VoiceDelivery } from './components/VoiceDelivery';
import { OrionChat } from './components/OrionChat';
import { EmotionCenter } from './components/EmotionCenter';
import { AuthScreen } from './components/AuthScreen';
import { CelestialParticles } from './components/CelestialParticles';
import { CustomCursor } from './components/CustomCursor';
import { CelestialLoading } from './components/CelestialLoading';
import { Wish } from './types';
import { motion, AnimatePresence } from 'motion/react';
import { safeStorage, secureFetch } from './lib/api';

export default function App() {
  const [activeTab, setActiveTab] = useState<'delivery' | 'chat' | 'emotion'>('delivery');
  const [wishes, setWishes] = useState<Wish[]>([]);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState<string | null>(safeStorage.getItem('orion_token'));
  const [user, setUser] = useState<{ uid: string; username: string; nickname: string; mbti: string } | null>(null);

  // References for inertial scrolling and section detection
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const deliveryRef = useRef<HTMLDivElement | null>(null);
  const chatRef = useRef<HTMLDivElement | null>(null);
  const emotionRef = useRef<HTMLDivElement | null>(null);

  const targetScrollY = useRef(0);
  const currentScrollY = useRef(0);
  const isClickingTabRef = useRef(false);
  const isAnimatingRef = useRef(false);
  const lastScrollTime = useRef<number>(0);

  // Setup smooth wheel listener configuration
  useEffect(() => {
    if (loading || !user) return;

    const handleWheel = (e: WheelEvent) => {
      const container = scrollContainerRef.current;
      if (!container) return;

      // Allow scrolling sub-containers (like chat panels) natively
      const target = e.target as HTMLElement;
      const scrollableChild = target.closest('.overflow-y-auto, .overflow-y-scroll, .scrollbar-thin');
      
      if (scrollableChild && scrollableChild !== container) {
        // Double check if the element can currently scroll in the vertical direction
        if (scrollableChild.scrollHeight > scrollableChild.clientHeight) {
          return;
        }
      }

      // Check key limits and block page-level standard snapping, accumulate deltas
      e.preventDefault();
      const maxScroll = container.scrollHeight - container.clientHeight;
      targetScrollY.current = Math.max(0, Math.min(maxScroll, targetScrollY.current + e.deltaY));
      isAnimatingRef.current = true;
      lastScrollTime.current = Date.now();
    };

    // Bind to the global window to capture scrolls absolutely anywhere on screen
    window.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      window.removeEventListener('wheel', handleWheel);
    };
  }, [loading, user]);

  // Sync animation loops for inertial scroll interpolation and tab indices mapping
  useEffect(() => {
    if (loading || !user) return;

    let animId: number;

    const smoothScrollLoop = () => {
      const container = scrollContainerRef.current;
      if (container) {
        if (isAnimatingRef.current) {
          const diff = targetScrollY.current - currentScrollY.current;
          
          // Linear interpolation with moderate responsive deceleration (0.15 snappy factor) to reduce scroll inertia
          if (Math.abs(diff) > 0.5) {
            currentScrollY.current += diff * 0.15;
            container.scrollTop = currentScrollY.current;
          } else {
            currentScrollY.current = targetScrollY.current;
            container.scrollTop = currentScrollY.current;
            isAnimatingRef.current = false;
          }
        }

        // Section focal tracking & Intelligent automatic snapping
        const scrollTop = container.scrollTop;
        if (!isClickingTabRef.current) {
          const deliveryOffset = deliveryRef.current?.offsetTop || 0;
          const chatOffset = chatRef.current?.offsetTop || 0;
          const emotionOffset = emotionRef.current?.offsetTop || 0;

          // Safe snappy offsets designed to avoid overlapping with the sticky top header (Header height + padding padding buffer)
          const deliverySnap = deliveryOffset;
          const chatSnap = Math.max(0, chatOffset - 80);
          const emotionSnap = Math.max(0, emotionOffset - 80);
          
          // Focus detector offsets (using the exact middle point between adjusted snap positions)
          const viewportCenter = scrollTop + container.clientHeight / 2.5;

          let currentActive: 'delivery' | 'chat' | 'emotion' = 'delivery';
          if (viewportCenter >= emotionOffset - 90) {
            currentActive = 'emotion';
          } else if (viewportCenter >= chatOffset - 80) {
            currentActive = 'chat';
          } else {
            currentActive = 'delivery';
          }

          if (activeTab !== currentActive) {
            setActiveTab(currentActive);
          }
        }
      }
      animId = requestAnimationFrame(smoothScrollLoop);
    };

    animId = requestAnimationFrame(smoothScrollLoop);
    return () => cancelAnimationFrame(animId);
  }, [loading, user, activeTab]);

  // Sync targets if users scroll natively with touch swipes / swipe-pads / scrollbar
  const handleScroll = () => {
    const container = scrollContainerRef.current;
    if (!container) return;
    
    // If not in active custom anime loop, sync key coordinates
    if (!isAnimatingRef.current) {
      currentScrollY.current = container.scrollTop;
      targetScrollY.current = container.scrollTop;
    }
    lastScrollTime.current = Date.now();
  };

  // Safe tab actions scroll trigger
  const handleTabChange = (tab: 'delivery' | 'chat' | 'emotion') => {
    setActiveTab(tab);
    
    const container = scrollContainerRef.current;
    if (!container) return;

    let targetOffset = 0;
    if (tab === 'delivery') {
      targetOffset = deliveryRef.current?.offsetTop || 0;
    } else if (tab === 'chat') {
      // Offset by 80px to avoid sticky top header overlap and reserve breathing room (matching emotion segment)
      targetOffset = Math.max(0, (chatRef.current?.offsetTop || 0) - 80);
    } else if (tab === 'emotion') {
      // Offset by 80px downward so the title "个人情绪档案" won't be blocked by the header
      targetOffset = Math.max(0, (emotionRef.current?.offsetTop || 0) - 80);
    }

    isClickingTabRef.current = true;
    targetScrollY.current = targetOffset;
    isAnimatingRef.current = true;
    
    // Safely free the scroll event triggers after transition finishes
    setTimeout(() => {
      isClickingTabRef.current = false;
    }, 1000);
  };

  // Validate the existing login session / load current profile on startup
  useEffect(() => {
    if (token) {
      validateSession();
    } else {
      setLoading(false);
    }
  }, [token]);

  const validateSession = async () => {
    try {
      const resp = await secureFetch('/api/auth/me');
      if (resp.ok) {
        const data = await resp.json();
        setUser(data.user);
        fetchWishes();
      } else {
        // Token is invalid/expired
        handleLogout();
      }
    } catch (err) {
      console.error("Session validation failed:", err);
      setLoading(false);
    }
  };

  const fetchWishes = async () => {
    try {
      const resp = await secureFetch('/api/wishes');
      if (resp.ok) {
        const data = await resp.json();
        setWishes(data);
      }
    } catch (err) {
      console.error("Error fetching wishes:", err);
    } finally {
      setLoading(false);
    }
  };

  const handleAuthSuccess = (newToken: string, newUser: { uid: string; username: string; nickname: string; mbti: string }) => {
    safeStorage.setItem('orion_token', newToken);
    setToken(newToken);
    setUser(newUser);
    setActiveTab('delivery');
  };

  const handleLogout = () => {
    safeStorage.removeItem('orion_token');
    setToken(null);
    setUser(null);
    setWishes([]);
  };

  const handleWishDelivered = (newWish: Wish) => {
    setWishes(prev => [newWish, ...prev]);
  };

  // If the user isn't authenticated yet, show the premium Auth gateway
  if (!token || !user) {
    if (loading && token) {
      return (
        <div className="min-h-screen bg-[#0A0A0B] text-slate-200 flex flex-col items-center justify-center font-sans p-6">
          <CelestialLoading label="VALIDATING SECURE PORTAL..." sublabel="正在验证安全密钥，连接星空信使服务..." />
        </div>
      );
    }
    return <AuthScreen onAuthSuccess={handleAuthSuccess} />;
  }

  return (
    <div className="h-screen bg-[#0A0A0B] text-slate-200 flex flex-col font-sans relative overflow-hidden antialiased">
      
      {/* High Inertia Magnetic Celestial Cursor */}
      <CustomCursor />

      {/* Interactive Galactic Particles Background */}
      <CelestialParticles intensity="medium" themeColor="mixed" emotion={wishes[0]?.emotion || 'mixed'} />

      {/* Immersive space dust backgrounds */}
      <div 
        className="absolute inset-0 z-0 pointer-events-none opacity-40" 
        style={{
          backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.04) 1px, transparent 1px)',
          backgroundSize: '24px 24px'
        }}
      />
      <div className="absolute top-1/4 left-1/3 w-96 h-96 rounded-full bg-indigo-500/5 blur-3xl pointer-events-none" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 rounded-full bg-violet-500/5 blur-3xl pointer-events-none" />

      {/* Branding Header with dynamic auth props */}
      <Header activeTab={activeTab} setActiveTab={handleTabChange} user={user} onLogout={handleLogout} />

      {/* Floating Side Navigation */}
      <SideNavigation activeTab={activeTab} setActiveTab={handleTabChange} />

      {/* Main Content Layout with smooth section offsets */}
      <main 
        ref={scrollContainerRef}
        onScroll={handleScroll}
        className="flex-1 relative z-10 overflow-y-auto scrollbar-none"
        style={{ scrollBehavior: 'auto' }}
      >
        {loading ? (
          <CelestialLoading 
            label="LOADING CELESTIAL HEART SPACE..." 
            sublabel="正在连通星盘，同步专属心灵档案，编织深夜树洞..." 
            className="py-44" 
          />
        ) : (
          <div className="flex flex-col w-full pb-32">
            {/* Section 1: 投递心声 */}
            <div 
              id="section-delivery" 
              ref={deliveryRef} 
              className="w-full flex items-center justify-center p-4 md:p-6 min-h-screen pt-28 relative"
            >
              <div className="w-full max-w-5xl mx-auto">
                <VoiceDelivery onWishDelivered={handleWishDelivered} />
              </div>
            </div>

            {/* Section 2: 智能回信 */}
            <div 
              id="section-chat" 
              ref={chatRef} 
              className="w-full flex items-center justify-center p-4 md:p-6 min-h-screen pt-28 relative"
            >
              <div className="w-full max-w-5xl mx-auto">
                <OrionChat wishes={wishes} />
              </div>
            </div>

            {/* Section 3: 情绪中心 */}
            <div 
              id="section-emotion" 
              ref={emotionRef} 
              className="w-full flex items-center justify-center p-4 md:p-6 min-h-screen pt-28 relative"
            >
              <div className="w-full max-w-5xl mx-auto">
                <EmotionCenter onStateUpdate={fetchWishes} wishes={wishes} />
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Immersive Sophisticated Dark status info bar footer */}
      <footer className="h-9 bg-[#0E0E10] border-t border-white/10 flex items-center justify-between px-6 shrink-0 relative z-20 select-none">
        <div className="flex items-center gap-6 text-[10px] text-slate-500 font-mono">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500"></span> 
            <span>Orion System Ready</span>
          </div>
          <div className="hidden sm:flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse"></span> 
            <span>Syncing Celestial Nodes ({wishes.length} stars)</span>
          </div>
          <div className="hidden md:block">
            SECURITY LEVEL: E2EE
          </div>
        </div>
        <div className="text-[10px] text-slate-500 font-mono tracking-tighter">
          v2.0.4-beta_celestial
        </div>
      </footer>
    </div>
  );
}
