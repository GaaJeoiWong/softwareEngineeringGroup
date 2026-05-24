/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { VoiceDelivery } from './components/VoiceDelivery';
import { OrionChat } from './components/OrionChat';
import { EmotionCenter } from './components/EmotionCenter';
import { AuthScreen } from './components/AuthScreen';
import { Wish } from './types';
import { motion, AnimatePresence } from 'motion/react';
import { safeStorage, secureFetch } from './lib/api';

export default function App() {
  const [activeTab, setActiveTab] = useState<'delivery' | 'chat' | 'emotion'>('delivery');
  const [wishes, setWishes] = useState<Wish[]>([]);
  const [loading, setLoading] = useState(true);
  const [token, setToken] = useState<string | null>(safeStorage.getItem('orion_token'));
  const [user, setUser] = useState<{ uid: string; username: string; nickname: string; mbti: string } | null>(null);

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
        <div className="min-h-screen bg-[#0A0A0B] text-slate-200 flex flex-col items-center justify-center font-sans">
          <span className="h-6 w-6 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin"></span>
          <p className="text-xs text-slate-500 font-mono tracking-widest mt-4">VALIDATING SECURE PORTAL...</p>
        </div>
      );
    }
    return <AuthScreen onAuthSuccess={handleAuthSuccess} />;
  }

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-slate-200 flex flex-col font-sans relative overflow-x-hidden antialiased">
      
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
      <Header activeTab={activeTab} setActiveTab={setActiveTab} user={user} onLogout={handleLogout} />

      {/* Main Content Layout with tabs transitions */}
      <main className="flex-1 relative z-10 py-6 pb-12">
        {loading ? (
          <div className="flex flex-col items-center justify-center py-36 space-y-4">
            <span className="h-6 w-6 rounded-full border-2 border-indigo-500 border-t-transparent animate-spin"></span>
            <p className="text-xs text-slate-500 font-mono tracking-widest">LOADING CELESTIAL HEART SPACE...</p>
          </div>
        ) : (
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              {activeTab === 'delivery' && (
                <VoiceDelivery onWishDelivered={handleWishDelivered} />
              )}
              {activeTab === 'chat' && (
                <OrionChat wishes={wishes} />
              )}
              {activeTab === 'emotion' && (
                <EmotionCenter onStateUpdate={fetchWishes} />
              )}
            </motion.div>
          </AnimatePresence>
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
