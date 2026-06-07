import { useEffect, useRef } from 'react';
import { EmotionType } from '../types';

interface ComponentProps {
  intensity?: 'low' | 'medium' | 'high';
  themeColor?: 'indigo' | 'amber' | 'rose' | 'emerald' | 'violet' | 'mixed';
  emotion?: EmotionType | 'mixed';
}

export function CelestialParticles({ 
  intensity = 'medium', 
  themeColor = 'mixed', 
  emotion = 'mixed' 
}: ComponentProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let particles: Particle[] = [];
    let shootingStars: ShootingStar[] = [];
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    // Mouse coordinates
    const mouse = {
      x: null as number | null,
      y: null as number | null,
      radius: 120, // Interaction radius
    };

    // Configuration map based on user's active emotional state
    const getEmotionConfig = (emo: string) => {
      switch (emo) {
        case 'calm':
          return {
            colors: ['#e0f2fe', '#bae6fd', '#38bdf8', '#a5f3fc', '#a7f3d0'],
            speedMultiplier: 0.4,
            twinkleMultiplier: 0.5,
            shootingStarChance: 0.994, // Increased frequency (was 0.999)
            cometSpeedScale: 0.75,
            baseColor: '56, 189, 248'
          };
        case 'melancholy':
          return {
            colors: ['#1e40af', '#1e3a8a', '#3b82f6', '#60a5fa', '#818cf8', '#a5b4fc'],
            speedMultiplier: 0.35,
            twinkleMultiplier: 0.6,
            shootingStarChance: 0.995, // Increased frequency (was 0.9995)
            cometSpeedScale: 0.6,
            baseColor: '129, 140, 248'
          };
        case 'joyful':
          return {
            colors: ['#fef08a', '#fde68a', '#fbbf24', '#fbcfe8', '#f472b6', '#fda4af'],
            speedMultiplier: 1.8,
            twinkleMultiplier: 1.8,
            shootingStarChance: 0.978, // Highly elegant frequent showers! (was 0.982)
            cometSpeedScale: 1.6,
            baseColor: '251, 191, 36'
          };
        case 'grateful':
          return {
            colors: ['#f3e8ff', '#ddd6fe', '#c084fc', '#ebd5fc', '#ffffff', '#fed7aa'],
            speedMultiplier: 0.8,
            twinkleMultiplier: 1.0,
            shootingStarChance: 0.991, // Increased frequency (was 0.994)
            cometSpeedScale: 1.1,
            baseColor: '192, 132, 252'
          };
        case 'anxious':
          return {
            colors: ['#fecdd3', '#fda4af', '#f43f5e', '#f97316', '#ef4444', '#f59e0b'],
            speedMultiplier: 1.5,
            twinkleMultiplier: 2.2,
            shootingStarChance: 0.988, // Increased frequency (was 0.992)
            cometSpeedScale: 1.4,
            baseColor: '244, 63, 94'
          };
        default: {
          // Map standard themes to colors if emotion is mixed
          const themeMap = {
            indigo: ['#818cf8', '#6366f1', '#4f46e5', '#a5b4fc'],
            amber: ['#fbbf24', '#f59e0b', '#d97706', '#fde68a'],
            rose: ['#fb7185', '#f43f5e', '#e11d48', '#fecdd3'],
            emerald: ['#34d399', '#10b981', '#059669', '#a7f3d0'],
            violet: ['#a78bfa', '#8b5cf6', '#7c3aed', '#ddd6fe'],
            mixed: ['#818cf8', '#a78bfa', '#fbcfe8', '#67e8f9', '#a7f3d0', '#fde68a'],
          };
          const colors = themeMap[themeColor as keyof typeof themeMap] || themeMap.mixed;
          return {
            colors,
            speedMultiplier: 1.0,
            twinkleMultiplier: 1.0,
            shootingStarChance: 0.992, // Increased frequency (was 0.995)
            cometSpeedScale: 1.1,
            baseColor: '129, 140, 248'
          };
        }
      }
    };

    const activeConfig = getEmotionConfig(emotion);

    class Particle {
      x: number;
      y: number;
      size: number;
      baseX: number;
      baseY: number;
      density: number;
      opacity: number;
      fadeSpeed: number;
      growing: boolean;
      vx: number;
      vy: number;
      color: string;

      constructor() {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        this.baseX = this.x;
        this.baseY = this.y;
        this.size = Math.random() * 2 + 0.5;
        this.vx = (Math.random() - 0.5) * 0.15 * activeConfig.speedMultiplier;
        this.vy = (Math.random() - 0.5) * 0.15 * activeConfig.speedMultiplier;

        // Density for parallax / mouse push
        this.density = Math.random() * 20 + 2;
        this.opacity = Math.random() * 0.5 + 0.2;
        this.fadeSpeed = (Math.random() * 0.005 + 0.002) * activeConfig.twinkleMultiplier;
        this.growing = Math.random() > 0.5;

        // Visual Colors matching Orion Palette or active Emotion Config
        this.color = activeConfig.colors[Math.floor(Math.random() * activeConfig.colors.length)];
      }

      update() {
        // Star Twinkle Effect
        if (this.growing) {
          this.opacity += this.fadeSpeed;
          if (this.opacity >= 0.85) this.growing = false;
        } else {
          this.opacity -= this.fadeSpeed;
          if (this.opacity <= 0.15) this.growing = true;
        }

        // Mouse Interactivity - Soft Gravity Push
        if (mouse.x !== null && mouse.y !== null) {
          const dx = mouse.x - this.x;
          const dy = mouse.y - this.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < mouse.radius) {
            const forceDirectionX = dx / distance;
            const forceDirectionY = dy / distance;
            const maxForce = mouse.radius;
            const force = (maxForce - distance) / maxForce;
            const directionX = forceDirectionX * force * this.density * 0.6;
            const directionY = forceDirectionY * force * this.density * 0.6;

            this.x -= directionX;
            this.y -= directionY;
          }
        }

        // Float drift
        this.x += this.vx;
        this.y += this.vy;

        // Add specialized jitter if user is anxious
        if (emotion === 'anxious') {
          this.x += (Math.random() - 0.5) * 0.35;
          this.y += (Math.random() - 0.5) * 0.35;
        }

        // Soft rebound inside bounds
        if (this.x < 0) this.x = width;
        if (this.x > width) this.x = 0;
        if (this.y < 0) this.y = height;
        if (this.y > height) this.y = 0;
      }

      draw(context: CanvasRenderingContext2D) {
        context.save();
        context.beginPath();
        context.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        context.fillStyle = this.color;
        context.shadowBlur = this.size * 3;
        context.shadowColor = this.color;
        context.globalAlpha = this.opacity;
        context.fill();
        context.restore();
      }
    }

    class ShootingStar {
      x: number;
      y: number;
      length: number;
      speed: number;
      angle: number;
      opacity: number;
      active: boolean;

      constructor() {
        this.active = false;
        this.x = 0;
        this.y = 0;
        this.length = 0;
        this.speed = 0;
        this.angle = 0;
        this.opacity = 0;
        this.reset();
      }

      reset() {
        this.active = Math.random() > activeConfig.shootingStarChance;
        if (!this.active) return;
        this.x = Math.random() * width * 0.8;
        this.y = Math.random() * height * 0.2;

        const baseLength = emotion === 'joyful' ? 120 : 80;
        const baseSpeed = emotion === 'joyful' ? 12 : 8;

        this.length = (Math.random() * baseLength + 40) * activeConfig.cometSpeedScale;
        this.speed = (Math.random() * baseSpeed + 6) * activeConfig.cometSpeedScale;
        this.angle = Math.PI / 6 + (Math.random() - 0.5) * 0.15; // standard 30deg angle
        this.opacity = 1.0;
      }

      update() {
        if (!this.active) {
          this.reset();
          return;
        }

        // Move across angle
        this.x += Math.cos(this.angle) * this.speed;
        this.y += Math.sin(this.angle) * this.speed;
        this.opacity -= 0.02;

        if (this.opacity <= 0 || this.x > width || this.y > height) {
          this.active = false;
        }
      }

      draw(context: CanvasRenderingContext2D) {
        if (!this.active) return;

        context.save();
        const endX = this.x - Math.cos(this.angle) * this.length;
        const endY = this.y - Math.sin(this.angle) * this.length;

        // Custom meteor tail gradient based on active emotion color mapping
        const grad = context.createLinearGradient(this.x, this.y, endX, endY);
        grad.addColorStop(0, `rgba(255, 255, 255, ${this.opacity})`);
        grad.addColorStop(0.25, `rgba(${activeConfig.baseColor}, ${this.opacity * 0.75})`);
        grad.addColorStop(1, `rgba(${activeConfig.baseColor}, 0)`);

        // Draw meteor trail with shadow glow
        context.shadowBlur = 10;
        context.shadowColor = `rgba(${activeConfig.baseColor}, ${this.opacity * 0.8})`;
        context.beginPath();
        context.strokeStyle = grad;
        context.lineWidth = 2.2;
        context.moveTo(this.x, this.y);
        context.lineTo(endX, endY);
        context.stroke();

        // Draw glowing comet head/nucleus
        context.shadowBlur = 14;
        context.shadowColor = `rgba(${activeConfig.baseColor}, ${this.opacity * 0.95})`;
        context.beginPath();
        context.arc(this.x, this.y, 2.0, 0, Math.PI * 2);
        context.fillStyle = `rgba(255, 255, 255, ${this.opacity})`;
        context.fill();

        // Subtle outer flare ring
        context.shadowBlur = 0;
        context.beginPath();
        context.arc(this.x, this.y, 4.5, 0, Math.PI * 2);
        context.fillStyle = `rgba(${activeConfig.baseColor}, ${this.opacity * 0.35})`;
        context.fill();

        context.restore();
      }
    }

    // Determine particle count based on device strength (width)
    const getParticleCount = () => {
      const base = width < 768 ? 40 : 100;
      if (intensity === 'low') return base * 0.5;
      if (intensity === 'high') return base * 1.5;
      return base;
    };

    const init = () => {
      particles = [];
      const count = getParticleCount();
      for (let i = 0; i < count; i++) {
        particles.push(new Particle());
      }

      shootingStars = [];
      const sCount = Math.max(1, Math.floor(count / 30));
      for (let i = 0; i < sCount; i++) {
        shootingStars.push(new ShootingStar());
      }
    };

    // Connection Constellation Linkages
    const drawConstellations = () => {
      const maxDistance = 90;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x;
          const dy = particles[i].y - particles[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < maxDistance) {
            // Check if mouse is near to make links light up!
            let bonusGlow = 0;
            if (mouse.x !== null && mouse.y !== null) {
              const mx1 = mouse.x - particles[i].x;
              const my1 = mouse.y - particles[i].y;
              const mDist = Math.sqrt(mx1 * mx1 + my1 * my1);
              if (mDist < mouse.radius) {
                bonusGlow = (mouse.radius - mDist) / mouse.radius;
              }
            }

            const baseAlpha = (1 - dist / maxDistance) * 0.12;
            const alpha = Math.min(0.35, baseAlpha + bonusGlow * 0.15);

            ctx.save();
            ctx.beginPath();
            ctx.moveTo(particles[i].x, particles[i].y);
            ctx.lineTo(particles[j].x, particles[j].y);
            
            // Constellation line color mixes Indigo, Gold, Emerald, or Rose based on emotional theme
            ctx.strokeStyle = `rgba(${activeConfig.baseColor}, ${alpha})`;
            ctx.lineWidth = 0.6 + bonusGlow * 0.4;
            ctx.stroke();
            ctx.restore();
          }
        }
      }
    };

    const animate = () => {
      ctx.clearRect(0, 0, width, height);

      // Animation stars
      particles.forEach((p) => {
        p.update();
        p.draw(ctx);
      });

      // Constellation Web
      drawConstellations();

      // Shooting stars
      shootingStars.forEach((s) => {
        s.update();
        s.draw(ctx);
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    // Init and listeners Handles
    init();
    animate();

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      init();
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };

    const handleMouseLeave = () => {
      mouse.x = null;
      mouse.y = null;
    };

    const handleTouchMove = (e: TouchEvent) => {
      if (e.touches.length > 0) {
        mouse.x = e.touches[0].clientX;
        mouse.y = e.touches[0].clientY;
      }
    };

    const handleTouchEnd = () => {
      mouse.x = null;
      mouse.y = null;
    };

    window.addEventListener('resize', handleResize, { passive: true });
    window.addEventListener('mousemove', handleMouseMove, { passive: true });
    window.addEventListener('mouseleave', handleMouseLeave, { passive: true });
    window.addEventListener('touchmove', handleTouchMove, { passive: true });
    window.addEventListener('touchend', handleTouchEnd, { passive: true });

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseleave', handleMouseLeave);
      window.removeEventListener('touchmove', handleTouchMove);
      window.removeEventListener('touchend', handleTouchEnd);
    };
  }, [intensity, themeColor, emotion]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none z-0"
      style={{ mixBlendMode: 'screen' }}
    />
  );
}
