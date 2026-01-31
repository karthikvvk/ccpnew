import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import SplashSVG from './splash';
import { useTheme } from './ThemeContext';

const AuraFlowLanding = () => {
  const { isDark, toggleTheme } = useTheme();
  const soundWaveRef = useRef(null);
  const iphoneRef = useRef(null);
  const studioRef = useRef(null);
  const heroRef = useRef(null);
  const multiplatformRef = useRef(null);
  const cardsRef = useRef(null);
  const bentoRef = useRef(null);
  const whoRef = useRef(null);
  const [iphoneRotate, setIphoneRotate] = useState({ x: 0, y: 0 });
  const [heroScroll, setHeroScroll] = useState({ rotateX: 0, rotateY: 0, scale: 1 });
  const [studioScroll, setStudioScroll] = useState({ rotateX: 20, rotateY: -15, scale: 0.75 });
  const [multiplatformScroll, setMultiplatformScroll] = useState({ rotateX: 20, rotateY: 15, scale: 0.8 });
  const [cardsScroll, setCardsScroll] = useState({ rotateX: 18, rotateY: -12, scale: 0.85 });
  const [bentoScroll, setBentoScroll] = useState({ rotateX: 15, rotateY: 10, scale: 0.85 });
  const [whoScroll, setWhoScroll] = useState({ rotateX: 20, rotateY: -15, scale: 0.8 });

  useEffect(() => {
    // Intersection Observer for scroll animations
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('opacity-100', 'translate-y-0');
          entry.target.classList.remove('opacity-0', 'translate-y-10');
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.animate-on-scroll').forEach(el => {
      observer.observe(el);
    });

    // Interactive Sound Bars logic (random heights for natural feel)
    const interval = setInterval(() => {
      if (soundWaveRef.current) {
        const bars = soundWaveRef.current.querySelectorAll('.wave-bar');
        bars.forEach(bar => {
          const height = Math.floor(Math.random() * 40) + 10;
          bar.style.height = `${height}px`;
        });
      }
    }, 300);

    // 3D iPhone rotation on mouse move
    const handleMouseMove = (e) => {
      if (!heroRef.current) return;
      const rect = heroRef.current.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      const rotateY = ((e.clientX - centerX) / rect.width) * 20;
      const rotateX = ((centerY - e.clientY) / rect.height) * 15;
      setIphoneRotate({ x: rotateX, y: rotateY });
    };

    const handleMouseLeave = () => {
      setIphoneRotate({ x: 0, y: 0 });
    };

    const heroEl = heroRef.current;
    if (heroEl) {
      heroEl.addEventListener('mousemove', handleMouseMove);
      heroEl.addEventListener('mouseleave', handleMouseLeave);
    }

    // 3D scroll morphing for all sections
    const handleScroll = () => {
      const windowHeight = window.innerHeight;
      const viewportCenter = windowHeight / 2;

      // Helper function to calculate 3D transforms
      const calc3D = (ref, maxRotateX = 20, maxRotateY = 15, minScale = 0.8) => {
        if (!ref.current) return null;
        const rect = ref.current.getBoundingClientRect();
        const center = rect.top + rect.height / 2;
        const distance = Math.abs(center - viewportCenter) / windowHeight;
        return {
          rotateX: distance * maxRotateX,
          rotateY: (center > viewportCenter ? -1 : 1) * distance * maxRotateY,
          scale: Math.max(minScale, 1 - distance * (1 - minScale))
        };
      };

      // Hero section
      const hero = calc3D(heroRef, 15, 12, 0.85);
      if (hero) setHeroScroll(hero);

      // Studio section
      const studio = calc3D(studioRef, 20, 15, 0.75);
      if (studio) setStudioScroll(studio);

      // Multiplatform section
      const multi = calc3D(multiplatformRef, 18, 12, 0.85);
      if (multi) setMultiplatformScroll(multi);

      // Cards section
      const cards = calc3D(cardsRef, 15, 10, 0.88);
      if (cards) setCardsScroll(cards);

      // Bento section
      const bento = calc3D(bentoRef, 12, 8, 0.9);
      if (bento) setBentoScroll(bento);

      // Who section
      const who = calc3D(whoRef, 18, 14, 0.82);
      if (who) setWhoScroll(who);
    };

    window.addEventListener('scroll', handleScroll);
    handleScroll(); // Initial call

    return () => {
      observer.disconnect();
      clearInterval(interval);
      window.removeEventListener('scroll', handleScroll);
      if (heroEl) {
        heroEl.removeEventListener('mousemove', handleMouseMove);
        heroEl.removeEventListener('mouseleave', handleMouseLeave);
      }
    };
  }, []);

  return (
    <div className={`selection:bg-indigo-100 selection:text-indigo-900 font-['Plus_Jakarta_Sans',_sans-serif] overflow-x-hidden transition-colors duration-300 ${isDark ? 'bg-[#161616] text-[#f4f4f4] dark' : 'bg-white text-slate-900'}`}>
      {/* Custom Styles for Glassmorphism and Animations */}
      <style>{`
        .glass-panel {
          background: rgba(255, 255, 255, 0.6);
          backdrop-filter: blur(20px);
          -webkit-backdrop-filter: blur(20px);
          border: 1px solid rgba(255, 255, 255, 0.4);
          box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.07);
        }

        /* Shimmer/Shine effect for buttons */
        @keyframes shimmer {
          0% { transform: translateX(-100%) skewX(-15deg); }
          50% { transform: translateX(200%) skewX(-15deg); }
          100% { transform: translateX(200%) skewX(-15deg); }
        }
        .btn-shimmer::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 50%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
          transform: translateX(-100%) skewX(-15deg);
          animation: shimmer 2.0s ease-in-out infinite;
        }
        /* Shimmer only on hover */
        .btn-shimmer-hover::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          width: 50%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
          transform: translateX(-100%) skewX(-15deg);
        }
        .btn-shimmer-hover:hover::before {
          animation: shimmer 0.8s ease-in-out;
        }

        /* Shake head animation on hover */
        @keyframes shake-head {
          0% { transform: rotate(0deg); }
          20% { transform: rotate(-5deg); }
          40% { transform: rotate(5deg); }
          60% { transform: rotate(-5deg); }
          80% { transform: rotate(5deg); }
          100% { transform: rotate(0deg); }
        }
        .shake-on-hover:hover {
          animation: shake-head 0.5s ease-in-out;
        }

        .text-gradient {
          background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 40%, #0891b2 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }

        .blob-bg {
          position: absolute;
          width: 800px;
          height: 800px;
          filter: blur(80px);
          z-index: -1;
          opacity: 0.4;
          animation: float-blob 20s infinite alternate cubic-bezier(0.45, 0, 0.55, 1);
        }

        @keyframes float-blob {
          0% { transform: translate(-10%, -10%) scale(1) rotate(0deg); }
          100% { transform: translate(15%, 15%) scale(1.3) rotate(180deg); }
        }

        .float-fast { animation: floating 4s infinite ease-in-out; }
        .float-slow { animation: floating 7s infinite ease-in-out; }
        .float-delayed { animation: floating 6s infinite ease-in-out -2s; }

        @keyframes floating {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-25px) rotate(2deg); }
        }

        .gradient-border {
          position: relative;
          background: white;
          padding: 2px;
          border-radius: 2rem;
        }

        .gradient-border::before {
          content: "";
          position: absolute;
          inset: 0;
          border-radius: 2rem;
          padding: 2px; 
          background: linear-gradient(135deg, #6366f1, #06b6d4);
          -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
          -webkit-mask-composite: xor;
          mask-composite: exclude;
          pointer-events: none;
        }

        /* Float Animations for shim(2) style */
        .float-1 { animation: float-anim 6s infinite ease-in-out; }
        .float-2 { animation: float-anim 8s infinite ease-in-out -2s; }
        .float-3 { animation: float-anim 7s infinite ease-in-out -4s; }

        @keyframes float-anim {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-20px); }
        }

        /* Wave Container */
        .wave-container {
          display: flex;
          align-items: flex-end;
          gap: 3px;
          height: 40px;
        }

        .wave-bar {
          animation: wave-play 0.8s infinite ease-in-out alternate;
        }

        @keyframes wave-play {
          0% { height: 10px; }
          100% { height: 40px; }
        }

        /* Perspective Cards with smoother transitions */
        .perspective-container {
          perspective: 2000px;
          transform-style: preserve-3d;
        }
        .tilt-card {
          transform: rotateY(-8deg) rotateX(5deg);
          transition: all 1.2s cubic-bezier(0.25, 0.46, 0.45, 0.94);
          will-change: transform;
        }
        .tilt-card:hover {
          transform: rotateY(0deg) rotateX(0deg) scale(1.03);
        }

        /* Smooth scroll behavior */
        html {
          scroll-behavior: smooth;
        }

        /* Enhanced smooth transitions */
        .smooth-appear {
          transition: all 1s cubic-bezier(0.4, 0, 0.2, 1);
        }

        /* Smoother float animations */
        @keyframes smooth-float {
          0%, 100% { transform: translateY(0) rotate(0deg); }
          25% { transform: translateY(-10px) rotate(1deg); }
          50% { transform: translateY(-20px) rotate(0deg); }
          75% { transform: translateY(-10px) rotate(-1deg); }
        }
        .smooth-float { animation: smooth-float 8s infinite ease-in-out; }

        /* Glass style from shim(2) */
        .glass {
          background: rgba(255, 255, 255, 0.7);
          backdrop-filter: blur(14px);
          -webkit-backdrop-filter: blur(14px);
          border: 1px solid rgba(255, 255, 255, 0.4);
        }

        /* Blob from shim(2) */
        .blob {
          position: absolute;
          filter: blur(80px);
          z-index: -1;
          border-radius: 50%;
          opacity: 0.45;
          animation: blob-float 25s infinite alternate ease-in-out;
        }

        @keyframes blob-float {
          0% { transform: translate(0, 0) scale(1) rotate(0deg); }
          33% { transform: translate(100px, -50px) scale(1.2) rotate(120deg); }
          66% { transform: translate(-50px, 100px) scale(0.9) rotate(240deg); }
          100% { transform: translate(0, 0) scale(1) rotate(360deg); }
        }

        /* Dark Mode Overrides */
        .dark .glass-panel {
          background: rgba(38, 38, 38, 0.6) !important;
          border: 1px solid rgba(255, 255, 255, 0.1) !important;
          box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4) !important;
        }
        .dark .glass {
          background: rgba(38, 38, 38, 0.6) !important;
          border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        .dark .text-slate-900 { color: #f4f4f4 !important; }
        .dark .text-slate-500 { color: #c6c6c6 !important; }
        .dark .text-slate-600 { color: #e0e0e0 !important; }
        .dark .bg-slate-900 { background-color: #f4f4f4 !important; color: #161616 !important; } /* Invert buttons */
        .dark .bg-slate-900.text-white, 
        .dark .bg-slate-900 .text-white {
          color: #161616 !important;
        }
        .dark .text-slate-800 { color: #f4f4f4 !important; } /* Fix hero card text */
        .dark .bg-white { background-color: #161616 !important; }
        .dark .bg-slate-50 { background-color: #161616 !important; }
        .dark .bg-slate-50\/50 { background-color: #161616 !important; }
        .dark nav .bg-slate-900 { background-color: #f4f4f4 !important; color: #161616 !important; }
        .dark .shadow-slate-200 { --tw-shadow-color: #000; }
      `}</style>

      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-[100] p-6 pointer-events-none">
        <div className="max-w-7xl mx-auto flex items-center justify-between pointer-events-auto">
          <div className="glass-panel px-1.5 py-1.5 rounded-full transition-all duration-1000 opacity-100 translate-y-0">
            <Link to="/" className="btn-shimmer shake-on-hover relative px-6 py-2.5 bg-slate-900 rounded-full flex items-center gap-3 transition-all duration-500 shadow-xl shadow-slate-200 overflow-hidden">
              <span className="text-xl font-extrabold tracking-tighter text-white relative z-10">Aura Flow</span>
            </Link>
          </div>

          <div className="hidden lg:flex items-center gap-1 glass-panel px-2 py-1.5 rounded-full transition-all duration-1000 opacity-100 translate-y-0">
            <a href="#how" className="px-5 py-2 text-sm font-semibold text-slate-600 hover:text-indigo-600 transition-colors">How it works</a>
            <a href="#features" className="px-5 py-2 text-sm font-semibold text-slate-600 hover:text-indigo-600 transition-colors">Features</a>
            <a href="#creators" className="px-5 py-2 text-sm font-semibold text-slate-600 hover:text-indigo-600 transition-colors">For Creators</a>
          </div>

          <div className="glass-panel px-1.5 py-1.5 rounded-full transition-all duration-1000 opacity-100 translate-y-0 flex items-center gap-2">
            <button
              onClick={toggleTheme}
              className="relative w-14 h-9 rounded-full bg-slate-200 dark:bg-slate-800 transition-colors duration-300 flex items-center px-1 border border-slate-300 dark:border-slate-600 hover:border-indigo-500/50 focus:outline-none overflow-hidden"
              aria-label="Toggle Theme"
            >
              <div
                className={`absolute w-7 h-7 rounded-full bg-white dark:bg-[#020617] shadow-sm flex items-center justify-center transition-transform duration-300 overflow-hidden ${isDark ? 'translate-x-[20px]' : 'translate-x-0'}`}
              >
                <span className={`material-symbols-outlined text-[16px] absolute transition-all duration-300 ${isDark ? 'text-indigo-400 opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-50'}`}>
                  dark_mode
                </span>
                <span className={`material-symbols-outlined text-[16px] absolute text-amber-500 transition-all duration-300 ${isDark ? 'opacity-0 rotate-90 scale-50' : 'opacity-100 rotate-0 scale-100'}`}>
                  light_mode
                </span>
              </div>
            </button>
            <button className="bg-slate-900 text-white px-6 py-2.5 rounded-full text-sm font-bold hover:bg-indigo-600 transition-all shadow-xl shadow-slate-200 dark:bg-white dark:text-black dark:hover:bg-gray-200">
              Join Waitlist
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative min-h-screen pt-32 pb-12 overflow-hidden flex flex-col items-center">
        {/* Animated Background Blobs */}
        <div className="blob-bg top-0 -left-20 bg-indigo-200 dark:bg-indigo-900/30"></div>
        <div className="blob-bg bottom-0 -right-20 bg-cyan-100 dark:bg-cyan-900/30" style={{ animationDelay: '-10s' }}></div>
        <div className="blob-bg top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-violet-100 dark:bg-violet-900/20 opacity-20"></div>

        <div className="container max-w-7xl mx-auto px-6 grid lg:grid-cols-12 gap-12 items-center relative z-10 mt-4">

          {/* Left Content */}
          <div className="lg:col-span-6 space-y-10 text-center lg:text-left">


            <h1 className="text-6xl md:text-6xl font-black tracking-tight leading-[0.95] text-slate-900">
              One Voice.<br />
              <span className="text-gradient">Every Language.</span>
            </h1>

            <p className="text-xl md:text-2xl text-slate-500 font-medium max-w-xl mx-auto lg:mx-0 leading-relaxed">
              Instantly clone your voice for video translation. Reach billions of viewers in their native tongue without recording a single extra word.
            </p>

            <div className="flex flex-col sm:flex-row items-center gap-4 justify-center lg:justify-start">
              <Link to="/how-it-works" className="btn-shimmer-hover group relative px-10 py-5 bg-slate-900 text-white rounded-2xl font-bold text-xl overflow-hidden transition-all duration-500 hover:scale-105 shadow-2xl shadow-slate-400/30 cursor-pointer flex items-center gap-3">
                <span className="material-symbols-outlined text-white relative z-10">auto_awesome</span>
                <span className="relative z-10">See Magic</span>
              </Link>
            </div>

            <div className="flex items-center justify-center lg:justify-start gap-8 pt-6 border-t border-slate-100">
              <div>
                <p className="text-3xl font-black text-slate-900">40+</p>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Languages</p>
              </div>
              <div className="w-[1px] h-10 bg-slate-200"></div>
              <div>
                <p className="text-3xl font-black text-slate-900">99%</p>
                <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Voice Accuracy</p>
              </div>
            </div>
          </div>

          {/* Right Visual Scene */}
          <div
            ref={heroRef}
            className="lg:col-span-6 relative h-[600px] flex items-center justify-center overflow-visible cursor-pointer transition-transform duration-500 ease-out"
            style={{ perspective: '1200px', transform: `scale(${heroScroll.scale})` }}
          >

            {/* Splash SVG Background with Gradient */}
            <svg width="0" height="0" className="absolute">
              <defs>
                <linearGradient id="splashGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#f472b6" />
                  <stop offset="25%" stopColor="#e879f9" />
                  <stop offset="50%" stopColor="#a78bfa" />
                  <stop offset="75%" stopColor="#818cf8" />
                  <stop offset="100%" stopColor="#60a5fa" />
                </linearGradient>
              </defs>
            </svg>
            <SplashSVG
              className="absolute scale-[1.3] -translate-y-[50px] opacity-100 transition-transform duration-150 ease-out pointer-events-none"
              fill="url(#splashGradient)"
              style={{ transform: `rotateX(${heroScroll.rotateX + iphoneRotate.x * 0.5}deg) rotateY(${heroScroll.rotateY + iphoneRotate.y * 0.5}deg)` }}
            />

            {/* Main Image Card - iPhone Frame */}
            <div
              ref={iphoneRef}
              className="relative z-30 transition-transform duration-150 ease-out opacity-100 -translate-y-8 w-full max-w-[260px] group"
              style={{ transform: `rotateX(${heroScroll.rotateX + iphoneRotate.x}deg) rotateY(${heroScroll.rotateY + iphoneRotate.y}deg)` }}
            >
              {/* iPhone Frame */}
              <div className="relative bg-[#0f172a] rounded-[2.5rem] p-[8px] shadow-2xl shadow-slate-900/50">
                {/* iPhone Side Buttons */}
                <div className="absolute -left-[3px] top-20 w-[3px] h-6 bg-slate-700 rounded-l-sm"></div>
                <div className="absolute -left-[3px] top-28 w-[3px] h-10 bg-slate-700 rounded-l-sm"></div>
                <div className="absolute -left-[3px] top-40 w-[3px] h-10 bg-slate-700 rounded-l-sm"></div>
                <div className="absolute -right-[3px] top-28 w-[3px] h-14 bg-slate-700 rounded-r-sm"></div>

                {/* iPhone Screen */}
                <div className="relative rounded-[2rem] overflow-hidden aspect-[9/19.5] bg-black">
                  {/* Dynamic Island */}
                  <div className="absolute top-3 left-1/2 -translate-x-1/2 z-50 bg-black w-[100px] h-[28px] rounded-full flex items-center justify-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-slate-800"></div>
                    <div className="w-3 h-3 rounded-full bg-slate-800 ring-1 ring-slate-700"></div>
                  </div>

                  {/* Screen Content */}
                  <div className="relative w-full h-full">
                    <img src="/user.png" alt="Main Creator" className="w-full h-full object-cover object-top scale-105" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent"></div>
                    <div className="absolute bottom-12 left-0 right-0 px-4 text-center">
                      <div className="glass-panel py-3 px-4 rounded-xl inline-block animate-on-scroll opacity-0 translate-y-10 group-hover:bg-white transition-all">
                        <p className="text-sm font-bold text-slate-900">Hello, I'm reaching you globally!</p>
                      </div>
                    </div>

                    {/* iPhone Home Indicator */}
                    <div className="absolute bottom-2 left-1/2 -translate-x-1/2 w-[120px] h-[4px] bg-white/80 rounded-full"></div>
                  </div>
                </div>
              </div>
            </div>

            {/* Floating Language Cards */}
            <div
              className="absolute -left-20 top-0 z-20 float-slow w-48 hidden md:block transition-transform duration-150 ease-out"
              style={{ transform: `rotateX(${heroScroll.rotateX * 0.7 + iphoneRotate.x * 0.7}deg) rotateY(${heroScroll.rotateY * 0.7 + iphoneRotate.y * 0.7}deg)` }}
            >
              <div className="glass-panel p-2 rounded-3xl shadow-2xl rotate-[-10deg]">
                <img src="https://images.unsplash.com/photo-1544005313-94ddf0286df2?q=80&w=400&auto=format&fit=crop" alt="Spanish Version" className="w-full aspect-square rounded-2xl object-cover mb-3" />
                <div className="flex items-center gap-2 px-2 pb-1">
                  <span className="text-lg">🇪🇸</span>
                  <span className="text-xs font-bold text-slate-800 tracking-tight">¡Hola a todos!</span>
                </div>
              </div>
            </div>

            <div
              className="absolute -right-16 bottom-10 z-20 float-delayed w-48 hidden md:block transition-transform duration-150 ease-out"
              style={{ transform: `rotateX(${heroScroll.rotateX * 0.7 + iphoneRotate.x * 0.7}deg) rotateY(${heroScroll.rotateY * 0.7 + iphoneRotate.y * 0.7}deg)` }}
            >
              <div className="glass-panel p-2 rounded-3xl shadow-2xl rotate-[8deg]">
                <img src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=400&auto=format&fit=crop" alt="Japanese Version" className="w-full aspect-square rounded-2xl object-cover mb-3" />
                <div className="flex items-center gap-2 px-2 pb-1">
                  <span className="text-lg">🇯🇵</span>
                  <span className="text-xs font-bold text-slate-800 tracking-tight">こんにちは、世界</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* AI Dubbing Studio Section */}
      <section className={`py-32 relative overflow-hidden transition-colors duration-300 ${isDark ? 'bg-gradient-to-b from-[#161616] to-[#121212]' : 'bg-gradient-to-b from-white to-slate-50/50'}`} id="how">
        <div className="container max-w-7xl mx-auto px-6">
          {/* Section Header */}
          <div className="text-center mb-16 space-y-4 animate-on-scroll opacity-0 translate-y-12">
            <p className="text-indigo-600 font-bold uppercase tracking-widest text-sm">Powerful Studio</p>
            <h2 className="text-4xl md:text-5xl font-black text-slate-900">Your AI Dubbing Command Center</h2>
            <p className="text-lg text-slate-500 font-medium max-w-2xl mx-auto">Upload once, generate multiple audio tracks in different languages with your exact voice signature.</p>
          </div>

          <div
            ref={studioRef}
            className="relative max-w-5xl mx-auto h-[650px] transition-transform duration-300 ease-out"
            style={{ perspective: '2000px', transformStyle: 'preserve-3d' }}
          >
            {/* Main Mockup */}
            <div
              className="absolute inset-0 flex items-center justify-center z-20 transition-transform duration-500 ease-out"
              style={{ transform: `rotateX(${studioScroll.rotateX}deg) rotateY(${studioScroll.rotateY}deg) scale(${studioScroll.scale})`, transformStyle: 'preserve-3d' }}
            >
              <div className="w-full max-w-2xl rounded-[3rem] overflow-hidden glass p-4 shadow-2xl shadow-indigo-100/50 border-white transition-all duration-1000 animate-on-scroll opacity-0 translate-y-12">
                <img src="/barrier.png" alt="AI Dubbing Dashboard" className="rounded-[2rem] w-full shadow-inner object-cover object-top scale-120" />
              </div>
            </div>

            {/* Floating Audio Track Cards */}
            <div
              className="absolute -left-10 top-0 z-30 float-1 hidden lg:block transition-transform duration-500 ease-out"
              style={{ transform: `rotateX(${studioScroll.rotateX * 0.7}deg) rotateY(${studioScroll.rotateY * 0.7}deg) translateZ(80px) scale(${studioScroll.scale})`, transformStyle: 'preserve-3d' }}
            >
              <div className="glass p-3 rounded-[2rem] shadow-2xl w-48 border-l-4 border-indigo-400 animate-on-scroll opacity-0 translate-y-12">
                <img src="/clar.png" alt="Original" className="rounded-2xl w-full mb-3 object-cover object-top scale-105" />
                <div className="flex items-center gap-2 px-2">
                  <span className="text-xl">🇫🇷</span>
                  <span className="text-xs font-bold text-slate-800">Bonjour le monde!</span>
                </div>
              </div>
            </div>

            <div
              className="absolute -right-12 top-1/4 z-30 float-2 hidden lg:block transition-transform duration-500 ease-out"
              style={{ transform: `rotateX(${studioScroll.rotateX * 0.7}deg) rotateY(${studioScroll.rotateY * 0.7}deg) translateZ(60px) scale(${studioScroll.scale})`, transformStyle: 'preserve-3d' }}
            >
              <div className="glass p-3 rounded-[2rem] shadow-2xl w-48 border-l-4 border-cyan-400 animate-on-scroll opacity-0 translate-y-12">
                <img src="/pep.png" alt="Portuguese" className="rounded-2xl w-full mb-3" />
                <div className="flex items-center gap-2 px-2">
                  <span className="text-xl">🇧🇷</span>
                  <span className="text-xs font-bold text-slate-800">Olá mundo!</span>
                </div>
              </div>
            </div>

            <div
              className="absolute left-1/4 -bottom-10 z-40 float-3 hidden lg:block transition-transform duration-500 ease-out"
              style={{ transform: `rotateX(${studioScroll.rotateX * 0.5}deg) rotateY(${studioScroll.rotateY * 0.5}deg) translateZ(100px) scale(${studioScroll.scale})`, transformStyle: 'preserve-3d' }}
            >
              <div className="glass p-4 rounded-3xl shadow-xl flex items-center gap-4 animate-on-scroll opacity-0 translate-y-12">
                <div className="wave-container">
                  <div className="wave-bar bg-gradient-to-t from-indigo-600 to-cyan-500 rounded-full w-[3px]" style={{ animationDelay: '0.1s', height: '26px' }}></div>
                  <div className="wave-bar bg-gradient-to-t from-indigo-600 to-cyan-500 rounded-full w-[3px]" style={{ animationDelay: '0.3s', height: '30px' }}></div>
                  <div className="wave-bar bg-gradient-to-t from-indigo-600 to-cyan-500 rounded-full w-[3px]" style={{ animationDelay: '0.2s', height: '31px' }}></div>
                  <div className="wave-bar bg-gradient-to-t from-indigo-600 to-cyan-500 rounded-full w-[3px]" style={{ animationDelay: '0.5s', height: '33px' }}></div>
                  <div className="wave-bar bg-gradient-to-t from-indigo-600 to-cyan-500 rounded-full w-[3px]" style={{ animationDelay: '0.4s', height: '36px' }}></div>
                </div>
                <div className="text-left pr-4">
                  <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest">Multi-Agent AI</p>
                  <p className="text-sm font-bold text-slate-800">Parallel Processing</p>
                </div>
              </div>
            </div>

            {/* Abstract Decorations */}
            <div className="absolute -top-10 left-1/2 -translate-x-1/2 w-[120%] h-[120%] border border-slate-200/50 rounded-full -z-10 pointer-events-none opacity-20"></div>
            <div className="absolute -top-20 left-1/2 -translate-x-1/2 w-[150%] h-[150%] border border-slate-200/30 rounded-full -z-10 pointer-events-none opacity-10"></div>
          </div>
        </div>
      </section>

      {/* Multiplatform Publishing Section */}
      <section className={`py-32 relative overflow-hidden transition-colors duration-300 ${isDark ? 'bg-[#161616]' : 'bg-white'}`} id="creators">
        <div className="container max-w-7xl mx-auto px-6 grid lg:grid-cols-2 gap-20 items-center">
          <div className="space-y-10 animate-on-scroll opacity-0 translate-y-12">
            <h2 className="text-5xl font-black tracking-tight text-slate-900 leading-tight">
              Publish Everywhere.<br />
              <span className="text-indigo-600">Connect Globally.</span>
            </h2>
            <p className="text-xl text-slate-500 leading-relaxed">
              Seamlessly distribute your multilingual content across YouTube, TikTok, Instagram, LinkedIn, and Facebook. One upload, infinite reach across every major platform.
            </p>
            <div className="grid gap-6">
              <div className="flex items-start gap-4 p-6 glass rounded-3xl shadow-sm border-indigo-50 animate-on-scroll opacity-0 translate-y-12">
                <div className="bg-indigo-600 text-white p-3 rounded-2xl">
                  <span className="material-symbols-outlined">share</span>
                </div>
                <div>
                  <h4 className="font-bold text-slate-900 text-lg">Cross-Platform Sync</h4>
                  <p className="text-slate-500">Auto-format and publish to Instagram Reels, YouTube Shorts, TikTok, and LinkedIn simultaneously.</p>
                </div>
              </div>
              <div className="flex items-start gap-4 p-6 glass rounded-3xl shadow-sm border-cyan-50 animate-on-scroll opacity-0 translate-y-12">
                <div className="bg-cyan-500 text-white p-3 rounded-2xl">
                  <span className="material-symbols-outlined">hub</span>
                </div>
                <div>
                  <h4 className="font-bold text-slate-900 text-lg">Multi-Agent Technology</h4>
                  <p className="text-slate-500">Our AI agents work in parallel to translate, dub, and optimize your content for each platform's algorithm.</p>
                </div>
              </div>
            </div>
          </div>
          <div
            ref={multiplatformRef}
            className="grid grid-cols-2 gap-6 relative transition-transform duration-500 ease-out"
            style={{ perspective: '2000px', transform: `perspective(2000px) rotateX(${multiplatformScroll.rotateX}deg) rotateY(${multiplatformScroll.rotateY}deg) scale(${multiplatformScroll.scale})`, transformStyle: 'preserve-3d' }}
          >
            <div className="space-y-6 pt-12" style={{ transform: 'translateZ(30px)', transformStyle: 'preserve-3d' }}>
              <div className="relative rounded-3xl overflow-hidden shadow-2xl transform -rotate-3 hover:rotate-0 transition-all" style={{ transform: 'translateZ(20px)' }}>
                <img src="/inst.png" alt="Creator 1" className="w-full h-72 object-cover" />
                <div className="absolute top-4 left-4 glass px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest flex items-center gap-1.5">
                  <span className="text-slate-500">JP</span>
                  <span className="text-slate-800">JAPANESE</span>
                </div>
              </div>
              <div className="relative rounded-3xl overflow-hidden shadow-2xl transform rotate-2 hover:rotate-0 transition-all">
                <img src="/c3.png" alt="Creator 2" className="w-full h-72 object-cover" />
                <div className="absolute top-4 left-4 glass px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest flex items-center gap-1.5">
                  <span className="text-slate-500">US</span>
                  <span className="text-slate-800">ENGLISH</span>
                </div>
              </div>
            </div>
            <div className="space-y-6" style={{ transform: 'translateZ(50px)', transformStyle: 'preserve-3d' }}>
              <div className="relative rounded-3xl overflow-hidden shadow-2xl transform rotate-6 hover:rotate-0 transition-all" style={{ transform: 'translateZ(25px)' }}>
                <img src="/french.png" alt="Creator 3" className="w-full h-72 object-cover" />
                <div className="absolute top-4 left-4 glass px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest flex items-center gap-1.5">
                  <span className="text-slate-500">FR</span>
                  <span className="text-slate-800">FRENCH</span>
                </div>
              </div>
              <div className="relative rounded-3xl overflow-hidden shadow-2xl transform -rotate-2 hover:rotate-0 transition-all">
                <img src="/c4.png" alt="Creator 4" className="w-full h-72 object-cover" />
                <div className="absolute top-4 left-4 glass px-3 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest flex items-center gap-1.5">
                  <span className="text-slate-500">ES</span>
                  <span className="text-slate-800">SPANISH</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Visual Storytelling - Cards */}
      <section className={`py-32 relative overflow-hidden transition-colors duration-300 ${isDark ? 'bg-[#161616]' : 'bg-slate-50/50'}`}>
        <div className="container max-w-7xl mx-auto px-6">
          <div className="text-center mb-24 space-y-4">
            <h2 className="text-4xl md:text-6xl font-black text-slate-900">Your Content, Reimagined.</h2>
            <p className="text-lg text-slate-500 font-medium">Capture once. Publish in forty languages.</p>
          </div>

          <div
            ref={cardsRef}
            className="grid md:grid-cols-3 gap-10 transition-transform duration-500 ease-out"
            style={{ perspective: '2000px', transform: `perspective(2000px) rotateX(${cardsScroll.rotateX}deg) rotateY(${cardsScroll.rotateY}deg) scale(${cardsScroll.scale})`, transformStyle: 'preserve-3d' }}
          >
            <div className="group relative h-[500px] rounded-[3rem] overflow-hidden shadow-2xl transition-all hover:-translate-y-4" style={{ transform: 'translateZ(40px)', transformStyle: 'preserve-3d' }}>
              <img src="https://images.unsplash.com/photo-1516035069371-29a1b244cc32?q=80&w=800&auto=format&fit=crop" alt="Original Persona" className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-t from-slate-900/90 via-slate-900/20 to-transparent"></div>
              <div className="absolute bottom-0 p-10 text-white">
                <span className="material-symbols-outlined text-4xl text-indigo-400 mb-4">person_play</span>
                <h3 className="text-2xl font-extrabold mb-2">Original Persona</h3>
                <p className="text-slate-300 text-sm leading-relaxed">We keep your original facial expressions and tone, ensuring you remain authentic to your brand.</p>
              </div>
            </div>

            <div className="group relative h-[500px] rounded-[3rem] overflow-hidden shadow-2xl transition-all hover:-translate-y-4 md:mt-10" style={{ transform: 'translateZ(60px)', transformStyle: 'preserve-3d' }}>
              <img src="/trans.png" alt="Language Flow" className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-t from-violet-900/90 via-violet-900/20 to-transparent"></div>
              <div className="absolute bottom-0 p-10 text-white">
                <span className="material-symbols-outlined text-4xl text-violet-400 mb-4">translate</span>
                <h3 className="text-2xl font-extrabold mb-2">Infinite Translation</h3>
                <p className="text-slate-300 text-sm leading-relaxed">Our AI translates with context-awareness, handling slang and cultural references like a native.</p>
              </div>
            </div>

            <div className="group relative h-[500px] rounded-[3rem] overflow-hidden shadow-2xl transition-all hover:-translate-y-4" style={{ transform: 'translateZ(30px)', transformStyle: 'preserve-3d' }}>
              <img src="/audi.png" alt="Global Reach" className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-t from-cyan-900/90 via-cyan-900/20 to-transparent"></div>
              <div className="absolute bottom-0 p-10 text-white">
                <span className="material-symbols-outlined text-4xl text-cyan-400 mb-4">public</span>
                <h3 className="text-2xl font-extrabold mb-2">Limitless Audience</h3>
                <p className="text-slate-300 text-sm leading-relaxed">From São Paulo to Seoul, your videos will find their fans without language being a wall.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Bento Grid Features */}
      <section className="py-32 px-6" id="features">
        <div className="container max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row items-end justify-between gap-8 mb-20">
            <div className="max-w-2xl space-y-4">
              <p className="text-indigo-600 font-bold uppercase tracking-widest text-sm">Key Features</p>
              <h2 className="text-4xl md:text-5xl font-black text-slate-900 leading-tight">Advanced AI that works<br />as hard as you do.</h2>
            </div>
            <button className="px-8 py-4 glass-panel text-indigo-600 rounded-2xl font-bold flex items-center gap-2 hover:bg-indigo-50 transition-all animate-on-scroll opacity-0 translate-y-10">
              Explore all Features
              <span className="material-symbols-outlined">arrow_forward</span>
            </button>
          </div>

          <div
            ref={bentoRef}
            className="grid grid-cols-1 md:grid-cols-3 gap-6 transition-transform duration-500 ease-out"
            style={{ perspective: '2000px', transform: `perspective(2000px) rotateX(${bentoScroll.rotateX}deg) rotateY(${bentoScroll.rotateY}deg) scale(${bentoScroll.scale})`, transformStyle: 'preserve-3d' }}
          >
            <div className="md:col-span-2 glass-panel p-10 rounded-[3rem] relative overflow-hidden group transition-all animate-on-scroll opacity-0 translate-y-10" style={{ transform: 'translateZ(50px)', transformStyle: 'preserve-3d' }}>
              <div className="relative z-10 space-y-6">
                <div className="w-16 h-16 bg-indigo-600 rounded-2xl flex items-center justify-center text-white shadow-xl">
                  <span className="material-symbols-outlined text-3xl">radio_button_unchecked</span>
                </div>
                <h3 className="text-3xl font-extrabold text-slate-900">Zero-Shot Voice Cloning</h3>
                <p className="text-slate-500 max-w-md">No need for hours of recording. Upload just 30 seconds of video, and AuraFlow generates a perfect digital twin of your voice in any language.</p>
              </div>
              <div className="absolute right-0 bottom-0 w-1/2 h-full opacity-10 group-hover:opacity-20 transition-opacity translate-y-20">
                <span className="material-symbols-outlined text-[300px]">waves</span>
              </div>
            </div>

            <div className="glass-panel p-10 rounded-[3rem] group hover:bg-white transition-all animate-on-scroll opacity-0 translate-y-10" style={{ transform: 'translateZ(30px)', transformStyle: 'preserve-3d' }}>
              <span className="material-symbols-outlined text-4xl text-cyan-500 mb-6">subtitles</span>
              <h3 className="text-2xl font-extrabold text-slate-900 mb-4">Smart Subtitles</h3>
              <p className="text-slate-500">Subtitles are not just translated; they are natively localized to fit the rhythm and cadence of the speech.</p>
            </div>

            <div className="glass-panel p-10 rounded-[3rem] group hover:bg-white transition-all animate-on-scroll opacity-0 translate-y-10" style={{ transform: 'translateZ(20px)', transformStyle: 'preserve-3d' }}>
              <span className="material-symbols-outlined text-4xl text-violet-500 mb-6">bolt</span>
              <h3 className="text-2xl font-extrabold text-slate-900 mb-4">Lightning Rendering</h3>
              <p className="text-slate-500">Our cloud processing handles 4K videos in minutes, not hours. Perfect for short-form creators on a schedule.</p>
            </div>

            <div className="glass-panel p-10 rounded-[3rem] group hover:bg-white transition-all animate-on-scroll opacity-0 translate-y-10" style={{ transform: 'translateZ(40px)', transformStyle: 'preserve-3d' }}>
              <span className="material-symbols-outlined text-4xl text-amber-500 mb-6">check_circle</span>
              <h3 className="text-2xl font-extrabold text-slate-900 mb-4">Lip-Sync Ready</h3>
              <p className="text-slate-500">Experimental lip-sync coming soon to perfectly match your translated speech with facial movements.</p>
            </div>

            <div className="glass-panel p-10 rounded-[3rem] group hover:bg-white transition-all animate-on-scroll opacity-0 translate-y-10" style={{ transform: 'translateZ(35px)', transformStyle: 'preserve-3d' }}>
              <span className="material-symbols-outlined text-4xl text-emerald-500 mb-6">auto_awesome_motion</span>
              <h3 className="text-2xl font-extrabold text-slate-900 mb-4">Multi-Format Export</h3>
              <p className="text-slate-500">Directly export for TikTok, Reels, or Shorts with pre-formatted aspect ratios and burnt-in captions.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Who It's For Section */}
      <section className={`py-32 px-6 overflow-hidden relative transition-colors duration-300 ${isDark ? 'bg-[#161616]' : 'bg-slate-50'}`}>
        <div className="blob absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-indigo-200 opacity-30"></div>
        <div className="max-w-7xl mx-auto space-y-20 relative z-10">
          <div className="text-center space-y-4 animate-on-scroll opacity-0 translate-y-12">
            <h2 className="text-4xl md:text-5xl font-extrabold tracking-tight text-slate-900">Built for the next generation of storytellers.</h2>
            <p className="text-slate-500 text-lg">AuraFlow scales your personality across borders.</p>
          </div>

          <div
            ref={whoRef}
            className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 transition-transform duration-500 ease-out"
            style={{ perspective: '2000px', transform: `perspective(2000px) rotateX(${whoScroll.rotateX}deg) rotateY(${whoScroll.rotateY}deg) scale(${whoScroll.scale})`, transformStyle: 'preserve-3d' }}
          >
            {/* YouTubers */}
            <div className="group relative overflow-hidden rounded-[2rem] h-[400px] shadow-lg animate-on-scroll opacity-0 translate-y-12" style={{ transform: 'translateZ(40px)', transformStyle: 'preserve-3d' }}>
              <img src="/y1.png" alt="YouTubers" className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
              <div className="absolute bottom-0 left-0 p-8 text-white space-y-2">
                <h4 className="text-2xl font-bold">YouTubers</h4>
                <p className="text-sm opacity-80">Localize your shorts and reach 10x more viewers.</p>
              </div>
            </div>

            {/* Influencers */}
            <div className="group relative overflow-hidden rounded-[2rem] h-[400px] shadow-lg animate-on-scroll opacity-0 translate-y-12" style={{ transitionDelay: '100ms', transform: 'translateZ(60px)', transformStyle: 'preserve-3d' }}>
              <img src="/in.png" alt="Influencers" className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
              <div className="absolute bottom-0 left-0 p-8 text-white space-y-2">
                <h4 className="text-2xl font-bold">Influencers</h4>
                <p className="text-sm opacity-80">Build a global community without learning new languages.</p>
              </div>
            </div>

            {/* Educators */}
            <div className="group relative overflow-hidden rounded-[2rem] h-[400px] shadow-lg animate-on-scroll opacity-0 translate-y-12" style={{ transitionDelay: '200ms', transform: 'translateZ(50px)', transformStyle: 'preserve-3d' }}>
              <img src="/edu.png" alt="Educators" className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
              <div className="absolute bottom-0 left-0 p-8 text-white space-y-2">
                <h4 className="text-2xl font-bold">Educators</h4>
                <p className="text-sm opacity-80">Make knowledge accessible to students worldwide.</p>
              </div>
            </div>

            {/* Global Brands */}
            <div className="group relative overflow-hidden rounded-[2rem] h-[400px] shadow-lg animate-on-scroll opacity-0 translate-y-12" style={{ transitionDelay: '300ms', transform: 'translateZ(35px)', transformStyle: 'preserve-3d' }}>
              <img src="/bis.png" alt="Global Brands" className="absolute inset-0 w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-black/20 to-transparent"></div>
              <div className="absolute bottom-0 left-0 p-8 text-white space-y-2">
                <h4 className="text-2xl font-bold">Global Brands</h4>
                <p className="text-sm opacity-80">Consistent communication across all international offices.</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className={`py-20 border-t transition-colors duration-300 ${isDark ? 'bg-[#161616] border-gray-800' : 'bg-white border-slate-100'}`}>
        <div className="container max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-4 gap-12 mb-16">
            <div className="col-span-2 space-y-6">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-slate-900 rounded-lg flex items-center justify-center">
                  <span className="material-symbols-outlined text-white text-lg">radio_button_unchecked</span>
                </div>
                <span className="text-2xl font-black tracking-tighter">AUREX</span>
              </div>
              <p className="text-slate-500 max-w-sm">Leading the revolution in creator accessibility. Our mission is to break down language barriers through human-centric AI.</p>
            </div>
            <div>
              <h4 className="font-bold text-slate-900 mb-6">Product</h4>
              <ul className="space-y-4 text-slate-500 text-sm">
                <li><a href="#" className="hover:text-indigo-600 transition-colors">Voice Cloning</a></li>
                <li><a href="#" className="hover:text-indigo-600 transition-colors">Video Translation</a></li>
                <li><a href="#" className="hover:text-indigo-600 transition-colors">Subtitle Generator</a></li>
                <li><a href="#" className="hover:text-indigo-600 transition-colors">API Access</a></li>
              </ul>
            </div>
            <div>
              <h4 className="font-bold text-slate-900 mb-6">Company</h4>
              <ul className="space-y-4 text-slate-500 text-sm">
                <li><a href="#" className="hover:text-indigo-600 transition-colors">About Aurex</a></li>
                <li><a href="#" className="hover:text-indigo-600 transition-colors">Privacy Policy</a></li>
                <li><a href="#" className="hover:text-indigo-600 transition-colors">Terms of Use</a></li>
                <li><a href="#" className="hover:text-indigo-600 transition-colors">Support</a></li>
              </ul>
            </div>
          </div>
          <div className="pt-12 border-t border-slate-100 flex flex-col md:flex-row items-center justify-between gap-6">
            <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">© 2024 AUREX TECHNOLOGIES. LONDON, UK.</p>
            <div className="flex items-center gap-6">
              <a href="#" className="w-10 h-10 rounded-full glass-panel flex items-center justify-center hover:bg-indigo-600 hover:text-white transition-all animate-on-scroll opacity-0 translate-y-10">
                <span className="material-symbols-outlined text-xl">alternate_email</span>
              </a>
              <a href="#" className="w-10 h-10 rounded-full glass-panel flex items-center justify-center hover:bg-indigo-600 hover:text-white transition-all animate-on-scroll opacity-0 translate-y-10">
                <span className="material-symbols-outlined text-xl">public</span>
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default AuraFlowLanding;