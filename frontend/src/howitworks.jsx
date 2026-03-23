import React, { Suspense, useRef, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Points, PointMaterial, Float } from '@react-three/drei';
import * as THREE from 'three';
import { motion, useScroll, useTransform } from 'framer-motion';
import { useTheme } from './ThemeContext';

// --- Ultra-Smooth Interactive Particle Field ---
function ParticleField() {
  const ref = useRef();
  const { mouse, viewport } = useThree();

  const count = 8000;
  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const cols = new Float32Array(count * 3);
    const palette = [
      new THREE.Color('#6366f1'), // Indigo
      new THREE.Color('#ec4899'), // Pink
      new THREE.Color('#06b6d4'), // Cyan
      new THREE.Color('#f59e0b'), // Amber
    ];

    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 18;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 18;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 18;

      const color = palette[Math.floor(Math.random() * palette.length)];
      cols[i * 3] = color.r;
      cols[i * 3 + 1] = color.g;
      cols[i * 3 + 2] = color.b;
    }
    return [pos, cols];
  }, []);

  useFrame((state) => {
    const time = state.clock.getElapsedTime();
    ref.current.rotation.z = Math.sin(time / 4) * 0.1;
    ref.current.rotation.y = Math.cos(time / 4) * 0.1;

    const x = (mouse.x * viewport.width) / 2;
    const y = (mouse.y * viewport.height) / 2;
    ref.current.position.x += (x - ref.current.position.x) * 0.02;
    ref.current.position.y += (y - ref.current.position.y) * 0.02;
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <Points ref={ref} positions={positions} colors={colors} stride={3} frustumCulled={false}>
        <PointMaterial
          transparent
          vertexColors
          size={0.05}
          sizeAttenuation={true}
          depthWrite={false}
          opacity={0.6}
          blending={THREE.AdditiveBlending}
        />
      </Points>
    </Float>
  );
}

const HowItWorks = () => {
  const { scrollY } = useScroll();
  const navOpacity = useTransform(scrollY, [0, 100], [0, 1]);
  const heroScale = useTransform(scrollY, [0, 500], [1, 0.9]);

  const { isDark, toggleTheme } = useTheme();

  const steps = [
    { id: '01', icon: 'movie_creation', title: 'Open Studio', link: '/studio', desc: 'AI-generated video & audio from text.', gradient: 'from-blue-600 to-indigo-500' },
    { id: '02', icon: 'record_voice_over', title: 'AI Voice Gen', link: '/voice', desc: 'Ultra-realistic neural voice synthesis.', gradient: 'from-indigo-500 to-purple-500' },
    { id: '03', icon: 'translate', title: 'Localize', link: '/localize', desc: 'Context-aware translation agent.', gradient: 'from-purple-600 to-pink-500' },
    { id: '04', icon: 'rocket_launch', title: 'Publish', link: '/publish', desc: 'Multi-platform automated distribution.', gradient: 'from-orange-500 to-amber-500' },
    { id: '05', icon: 'smart_toy', title: 'AI Agents', link: '/agents', desc: 'Autonomous workflow automation.', gradient: 'from-pink-500 to-rose-500' },
  ];

  return (
    <div className="min-h-screen bg-theme-primary text-theme-primary font-['Plus_Jakarta_Sans',_sans-serif] selection:bg-indigo-500/30 overflow-x-hidden transition-colors duration-300">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');

        .glass-card {
          background: var(--glass-bg);
          backdrop-filter: blur(20px);
          border: 1px solid var(--glass-border);
          box-shadow: 0 8px 32px 0 var(--glass-shadow);
          transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
        }

        .hero-mesh-bg {
          background-color: var(--bg-primary);
          background-image: 
            radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.1) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(6, 182, 212, 0.08) 0px, transparent 50%);
          animation: meshMove 15s ease infinite alternate;
        }

        @keyframes meshMove {
          0% { background-position: 0% 0%; }
          100% { background-position: 10% 10%; }
        }
        .hero-gradient-text {
          background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-primary) 50%, var(--accent-secondary) 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .text-glow {
          text-shadow: 0 0 40px rgba(99, 102, 241, 0.15);
        }
      `}</style>

      {/* Navigation */}
      <nav className="fixed top-0 w-full z-[100] h-16 flex items-center justify-between px-10 transition-colors duration-300">
        <motion.div
          style={{ opacity: navOpacity }}
          className="absolute inset-0 nav-glass"
        />

        <div className="relative z-10 flex items-center gap-2">
          <div className="w-7 h-7 bg-indigo-600 rounded-lg flex items-center justify-center text-white">
            <span className="material-symbols-rounded text-base">graphic_eq</span>
          </div>
          <span className="text-base font-extrabold tracking-tighter text-theme-primary">AuraFlow.</span>
        </div>

        <div className="relative z-10 flex items-center gap-6">
          <Link to="/documentation" className="text-[10px] font-bold text-theme-secondary hover:text-indigo-500 transition-colors uppercase tracking-widest hidden md:block">
            API Reference
          </Link>
          <Link to="/studio" className="px-4 py-1.5 bg-slate-900 dark:bg-white text-white dark:text-slate-900 text-[10px] font-bold rounded-full hover:bg-indigo-600 dark:hover:bg-indigo-400 dark:hover:text-white transition-all shadow-lg active:scale-95 uppercase tracking-widest">
            Open Studio
          </Link>
        </div>
      </nav>

      {/* Main Integrated Hero Section */}
      <section className="relative h-screen flex flex-col items-center justify-between py-24 px-6 overflow-hidden hero-mesh-bg">
        {/* Particle Canvas Layer */}
        <div className="absolute inset-0 z-0 pointer-events-none">
          <Canvas camera={{ position: [0, 0, 5] }} dpr={[1, 2]}>
            <Suspense fallback={null}>
              <ParticleField />
            </Suspense>
          </Canvas>
        </div>

        {/* Center Content (Smaller Text) */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 1.2, ease: "easeOut" }}
          className="relative z-10 text-center max-w-5xl"
        >


          <h1 className="text-7xl md:text-8xl font-[800] tracking-tight leading-[0.85] mb-10">
            <span className="hero-gradient-text">Universal Voice.</span><br />
            <span className="hero-gradient-text italic font-light">Liquid Impact.</span>
          </h1>




        </motion.div>
        {/* Process Cards (Moved into Hero Viewport) */}
        <div className="relative z-10 w-full max-w-[1400px] mx-auto grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mt-12">
          {steps.map((s, i) => (
            <Link to={s.link} key={i}>
              <motion.div
                initial={{ opacity: 0, y: 40, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ delay: 0.5 + (i * 0.12), duration: 0.7, type: 'spring', stiffness: 80 }}
                whileHover={{
                  scale: 1.05,
                  y: -15,
                  zIndex: 50,
                  rotate: i % 2 === 0 ? 1 : -1,
                  boxShadow: isDark ? '0 25px 50px -12px rgba(0, 0, 0, 0.5)' : '0 20px 40px -10px rgba(79, 70, 229, 0.4)',
                  borderColor: isDark ? 'rgba(255, 255, 255, 0.1)' : 'rgba(255, 255, 255, 1)'
                }}
                className={`glass-card p-8 min-w-[200px] max-w-[280px] min-h-[340px] rounded-[2rem] group cursor-pointer bg-gradient-to-br ${s.gradient} relative overflow-hidden flex flex-col justify-between`}
                style={{
                  background: isDark
                    ? `linear-gradient(135deg, var(--tw-gradient-stops)), rgba(15, 23, 42, 0.6)` /* Vibrant Dark Glass */
                    : `linear-gradient(135deg, var(--tw-gradient-stops)), rgba(255,255,255,0.55)`
                }}
              >
                <div className="relative mb-8 self-start">
                  {/* Pulsing Glow Behind */}
                  <div className={`absolute -inset-2 rounded-full bg-gradient-to-br ${s.gradient} opacity-20 blur-lg group-hover:opacity-40 transition-all duration-500`} />

                  {/* Main Icon Container */}
                  <motion.div
                    className={`relative w-20 h-20 rounded-full flex items-center justify-center shadow-[0_8px_16px_-6px_rgba(0,0,0,0.1)] z-10 border ${isDark
                      ? 'bg-white/5 border-white/20 backdrop-blur-md'
                      : 'bg-gradient-to-b from-white to-slate-50 border-white/80'
                      }`}
                    whileHover={{ scale: 1.1, rotate: 5, boxShadow: "0 12px 24px -8px rgba(0,0,0,0.15)" }}
                    transition={{ type: 'spring', stiffness: 300, damping: 15 }}
                  >
                    {/* Inner subtle ring */}
                    <div className={`absolute inset-1 rounded-full border ${isDark ? 'border-white/5' : 'border-slate-100/50'}`} />

                    <span className={`material-symbols-rounded text-8xl bg-gradient-to-br ${s.gradient} bg-clip-text text-transparent filter drop-shadow-sm`}>
                      {s.icon}
                    </span>
                  </motion.div>
                </div>

                {/* Animated colorful overlay */}
                <motion.div
                  className="absolute inset-0 rounded-[2rem] pointer-events-none z-0"
                  initial={{ opacity: 0.1, background: 'radial-gradient(circle at 60% 40%, #a5b4fc 0%, #f472b6 40%, #facc15 100%)' }}
                  animate={{
                    opacity: [0.1, 0.25, 0.1], background: [
                      'radial-gradient(circle at 60% 40%, #a5b4fc 0%, #f472b6 40%, #facc15 100%)',
                      'radial-gradient(circle at 40% 60%, #f472b6 0%, #a5b4fc 40%, #34d399 100%)',
                      'radial-gradient(circle at 60% 40%, #a5b4fc 0%, #f472b6 40%, #facc15 100%)',
                    ]
                  }}
                  transition={{ duration: 4, repeat: Infinity, repeatType: 'loop', ease: 'easeInOut' }}
                />

                <div className="relative z-10 mt-auto space-y-3">
                  <h3 className={`text-xl font-bold leading-tight group-hover:text-indigo-700 transition-colors ${isDark ? 'text-white' : 'text-slate-900'}`}>{s.title}</h3>
                  <div className="h-0.5 w-8 bg-slate-900/20 group-hover:w-full group-hover:bg-indigo-600 transition-all duration-500 ease-out" />
                  <p className={`text-sm font-medium leading-relaxed ${isDark ? 'text-slate-300' : 'text-slate-600'}`}>{s.desc}</p>
                </div>

                {/* Decorative Big Number at Bottom Right */}
                <div className={`absolute -bottom-6 -right-2 text-[120px] leading-none font-black select-none pointer-events-none group-hover:text-indigo-900/10 transition-all duration-500 z-0 group-hover:scale-110 origin-bottom-right ${isDark ? 'text-white/5' : 'text-slate-900/5'}`}>
                  {s.id}
                </div>

              </motion.div>
            </Link>
          ))}
        </div>
      </section>


    </div>
  );
};

export default HowItWorks;