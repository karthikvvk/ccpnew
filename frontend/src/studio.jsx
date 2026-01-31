import React, { useState, useEffect, useRef } from 'react';
import { useTheme } from './ThemeContext';
import { useTranslation } from './hooks/useTranslation';
import { setApiBaseUrl } from './services/api';
import {
  Play, Pause, Volume2, Maximize, Settings, Subtitles,
  Palette, Zap, Globe, Share2, Plus, Search,
  ZoomIn, ZoomOut, User, Menu, ChevronRight,
  Video, Music, Layers, Download, Type, Wand2,
  FolderOpen, ChevronLeft, ChevronRight as ChevronRightIcon,
  Trash2, Copy, Eye, Sparkles, Smile, Cloud, Activity, Minimize2,
  Clock, HardDrive, Cpu, MousePointer2, Command, Bell, Upload, Loader2, CheckCircle, AlertCircle
} from 'lucide-react';

const AuraFlowStudioPro = () => {
  const { isDark } = useTheme();

  // --- TRANSLATION API HOOK ---
  const translation = useTranslation();
  const fileInputRef = useRef(null);
  const [apiUrl, setApiUrl] = useState('');

  // --- UI STATES ---
  const [leftBarOpen, setLeftBarOpen] = useState(true);
  const [primarySidebarOpen, setPrimarySidebarOpen] = useState(true);
  const [rightBarOpen, setRightBarOpen] = useState(true);
  const [sidebarContext, setSidebarContext] = useState('Text');
  const [activeTab, setActiveTab] = useState('Editor');

  // --- VIDEO STATES ---
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // --- CONTENT STATES ---
  const [activeLang, setActiveLang] = useState('TA');
  const [textStyle, setTextStyle] = useState('Modern');
  const [videoFilter, setVideoFilter] = useState('none');
  const [activeMagic, setActiveMagic] = useState(null);

  // --- API URL HANDLER ---
  const handleApiUrlChange = (url) => {
    setApiUrl(url);
    if (url) setApiBaseUrl(url);
  };

  // --- FILE UPLOAD HANDLER ---
  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      translation.handleVideoSelect(file);
    }
  };

  // --- START RENDER HANDLER ---
  const handleStartRender = () => {
    if (translation.canStart) {
      translation.startTranslation(activeLang, true);
    }
  };

  const [expandedSections, setExpandedSections] = useState({
    videos: true,
    audio: true,
    images: false,
    recent: true
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const duration = 180;
  const timelineRef = useRef(null);

  const translations = {
    TA: "AURAX AI டப்பிங் மூலம் உங்கள் படைப்புகளை அடுத்த கட்டத்திற்கு உயர்த்துங்கள். 🚀",
    JP: "AUREX AIダビングでコンテンツをレベルアップ 🚀",
    DE: "Verbessern Sie Ihre Inhalte mit AUREX AI Dubbing 🚀",
    EN: "Level up your content with AUREX AI Dubbing 🚀"
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) { setIsPlaying(false); return 100; }
          return prev + 0.1;
        });
      }, 50);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const handleTimelineClick = (e) => {
    const rect = timelineRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    setProgress(Math.min(Math.max((x / rect.width) * 100, 0), 100));
  };

  return (
    <div className={`flex h-screen font-sans overflow-hidden select-none antialiased transition-colors duration-300 ${isDark ? 'bg-slate-950 text-slate-100' : 'bg-[#F8FAFC] text-slate-900'}`}>

      {/* 1. PRIMARY ICON SIDEBAR */}
      <aside className={`border-r flex flex-col items-center py-6 gap-6 z-50 transition-all duration-300 ease-in-out ${primarySidebarOpen ? 'w-[72px]' : 'w-0 overflow-hidden opacity-0'} ${isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
        <div className="bg-indigo-600 p-2.5 rounded-2xl shadow-lg shadow-indigo-100 cursor-pointer hover:rotate-12 transition-all active:scale-95" onClick={() => setPrimarySidebarOpen(false)}>
          <Zap size={24} className="text-white fill-white" />
        </div>

        <div className="flex flex-col gap-4 text-slate-400 mt-4">
          <SideIcon icon={<FolderOpen size={22} />} label="Library" active={sidebarContext === 'Project'} onClick={() => setSidebarContext('Project')} />
          <SideIcon icon={<Type size={22} />} label="Typography" active={sidebarContext === 'Text'} onClick={() => setSidebarContext('Text')} />
          <SideIcon icon={<Palette size={22} />} label="Grade" active={sidebarContext === 'Colors'} onClick={() => setSidebarContext('Colors')} />
          <SideIcon icon={<Wand2 size={22} />} label="Magic AI" active={sidebarContext === 'Magic'} onClick={() => setSidebarContext('Magic')} />
        </div>

        <div className="mt-auto pb-4 flex flex-col gap-6">
          <div className="p-2 hover:bg-slate-100 rounded-xl cursor-pointer transition-colors text-slate-400">
            <Bell size={20} />
          </div>
          <button onClick={() => setLeftBarOpen(!leftBarOpen)} className="p-2.5 bg-slate-50 hover:bg-slate-100 rounded-xl text-slate-500 border border-slate-200 transition-all">
            {leftBarOpen ? <ChevronLeft size={18} /> : <ChevronRightIcon size={18} />}
          </button>
        </div>
      </aside>

      {!primarySidebarOpen && (
        <button
          onClick={() => setPrimarySidebarOpen(true)}
          className="fixed left-0 top-1/2 -translate-y-1/2 bg-indigo-600 hover:bg-indigo-700 p-2 rounded-r-2xl shadow-xl z-[60] transition-all"
        >
          <ChevronRightIcon size={20} className="text-white" />
        </button>
      )}

      {/* 2. DYNAMIC UTILITY PANEL */}
      <aside
        className={`border-r flex flex-col transition-all duration-300 ease-in-out relative ${leftBarOpen ? 'w-[320px]' : 'w-0 overflow-hidden'} ${isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}
      >
        <div className={`p-6 border-b flex justify-between items-center sticky top-0 z-10 ${isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-100'}`}>
          <div>
            <h2 className={`font-bold text-sm tracking-tight ${isDark ? 'text-slate-100' : 'text-slate-900'}`}>{sidebarContext} Tools</h2>
            <p className="text-[11px] text-slate-400 font-medium">Customize your sequence</p>
          </div>
          <div className="p-2 bg-slate-50 rounded-lg text-slate-400 hover:text-indigo-600 cursor-pointer transition-colors">
            <Search size={16} />
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-6 scrollbar-hide">
          {sidebarContext === 'Text' && (
            <div className="grid grid-cols-1 gap-3">
              <ToolCard title="Modern Sans" desc="Minimalist Swiss style" active={textStyle === 'Modern'} onClick={() => setTextStyle('Modern')} icon={<Type size={18} />} />
              <ToolCard title="Cyber Neon" desc="Atmospheric glow" active={textStyle === 'Neon'} onClick={() => setTextStyle('Neon')} icon={<Sparkles size={18} />} />
              <ToolCard title="Impact Bold" desc="High contrast display" active={textStyle === 'Bold'} onClick={() => setTextStyle('Bold')} icon={<Layers size={18} />} />
              <ToolCard title="Royal Serif" desc="Classic editorial vibe" active={textStyle === 'Elegant'} onClick={() => setTextStyle('Elegant')} icon={<Copy size={18} />} />
            </div>
          )}

          {sidebarContext === 'Colors' && (
            <div className="space-y-4">
              <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest px-1">Presets</p>
              <div className="grid grid-cols-2 gap-3">
                <FilterBox label="Original" filter="none" active={videoFilter === 'none'} set={setVideoFilter} img="https://images.unsplash.com/photo-1574717024653-61fd2cf4d44d?w=200" />
                <FilterBox label="Noir" filter="grayscale(100%)" active={videoFilter === 'grayscale(100%)'} set={setVideoFilter} img="https://images.unsplash.com/photo-1574717024653-61fd2cf4d44d?w=200" />
                <FilterBox label="Faded Sepia" filter="sepia(80%)" active={videoFilter === 'sepia(80%)'} set={setVideoFilter} img="https://images.unsplash.com/photo-1574717024653-61fd2cf4d44d?w=200" />
                <FilterBox label="Cold Snap" filter="hue-rotate(180deg) brightness(1.1)" active={videoFilter === 'hue-rotate(180deg) brightness(1.2)'} set={setVideoFilter} img="https://images.unsplash.com/photo-1574717024653-61fd2cf4d44d?w=200" />
              </div>
            </div>
          )}

          {sidebarContext === 'Magic' && (
            <div className="space-y-4">
              <div className="p-5 bg-gradient-to-br from-indigo-50 to-white border border-indigo-100 rounded-2xl shadow-sm">
                <div className="flex items-center gap-2 mb-4">
                  <div className="p-1.5 bg-indigo-600 rounded-lg text-white">
                    <Zap size={14} />
                  </div>
                  <p className="text-[11px] font-bold text-indigo-900 uppercase tracking-wider">AI Engines</p>
                </div>
                <div className="space-y-1">
                  <MagicItem icon={<Smile size={16} />} label="Neural Face Tracking" active={activeMagic === 'face'} onClick={() => setActiveMagic('face')} />
                  <MagicItem icon={<Cloud size={16} />} label="Magic BG Removal" active={activeMagic === 'bg'} onClick={() => setActiveMagic('bg')} />
                  <MagicItem icon={<Activity size={16} />} label="Lip Sync Pro v2.4" active={activeMagic === 'lip'} onClick={() => setActiveMagic('lip')} />
                </div>
              </div>
            </div>
          )}

          {sidebarContext === 'Project' && (
            <div className="space-y-4">
              <CollapsibleSection title="Source Videos" count={3} icon={<Video size={16} />} expanded={expandedSections.videos} onToggle={() => toggleSection('videos')}>
                {[1, 2, 3].map(i => (
                  <div key={i} className="group p-2.5 bg-white border border-slate-100 rounded-xl flex items-center gap-3 hover:border-indigo-300 hover:shadow-md transition-all cursor-pointer">
                    <div className="w-12 h-8 bg-slate-200 rounded-lg overflow-hidden relative">
                      <img src={`https://picsum.photos/seed/${i + 20}/100/100`} className="object-cover w-full h-full group-hover:scale-110 transition-transform" />
                      <div className="absolute inset-0 bg-black/10"></div>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[11px] font-bold truncate text-slate-700">Clip_Sequence_{i}.mp4</div>
                      <div className="text-[9px] text-slate-400 font-medium">00:45 • 4K HEVC</div>
                    </div>
                  </div>
                ))}
              </CollapsibleSection>

              <CollapsibleSection title="Audio Stems" count={2} icon={<Music size={16} />} expanded={expandedSections.audio} onToggle={() => toggleSection('audio')}>
                {['Voiceover_EN.wav', 'BGM_Ambient.mp3'].map((name, i) => (
                  <div key={i} className="p-2.5 bg-white border border-slate-100 rounded-xl flex items-center gap-3 hover:border-purple-300 transition-all cursor-pointer">
                    <div className="w-10 h-10 bg-purple-50 rounded-xl flex items-center justify-center text-purple-600 border border-purple-100">
                      <Volume2 size={16} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="text-[11px] font-bold truncate text-slate-700">{name}</div>
                      <div className="text-[9px] text-slate-400 font-medium">48kHz • 24-bit</div>
                    </div>
                  </div>
                ))}
              </CollapsibleSection>
            </div>
          )}
        </div>
      </aside>

      {/* 3. MAIN WORKSPACE */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* HEADER */}
        <header className={`h-[72px] border-b flex items-center justify-between px-8 backdrop-blur-md z-40 ${isDark ? 'bg-slate-900/80 border-slate-800' : 'bg-white/80 border-slate-200'}`}>
          <div className="flex items-center gap-10">


            <nav className="flex bg-slate-100/80 p-1.5 rounded-2xl border border-slate-200/50">
              {['Editor', 'Assets', 'Team'].map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab === 'Team' ? 'Collaborate' : tab)}
                  className={`px-6 py-2 rounded-xl text-[12px] font-bold transition-all ${(activeTab === tab || (activeTab === 'Collaborate' && tab === 'Team')) ? 'bg-white text-indigo-600 shadow-md ring-1 ring-slate-200' : 'text-slate-500 hover:text-slate-800'
                    }`}
                >
                  {tab}
                </button>
              ))}
            </nav>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center -space-x-3">
              <UserAvatar name="VD" color="from-pink-500 to-rose-500" />
              <UserAvatar name="AK" color="from-blue-500 to-indigo-500" />
              <button className="w-9 h-9 rounded-full bg-white border-2 border-dashed border-slate-200 flex items-center justify-center text-slate-400 hover:border-indigo-400 hover:text-indigo-600 transition-all">
                <Plus size={16} />
              </button>
            </div>



            <button className="bg-indigo-600 hover:bg-indigo-700 active:scale-95 text-white px-6 py-2.5 rounded-2xl text-[13px] font-bold shadow-xl shadow-indigo-200 flex items-center gap-2 transition-all">
              <Share2 size={16} /> Export
            </button>
          </div>
        </header>

        {/* EDITOR AREA */}
        <main className={`flex-1 overflow-y-auto p-8 transition-all duration-500 ${isFullscreen ? 'fixed inset-0 z-[100] bg-black p-0' : (isDark ? 'bg-slate-950' : 'bg-[#F8FAFC]')}`}>

          {activeTab === 'Editor' && (
            <div className={`max-w-6xl mx-auto space-y-8 ${isFullscreen ? 'max-w-none h-full' : ''}`}>

              {/* PREVIEW CONTAINER */}
              <div className={`group relative aspect-video rounded-[32px] overflow-hidden bg-[#0F172A] shadow-[0_20px_50px_rgba(0,0,0,0.2)] border border-slate-200/10 transition-all duration-700 ${isFullscreen ? 'rounded-none border-none h-full' : ''}`}>

                <div
                  className="w-full h-full transition-all duration-1000 ease-in-out"
                  style={{ filter: videoFilter }}
                >
                  <img
                    src="https://images.unsplash.com/photo-1492691527719-9d1e07e534b4?auto=format&fit=crop&q=80&w=2000"
                    className={`w-full h-full object-cover transition-transform duration-[10s] ease-linear ${isPlaying ? 'scale-110' : 'scale-100'}`}
                    alt="Video Preview"
                  />
                </div>

                {/* OVERLAYS */}
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none p-12 text-center z-10">
                  <div className={`transition-all duration-700 transform ${isPlaying ? 'scale-105' : 'scale-100 opacity-90'} 
                   ${textStyle === 'Neon' ? 'text-indigo-400 drop-shadow-[0_0_15px_rgba(129,140,248,1)] font-black italic tracking-widest' : ''}
                   ${textStyle === 'Bold' ? 'text-white font-black text-6xl uppercase tracking-tighter' : ''}
                   ${textStyle === 'Modern' ? 'text-white font-bold text-4xl bg-indigo-600/90 backdrop-blur-md px-8 py-3 rounded-2xl shadow-2xl ring-1 ring-white/20' : ''}
                   ${textStyle === 'Elegant' ? 'text-white font-serif italic text-5xl border-b-2 border-white/30 pb-4' : ''}
                 `}>
                    {translations[activeLang]}
                  </div>
                </div>

                {/* CENTER TRANSPORT ICON */}
                <div className="absolute inset-0 flex items-center justify-center pointer-events-none bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={() => setIsPlaying(!isPlaying)}
                    className="w-24 h-24 bg-white/10 backdrop-blur-2xl border border-white/20 rounded-full flex items-center justify-center shadow-2xl hover:scale-110 hover:bg-white/20 transition-all pointer-events-auto active:scale-90"
                  >
                    {isPlaying ? <Pause size={40} className="text-white" fill="white" /> : <Play size={40} className="text-white ml-2" fill="white" />}
                  </button>
                </div>

                {/* HUD OVERLAY - TOP */}
                <div className="absolute top-8 left-8 right-8 flex justify-between items-start pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="flex gap-4">
                    <div className="bg-black/40 backdrop-blur-xl border border-white/10 px-4 py-2 rounded-2xl text-white text-[10px] font-bold font-mono tracking-wider">
                      REC • {formatTime(progress * duration / 100)}
                    </div>
                    <div className="bg-black/40 backdrop-blur-xl border border-white/10 px-4 py-2 rounded-2xl text-white text-[10px] font-bold font-mono tracking-wider">
                      4K @ 60FPS
                    </div>
                  </div>
                  <button onClick={() => setIsFullscreen(!isFullscreen)} className="pointer-events-auto p-3 bg-black/40 backdrop-blur-xl border border-white/10 rounded-2xl text-white hover:bg-indigo-600 transition-colors">
                    {isFullscreen ? <Minimize2 size={20} /> : <Maximize size={20} />}
                  </button>
                </div>

                {/* HUD OVERLAY - BOTTOM (Scrubber) */}
                <div className="absolute bottom-0 inset-x-0 p-8 bg-gradient-to-t from-black via-black/40 to-transparent pt-20 translate-y-2 group-hover:translate-y-0 transition-transform">
                  <div className="relative mb-6 group/track cursor-pointer" onClick={handleTimelineClick}>
                    <div className="h-1.5 bg-white/10 rounded-full overflow-hidden">
                      <div className="h-full bg-gradient-to-r from-indigo-500 via-purple-500 to-indigo-400 rounded-full" style={{ width: `${progress}%` }}></div>
                    </div>
                    <div
                      className="absolute top-1/2 -translate-y-1/2 w-4 h-4 bg-white rounded-full shadow-[0_0_20px_rgba(255,255,255,0.5)] border-4 border-indigo-600 scale-0 group-hover/track:scale-100 transition-all"
                      style={{ left: `${progress}%`, marginLeft: '-8px' }}
                    ></div>
                  </div>

                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-6">
                      <button onClick={() => setIsPlaying(!isPlaying)} className="text-white hover:text-indigo-400 transition-colors transform active:scale-90">
                        {isPlaying ? <Pause size={24} fill="currentColor" /> : <Play size={24} fill="currentColor" />}
                      </button>
                      <div className="flex items-center gap-4 group/vol">
                        <Volume2 size={18} className="text-white/60 group-hover/vol:text-white transition-colors" />
                        <div className="w-24 h-1 bg-white/10 rounded-full">
                          <div className="w-3/4 h-full bg-indigo-500 rounded-full"></div>
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 bg-white/5 px-4 py-2 rounded-xl backdrop-blur-lg border border-white/5">
                      <span className="text-indigo-400 font-mono text-sm font-bold">{formatTime(progress * duration / 100)}</span>
                      <span className="text-white/20 font-mono text-sm">/</span>
                      <span className="text-white/60 font-mono text-sm">{formatTime(duration)}</span>
                    </div>

                    <div className="flex items-center gap-2">
                      <ControlIcon icon={<Settings size={18} />} />
                      <ControlIcon icon={<Subtitles size={18} />} />
                    </div>
                  </div>
                </div>
              </div>

              {/* TIMELINE VIEW */}
              {!isFullscreen && (
                <div className="bg-white border border-slate-200 rounded-[24px] overflow-hidden shadow-xl shadow-slate-200/50 transition-all hover:shadow-2xl">
                  <div className="px-6 py-4 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                    <div className="flex items-center gap-3">
                      <TimelineTool icon={<MousePointer2 size={14} />} active />
                      <TimelineTool icon={<Command size={14} />} />
                      <div className="w-px h-6 bg-slate-200 mx-2"></div>
                      <TimelineTool icon={<ZoomIn size={14} />} />
                      <TimelineTool icon={<ZoomOut size={14} />} />
                    </div>
                    <div className="text-[10px] font-black text-slate-400 tracking-[0.3em] uppercase flex items-center gap-2">
                      <div className="w-1.5 h-1.5 bg-indigo-500 rounded-full"></div>
                      Main Sequence Timeline
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="font-mono text-[13px] font-bold text-indigo-600 bg-indigo-50 px-4 py-1.5 rounded-xl border border-indigo-100">
                        00:01:24:12
                      </div>
                    </div>
                  </div>

                  <div className="p-8 space-y-4 relative bg-[#FBFCFD]" ref={timelineRef} onClick={handleTimelineClick}>
                    {/* Playhead */}
                    <div className="absolute top-0 bottom-0 w-0.5 bg-indigo-500 z-30 pointer-events-none" style={{ left: `${progress}%` }}>
                      <div className="absolute top-0 -left-[7px] w-0 h-0 border-l-[7px] border-l-transparent border-r-[7px] border-r-transparent border-top-[10px] border-t-indigo-600"></div>
                    </div>

                    {/* Time Markers */}
                    <div className="h-6 w-full flex justify-between px-2 text-[9px] font-bold text-slate-300 border-b border-slate-100 mb-6">
                      {[...Array(11)].map((_, i) => <span key={i}>{i * 10}s</span>)}
                    </div>

                    <TimelineTrack icon={<Video size={14} />} label="Primary Video" color="bg-slate-200" width="w-[85%]" />
                    <TimelineTrack icon={<Volume2 size={14} />} label="AI Dubbing" color="bg-indigo-500/10" border="border-indigo-500/30" width="w-[92%]" isAudio />
                    <TimelineTrack icon={<Subtitles size={14} />} label="Captions ES" color="bg-purple-500/10" border="border-purple-500/30" width="w-[60%]" />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ASSETS TAB */}
          {activeTab === 'Assets' && (
            <div className="max-w-6xl mx-auto py-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="flex items-center justify-between mb-8">
                <div>
                  <h2 className="text-2xl font-black text-slate-900 tracking-tight">Media Library</h2>
                  <p className="text-sm text-slate-500 font-medium">Manage and organize your project assets</p>
                </div>
                <div className="flex gap-3">
                  <div className="bg-white border border-slate-200 rounded-2xl flex items-center px-4 shadow-sm focus-within:ring-2 focus-within:ring-indigo-100 transition-all">
                    <Search size={16} className="text-slate-400" />
                    <input type="text" placeholder="Search project..." className="bg-transparent border-none py-2.5 px-3 text-sm outline-none w-48 font-medium" />
                  </div>
                  <button className="bg-indigo-600 text-white px-5 py-2.5 rounded-2xl text-[13px] font-bold shadow-lg shadow-indigo-100 flex items-center gap-2">
                    <Plus size={16} /> Add New
                  </button>
                </div>
              </div>

              <div className="grid grid-cols-4 gap-6">
                {[
                  { name: 'Drone_Shot_01.mp4', type: 'video', size: '256 MB', thumb: 'https://picsum.photos/seed/p1/300/200' },
                  { name: 'Podcast_Audio.wav', type: 'audio', size: '12 MB', thumb: null },
                  { name: 'Title_Graphic.png', type: 'image', size: '4 MB', thumb: 'https://picsum.photos/seed/p2/300/200' },
                  { name: 'Final_Broll.mp4', type: 'video', size: '1.2 GB', thumb: 'https://picsum.photos/seed/p3/300/200' },
                ].map((asset, i) => (
                  <div key={i} className="group bg-white rounded-[24px] border border-slate-200 overflow-hidden hover:shadow-2xl hover:border-indigo-400 transition-all cursor-pointer">
                    <div className="aspect-video bg-slate-100 relative">
                      {asset.thumb ? <img src={asset.thumb} className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" /> : <div className="w-full h-full flex items-center justify-center bg-indigo-50"><Music size={32} className="text-indigo-300" /></div>}
                      <div className="absolute top-3 right-3 px-2 py-1 bg-black/60 backdrop-blur-md rounded-lg text-[9px] font-bold text-white uppercase">{asset.type}</div>
                    </div>
                    <div className="p-4">
                      <p className="text-sm font-bold text-slate-900 truncate">{asset.name}</p>
                      <div className="flex justify-between items-center mt-1">
                        <span className="text-[10px] text-slate-400 font-bold uppercase">{asset.size}</span>
                        <div className="opacity-0 group-hover:opacity-100 flex gap-2 transition-opacity">
                          <button className="p-1.5 hover:bg-slate-100 rounded-lg text-slate-400 hover:text-indigo-600"><Copy size={14} /></button>
                          <button className="p-1.5 hover:bg-slate-100 rounded-lg text-slate-400 hover:text-rose-600"><Trash2 size={14} /></button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* COLLABORATE TAB */}
          {activeTab === 'Collaborate' && (
            <div className="max-w-4xl mx-auto py-4 animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div className="bg-white rounded-[32px] border border-slate-200 overflow-hidden shadow-xl">
                <div className="p-8 border-b border-slate-100 flex justify-between items-center bg-slate-50/50">
                  <div>
                    <h3 className="text-xl font-black text-slate-900 tracking-tight">Team Access</h3>
                    <p className="text-sm text-slate-500 font-medium">Manage project permissions</p>
                  </div>
                  <button className="bg-slate-900 text-white px-5 py-2.5 rounded-2xl text-[12px] font-bold shadow-lg flex items-center gap-2">
                    <User size={16} /> Manage Roles
                  </button>
                </div>
                <div className="divide-y divide-slate-100">
                  <MemberRow name="Vishal V D" email="vishal@auraflow.io" role="Owner" status="online" initials="VD" color="from-indigo-500 to-blue-500" />
                  <MemberRow name="Alex K" email="alex@auraflow.io" role="Editor" status="online" initials="AK" color="from-emerald-500 to-teal-500" />
                  <MemberRow name="Sarah J" email="sarah@auraflow.io" role="Viewer" status="away" initials="SJ" color="from-purple-500 to-indigo-500" />
                </div>
              </div>
            </div>
          )}
        </main>

        <footer className="h-10 bg-white border-t border-slate-200 px-8 flex justify-between items-center text-[10px] font-bold text-slate-400 tracking-widest uppercase">
          <div className="flex gap-8">
            <span className="flex items-center gap-2"><Cpu size={12} /> Engine: v4.0.2 Stable</span>
            <span className="flex items-center gap-2"><HardDrive size={12} /> Buffer: 2.4GB</span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-indigo-600 font-black">
              <div className="w-1.5 h-1.5 bg-indigo-600 rounded-full animate-pulse"></div>
              Live Link Enabled
            </div>
            <span className="text-slate-200">|</span>
            <span>© 2026 AuraFlow.io</span>
          </div>
        </footer>
      </div>

      {/* 4. RIGHT STUDIO CONTROLS */}
      <aside
        className={`border-l flex flex-col transition-all duration-300 ease-in-out ${rightBarOpen ? 'w-[360px]' : 'w-0 overflow-hidden'} ${isDark ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}
      >
        <div className="p-8 flex justify-between items-center border-b border-slate-50">
          <div>
            <h2 className="text-sm font-black uppercase tracking-widest text-slate-900">Studio Specs</h2>
            <p className="text-[10px] text-slate-400 font-bold tracking-tight">Configuration and Output</p>
          </div>
          <button onClick={() => setRightBarOpen(false)} className="p-2 bg-slate-50 text-slate-400 hover:text-slate-900 rounded-xl transition-all"><ChevronRightIcon size={20} /></button>
        </div>

        <div className="p-8 space-y-10 overflow-y-auto flex-1 scrollbar-hide">
          <section>
            <div className="flex justify-between items-center mb-6">
              <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em]">Target Language</h3>
              <div className="p-1 bg-indigo-50 rounded-lg text-indigo-600 cursor-pointer hover:bg-indigo-100"><Plus size={14} /></div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <DubCard lang="TA" title="Tamil" active={activeLang === 'TA'} onClick={() => setActiveLang('TA')} />
              <DubCard lang="JP" title="Japanese" active={activeLang === 'JP'} onClick={() => setActiveLang('JP')} />
              <DubCard lang="DE" title="German" active={activeLang === 'DE'} onClick={() => setActiveLang('DE')} />
              <DubCard lang="EN" title="English" active={activeLang === 'EN'} onClick={() => setActiveLang('EN')} />
            </div>
          </section>

          <section className="bg-slate-900 p-6 rounded-[24px] shadow-2xl relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/10 rounded-full -mr-16 -mt-16 blur-3xl group-hover:bg-indigo-500/20 transition-all"></div>
            <div className="relative z-10">
              <div className="flex justify-between items-center mb-6">
                <h3 className="text-[10px] font-black text-indigo-400 uppercase tracking-widest">Neural Speaker Model</h3>
                <Sparkles size={16} className="text-indigo-400 animate-pulse" />
              </div>
              <div className="flex items-center gap-5">
                <div className="w-14 h-14 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-2xl flex items-center justify-center text-white shadow-xl ring-2 ring-indigo-400/20">
                  <User size={28} />
                </div>
                <div>
                  <p className="text-sm font-black text-white tracking-tight">Vishal_V_D_Clone</p>
                  <p className="text-[11px] text-indigo-300 font-bold italic tracking-wide">Accuracy 99.4%</p>
                </div>
              </div>
              <div className="mt-6 pt-6 border-t border-white/10 flex justify-between items-center text-[10px] font-bold text-white/40 uppercase">
                <span>Latency: 40ms</span>
                <span>Neural Core v2</span>
              </div>
            </div>
          </section>

          <section>
            <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em] mb-4">Export Metadata</h3>
            <div className="space-y-2">
              <MetadataRow label="Format" value="H.264 MP4" />
              <MetadataRow label="Resolution" value="3840 x 2160" />
              <MetadataRow label="Bitrate" value="45 Mbps" />
            </div>
          </section>
        </div>

        {/* Hidden file input for video upload */}
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          accept="video/*"
          className="hidden"
        />

        {/* API URL Input (for Colab ngrok) */}
        <div className="px-8 py-4 border-t border-slate-100">
          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-widest block mb-2">Backend URL (Colab)</label>
          <input
            type="text"
            placeholder="https://your-ngrok-url.ngrok-free.dev"
            value={apiUrl}
            onChange={(e) => handleApiUrlChange(e.target.value)}
            className="w-full px-4 py-2 text-sm border border-slate-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-indigo-200"
          />
        </div>

        {/* Upload Video Button */}
        <div className="px-8 py-2">
          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full py-3 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-[16px] text-[11px] font-bold uppercase tracking-wide transition-all flex items-center justify-center gap-2"
          >
            <Upload size={14} />
            {translation.hasVideo ? translation.uploadedVideo?.name?.slice(0, 20) + '...' : 'Upload Video'}
          </button>
        </div>

        {/* Status Display */}
        {translation.progress && (
          <div className="px-8 py-2">
            <div className={`p-3 rounded-xl text-[11px] font-bold flex items-center gap-2 ${translation.isCompleted ? 'bg-emerald-50 text-emerald-700' :
              translation.isFailed ? 'bg-red-50 text-red-700' :
                'bg-indigo-50 text-indigo-700'
              }`}>
              {translation.isProcessing && <Loader2 size={14} className="animate-spin" />}
              {translation.isCompleted && <CheckCircle size={14} />}
              {translation.isFailed && <AlertCircle size={14} />}
              {translation.progress}
            </div>
          </div>
        )}

        {/* Download Button (shown when completed) */}
        {translation.isCompleted && (
          <div className="px-8 py-2">
            <button
              onClick={() => translation.downloadFile('final_video')}
              className="w-full py-3 bg-emerald-600 hover:bg-emerald-700 text-white rounded-[16px] text-[11px] font-bold uppercase tracking-wide transition-all flex items-center justify-center gap-2"
            >
              <Download size={14} /> Download Translated Video
            </button>
          </div>
        )}

        <div className="p-8 border-t border-slate-100 bg-white">
          <button
            onClick={handleStartRender}
            disabled={!translation.canStart || translation.isProcessing}
            className={`w-full py-4 rounded-[20px] text-[12px] font-black uppercase tracking-[0.1em] shadow-xl transition-all active:scale-95 flex items-center justify-center gap-3 group ${translation.canStart && !translation.isProcessing
              ? 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-100'
              : 'bg-slate-300 text-slate-500 cursor-not-allowed shadow-slate-100'
              }`}
          >
            {translation.isProcessing ? (
              <><Loader2 size={16} className="animate-spin" /> Processing...</>
            ) : (
              <>Start Sequence Render <ChevronRightIcon size={16} className="group-hover:translate-x-1 transition-transform" /></>
            )}
          </button>
        </div>
      </aside>

      {!rightBarOpen && (
        <button
          onClick={() => setRightBarOpen(true)}
          className="fixed right-0 top-1/2 -translate-y-1/2 bg-white border border-slate-200 p-2 rounded-l-2xl shadow-xl z-[60] text-indigo-600"
        >
          <ChevronLeft size={20} />
        </button>
      )}
    </div>
  );
};

/* --- REFINED SUB-COMPONENTS --- */

const SideIcon = ({ icon, active, onClick, label }) => (
  <button
    onClick={onClick}
    className={`group relative p-3 rounded-2xl transition-all duration-300 ${active ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-100' : 'hover:bg-slate-100 text-slate-400 hover:text-slate-600'}`}
  >
    {icon}
    <div className="absolute left-[120%] top-1/2 -translate-y-1/2 px-3 py-1 bg-slate-900 text-white text-[10px] font-bold rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-[100]">
      {label}
    </div>
  </button>
);

const UserAvatar = ({ name, color }) => (
  <div className={`w-10 h-10 rounded-full bg-gradient-to-br ${color} border-[3px] border-white shadow-md flex items-center justify-center text-white text-[11px] font-black ring-1 ring-slate-100 transition-transform hover:-translate-y-1 cursor-pointer`}>
    {name}
  </div>
);

const TimelineTool = ({ icon, active }) => (
  <button className={`p-2 rounded-lg transition-all ${active ? 'bg-white shadow-sm ring-1 ring-slate-200 text-indigo-600' : 'text-slate-400 hover:bg-white hover:text-slate-600'}`}>
    {icon}
  </button>
);

const ToolCard = ({ title, desc, active, onClick, icon }) => (
  <button
    onClick={onClick}
    className={`group p-4 rounded-[22px] border-2 flex items-center gap-4 transition-all text-left ${active ? 'border-indigo-600 bg-indigo-50/30 shadow-xl ring-4 ring-indigo-50/50' : 'border-slate-50 bg-[#FCFDFE] hover:border-slate-200 hover:bg-white'}`}
  >
    <div className={`p-2.5 rounded-xl transition-all ${active ? 'bg-indigo-600 text-white rotate-6' : 'bg-white border border-slate-100 text-slate-400'}`}>{icon}</div>
    <div>
      <p className="text-[12px] font-black text-slate-900 tracking-tight leading-tight">{title}</p>
      <p className="text-[10px] text-slate-400 font-bold mt-0.5">{desc}</p>
    </div>
  </button>
);

const FilterBox = ({ label, filter, active, set, img }) => (
  <div onClick={() => set(filter)} className={`cursor-pointer group relative rounded-[18px] overflow-hidden border-2 transition-all ${active ? 'border-indigo-600 scale-95 shadow-lg' : 'border-transparent opacity-80 hover:opacity-100'}`}>
    <img src={img} className="w-full h-20 object-cover" style={{ filter }} />
    <div className={`absolute inset-0 flex items-end p-2 transition-all ${active ? 'bg-indigo-600/20' : 'bg-gradient-to-t from-black/60 to-transparent'}`}>
      <span className="text-[9px] font-black text-white uppercase tracking-widest">{label}</span>
    </div>
  </div>
);

const MagicItem = ({ icon, label, active, onClick }) => (
  <div onClick={onClick} className={`flex items-center gap-4 p-3 rounded-xl cursor-pointer transition-all ${active ? 'bg-indigo-600 text-white shadow-lg' : 'hover:bg-indigo-50 text-slate-600'}`}>
    <span className={active ? 'text-white' : 'text-indigo-600'}>{icon}</span>
    <span className="text-[12px] font-bold tracking-tight">{label}</span>
  </div>
);

const TimelineTrack = ({ icon, label, color, border, width, isAudio }) => (
  <div className="flex gap-6 items-center group h-14">
    <div className="w-32 flex items-center gap-3 text-[11px] font-black text-slate-500 uppercase tracking-tighter">
      <div className="p-1.5 bg-white rounded-lg shadow-sm border border-slate-100">{icon}</div>
      {label}
    </div>
    <div className="flex-1 bg-white rounded-2xl h-12 border border-slate-200/60 relative group-hover:border-slate-300 transition-all overflow-hidden shadow-inner">
      <div className={`absolute top-1 bottom-1 left-1 ${color} ${border ? `${border} border` : ''} rounded-xl shadow-sm`} style={{ width }}>
        {isAudio && (
          <div className="flex items-center gap-1 h-full px-6 overflow-hidden opacity-40">
            {[...Array(80)].map((_, i) => (
              <div key={i} className="w-0.5 bg-indigo-600 rounded-full" style={{ height: `${20 + Math.sin(i * 0.5) * 60}%` }}></div>
            ))}
          </div>
        )}
        {!isAudio && (
          <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent"></div>
        )}
      </div>
    </div>
  </div>
);

const DubCard = ({ lang, title, active, onClick }) => (
  <div onClick={onClick} className={`p-5 rounded-[24px] border-2 transition-all cursor-pointer relative overflow-hidden ${active ? 'border-indigo-600 bg-white shadow-2xl ring-4 ring-indigo-50' : 'border-slate-50 bg-[#FCFDFE] hover:border-slate-200'}`}>
    <div className="flex justify-between items-center mb-3">
      <span className={`text-[11px] font-black tracking-widest ${active ? 'text-indigo-600' : 'text-slate-400'}`}>{lang}</span>
      {active && <div className="w-2.5 h-2.5 bg-indigo-600 rounded-full shadow-[0_0_10px_rgba(79,70,229,0.8)]"></div>}
    </div>
    <p className="text-[13px] font-black text-slate-900 uppercase tracking-tight">{title}</p>
    <p className="text-[10px] text-slate-400 font-bold mt-1 uppercase tracking-tighter">Ready to sync</p>
  </div>
);

const CollapsibleSection = ({ title, count, icon, expanded, onToggle, children }) => (
  <div className="bg-slate-50 border border-slate-200/50 rounded-2xl overflow-hidden transition-all">
    <button onClick={onToggle} className="w-full p-4 flex items-center justify-between hover:bg-slate-100 transition-colors">
      <div className="flex items-center gap-3">
        <span className="text-indigo-600 bg-white p-1.5 rounded-lg shadow-sm border border-slate-100">{icon}</span>
        <span className="text-[11px] font-black uppercase tracking-widest text-slate-600">{title}</span>
        <span className="text-[10px] bg-indigo-600 text-white px-2 py-0.5 rounded-full font-black">{count}</span>
      </div>
      <ChevronRight size={14} className={`text-slate-400 transition-transform duration-300 ${expanded ? 'rotate-90' : ''}`} />
    </button>
    {expanded && <div className="px-3 pb-4 space-y-2">{children}</div>}
  </div>
);

const MemberRow = ({ name, email, role, status, initials, color }) => (
  <div className="p-5 flex items-center justify-between hover:bg-slate-50 transition-colors group">
    <div className="flex items-center gap-4">
      <div className="relative">
        <div className={`w-11 h-11 rounded-2xl bg-gradient-to-br ${color} flex items-center justify-center text-white text-xs font-black shadow-lg shadow-indigo-100`}>
          {initials}
        </div>
        <div className={`absolute -bottom-1 -right-1 w-4 h-4 rounded-full border-[3px] border-white ${status === 'online' ? 'bg-emerald-500' : 'bg-slate-300'}`}></div>
      </div>
      <div>
        <p className="text-sm font-black text-slate-900 tracking-tight">{name}</p>
        <p className="text-[11px] text-slate-400 font-medium">{email}</p>
      </div>
    </div>
    <div className="flex items-center gap-6">
      <span className={`px-4 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest border ${role === 'Owner' ? 'bg-indigo-50 border-indigo-100 text-indigo-600' : 'bg-slate-50 border-slate-100 text-slate-500'}`}>{role}</span>
      <button className="p-2 text-slate-300 hover:text-slate-900 hover:bg-white rounded-lg transition-all shadow-sm">
        <Settings size={16} />
      </button>
    </div>
  </div>
);

const ControlIcon = ({ icon }) => (
  <button className="p-2.5 text-white/50 hover:text-white hover:bg-white/10 rounded-2xl transition-all active:scale-90">
    {icon}
  </button>
);

const MetadataRow = ({ label, value }) => (
  <div className="flex justify-between items-center p-3 bg-slate-50 border border-slate-100 rounded-xl">
    <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">{label}</span>
    <span className="text-[11px] font-black text-slate-900 tracking-tight">{value}</span>
  </div>
);

export default AuraFlowStudioPro;