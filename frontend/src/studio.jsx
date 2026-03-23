import React, { useState, useEffect, useRef } from 'react';
import { useTheme } from './ThemeContext';
import {
  Play, Pause, Volume2, VolumeX, Maximize, Settings, Subtitles,
  Palette, Zap, Globe, Share2, Plus, Search,
  ZoomIn, ZoomOut, User, Menu, ChevronRight,
  Video, Music, Layers, Download, Type, Wand2,
  FolderOpen, ChevronLeft, ChevronRight as ChevronRightIcon,
  Trash2, Copy, Eye, Sparkles, Smile, Cloud, Activity, Minimize2,
  Clock, HardDrive, Cpu, MousePointer2, Command, Bell, Upload, Loader2,
  Scissors, GripVertical, Move
} from 'lucide-react';
import api from './services/api';
import apiMode from './services/apiMode';

const AuraFlowStudioPro = () => {
  const { isDark } = useTheme();
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

  // --- VIDEO LOADING STATES ---
  const [jobStatus, setJobStatus] = useState('idle');
  const [jobProgress, setJobProgress] = useState('');
  const [convertedVideoUrl, setConvertedVideoUrl] = useState(null);
  const [processingMode, setProcessingMode] = useState('backend'); // 'backend' | 'api'
  const videoRef = useRef(null);

  const [expandedSections, setExpandedSections] = useState({
    videos: true,
    audio: true,
    images: false,
    recent: true
  });

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const [duration, setDuration] = useState(180);
  const timelineRef = useRef(null);
  const fileInputRef = useRef(null);
  const uploadInputRef = useRef(null);
  const [activeTrackIdForAdd, setActiveTrackIdForAdd] = useState(null);

  // --- EDITING STATES ---
  const [editMode, setEditMode] = useState('select'); // 'select' | 'cut' | 'move'
  const [selectedClipId, setSelectedClipId] = useState(null);
  const [tracks, setTracks] = useState([
    {
      id: 'video-1',
      type: 'video',
      label: 'Video Track',
      muted: false,
      clips: []
    },
    {
      id: 'audio-1',
      type: 'audio',
      label: 'AI Dubbed Audio',
      muted: false,
      clips: []
    },
    {
      id: 'caption-1',
      type: 'caption',
      label: 'Captions',
      muted: false,
      clips: []
    }
  ]);

  // Editing functions - VIRTUAL CUT (non-destructive)
  const handleCutClip = (trackId, clipId, cutTime) => {
    setTracks(prevTracks => prevTracks.map(track => {
      if (track.id !== trackId) return track;

      const clipIndex = track.clips.findIndex(c => c.id === clipId);
      if (clipIndex === -1) return track;

      const clip = track.clips[clipIndex];

      // Ensure cut is within clip bounds
      if (cutTime <= clip.startTime || cutTime >= clip.endTime) return track;

      // Create two virtual clips from one physical clip
      // Both reference the same source file, just different time ranges
      const newClip1 = {
        ...clip,
        endTime: cutTime,
        id: `${clip.id}-part1-${Date.now()}`,
        // Store original source reference
        originalSrc: clip.originalSrc || clip.src,
        // For video clips, we'd use these for playback
        sourceStartTime: clip.sourceStartTime || 0,
        sourceEndTime: cutTime - clip.startTime + (clip.sourceStartTime || 0)
      };

      const newClip2 = {
        ...clip,
        startTime: cutTime,
        id: `${clip.id}-part2-${Date.now()}`,
        label: `${clip.label} (cut)`,
        originalSrc: clip.originalSrc || clip.src,
        sourceStartTime: cutTime - clip.startTime + (clip.sourceStartTime || 0),
        sourceEndTime: clip.sourceEndTime || (clip.endTime - clip.startTime)
      };

      const newClips = [...track.clips];
      newClips.splice(clipIndex, 1, newClip1, newClip2);

      return { ...track, clips: newClips };
    }));
    setEditMode('select');
    setSelectedClipId(null);
  };

  const handleDeleteClip = (trackId, clipId) => {
    setTracks(prevTracks => prevTracks.map(track => {
      if (track.id !== trackId) return track;
      return { ...track, clips: track.clips.filter(c => c.id !== clipId) };
    }));
    setSelectedClipId(null);
  };

  const handleToggleTrackMute = (trackId) => {
    setTracks(prevTracks => prevTracks.map(track => {
      if (track.id !== trackId) return track;
      return { ...track, muted: !track.muted };
    }));
  };

  const handleAddClip = (trackId) => {
    setActiveTrackIdForAdd(trackId);
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (!file || !activeTrackIdForAdd) return;

    const fileUrl = URL.createObjectURL(file);

    setTracks(prevTracks => prevTracks.map(track => {
      if (track.id !== activeTrackIdForAdd) return track;

      // Find the end of the last clip
      const lastEnd = track.clips.length > 0
        ? Math.max(...track.clips.map(c => c.endTime))
        : 0;

      // Default duration based on file type
      const defaultDuration = 15;

      const newClip = {
        id: `${activeTrackIdForAdd}-clip-${Date.now()}`,
        startTime: lastEnd + 1,
        endTime: Math.min(lastEnd + 1 + defaultDuration, duration),
        label: file.name,
        src: fileUrl,
        originalSrc: fileUrl,
        type: track.type,
        sourceStartTime: 0,
        sourceEndTime: defaultDuration,
        color: track.type === 'video'
          ? 'bg-gradient-to-r from-emerald-400 to-teal-500'
          : track.type === 'audio'
            ? 'bg-gradient-to-r from-amber-400 to-orange-500'
            : 'bg-gradient-to-r from-cyan-400 to-blue-500'
      };

      return { ...track, clips: [...track.clips, newClip] };
    }));

    // Reset selection
    e.target.value = null;
    setActiveTrackIdForAdd(null);
  };

  const handleClipDrag = (trackId, clipId, newStartTime) => {
    setTracks(prevTracks => prevTracks.map(track => {
      if (track.id !== trackId) return track;

      return {
        ...track,
        clips: track.clips.map(clip => {
          if (clip.id !== clipId) return clip;
          const clipDuration = clip.endTime - clip.startTime;
          const boundedStart = Math.max(0, Math.min(newStartTime, duration - clipDuration));
          return {
            ...clip,
            startTime: boundedStart,
            endTime: boundedStart + clipDuration
          };
        })
      };
    }));
  };

  const handleClipResize = (trackId, clipId, newEndTime) => {
    setTracks(prevTracks => prevTracks.map(track => {
      if (track.id !== trackId) return track;

      return {
        ...track,
        clips: track.clips.map(clip => {
          if (clip.id !== clipId) return clip;
          const boundedEnd = Math.max(clip.startTime + 1, Math.min(newEndTime, duration));
          return { ...clip, endTime: boundedEnd };
        })
      };
    }));
  };

  const translations = {
    TA: "AUREX AI டப்பிங் மூலம் உங்கள் வேலையை அடுத்த கட்டத்திற்கு உயர்த்துங்கள். 🚀",
    JP: "AUREX AIダビングでコンテンツをレベルアップ 🚀",
    DE: "Verbessern Sie Ihre Inhalte mit AUREX AI Dubbing 🚀",
    EN: "Level up your content with AUREX AI Dubbing 🚀"
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  // Dragging state for playhead
  const [isDraggingPlayhead, setIsDraggingPlayhead] = useState(false);
  const [isInGap, setIsInGap] = useState(false); // Track if playhead is in a gap

  // ============================================================
  // TIMELINE MAP ARCHITECTURE (NLE-grade)
  // Timeline is authoritative, video is just a renderer
  // ============================================================

  /**
   * Build timeline map from tracks
   * Creates segments representing clips and gaps
   * Each segment maps timeline time → source time
   */
  const buildTimelineMap = () => {
    const videoTrack = tracks.find(t => t.type === 'video' && !t.muted);
    if (!videoTrack || videoTrack.clips.length === 0) return [];

    const clips = [...videoTrack.clips].sort((a, b) => a.startTime - b.startTime);
    const segments = [];
    let cursor = 0;

    for (const clip of clips) {
      // Gap before this clip
      if (clip.startTime > cursor) {
        segments.push({
          timelineStart: cursor,
          timelineEnd: clip.startTime,
          sourceStart: null,  // null = gap (blank/silent)
          sourceEnd: null,
          isGap: true
        });
      }

      // The clip segment
      segments.push({
        timelineStart: clip.startTime,
        timelineEnd: clip.endTime,
        sourceStart: clip.sourceStartTime ?? 0,
        sourceEnd: clip.sourceEndTime ?? (clip.endTime - clip.startTime),
        clipId: clip.id,
        src: clip.originalSrc || clip.src,
        isGap: false
      });

      cursor = clip.endTime;
    }

    return segments;
  };

  /**
   * Get maximum playable time on the timeline
   * Cursor/playhead/seek cannot exceed this
   */
  const getMaxPlayableTime = (timelineMap) => {
    if (!timelineMap || timelineMap.length === 0) return duration;

    // Find last playable segment (not a gap)
    const lastPlayable = [...timelineMap]
      .reverse()
      .find(seg => !seg.isGap);

    return lastPlayable ? lastPlayable.timelineEnd : 0;
  };

  /**
   * Get segment at a given timeline time
   */
  const getSegmentAtTime = (timelineTime, timelineMap) => {
    return timelineMap.find(
      seg => timelineTime >= seg.timelineStart && timelineTime < seg.timelineEnd
    );
  };

  /**
   * Sync video element to timeline time
   * This is where the magic happens - timeline controls video, not vice versa
   */
  const syncVideoToTimeline = (timelineTime, timelineMap) => {
    if (!videoRef.current || !convertedVideoUrl) return;

    const segment = getSegmentAtTime(timelineTime, timelineMap);

    if (!segment || segment.isGap) {
      // In a gap - pause video, show black
      videoRef.current.pause();
      setIsInGap(true);
      return;
    }

    setIsInGap(false);

    // Calculate offset within the clip
    const offset = timelineTime - segment.timelineStart;
    const targetVideoTime = segment.sourceStart + offset;

    // Only seek if significantly different (avoid jitter)
    if (Math.abs(videoRef.current.currentTime - targetVideoTime) > 0.1) {
      videoRef.current.currentTime = targetVideoTime;
    }

    // Resume playback if we're supposed to be playing
    if (isPlaying && videoRef.current.paused) {
      videoRef.current.play().catch(console.error);
    }
  };

  /**
   * Get the next valid (non-gap) segment start time after current time
   */
  const getNextSegmentStart = (timelineTime, timelineMap) => {
    const nextSeg = timelineMap.find(
      seg => !seg.isGap && seg.timelineStart > timelineTime
    );
    return nextSeg ? nextSeg.timelineStart : null;
  };

  // ============================================================
  // TIMELINE-DRIVEN PLAYBACK LOOP
  // Replaces free-running video playback
  // ============================================================

  useEffect(() => {
    if (!isPlaying) return;

    const timelineMap = buildTimelineMap();
    const maxTime = getMaxPlayableTime(timelineMap);

    // If no clips, nothing to play
    if (timelineMap.length === 0 || maxTime === 0) {
      setIsPlaying(false);
      return;
    }

    const tick = () => {
      setCurrentTime(prev => {
        let nextTime = prev + 0.04; // ~25fps timeline tick

        // Check if we're in a gap
        const segment = getSegmentAtTime(nextTime, timelineMap);

        if (segment && segment.isGap) {
          // Skip over the gap to next clip
          const nextStart = getNextSegmentStart(prev, timelineMap);
          if (nextStart !== null) {
            nextTime = nextStart;
          } else {
            // No more content - stop
            setIsPlaying(false);
            return maxTime;
          }
        }

        // Stop at max playable time
        if (nextTime >= maxTime) {
          setIsPlaying(false);
          return maxTime;
        }

        // Sync video to this timeline position
        syncVideoToTimeline(nextTime, timelineMap);

        return nextTime;
      });
    };

    const intervalId = setInterval(tick, 40); // 25fps
    return () => clearInterval(intervalId);
  }, [isPlaying, tracks, convertedVideoUrl]);

  // Update progress when currentTime changes
  useEffect(() => {
    if (duration > 0) {
      setProgress((currentTime / duration) * 100);
    }
  }, [currentTime, duration]);

  // Pause video when playback stops
  useEffect(() => {
    if (!isPlaying && videoRef.current && !videoRef.current.paused) {
      videoRef.current.pause();
    }
  }, [isPlaying]);

  // Disable video's native onTimeUpdate, we drive playback ourselves
  const handleVideoTimeUpdate = () => {
    // Intentionally empty - timeline drives playback, not video
  };

  // ============================================================
  // SEEK / SCRUB with timeline limits
  // ============================================================

  /**
   * Seek to a timeline position (respects max playable time)
   */
  const seekToTime = (timeInSeconds) => {
    const timelineMap = buildTimelineMap();
    const maxTime = getMaxPlayableTime(timelineMap);

    // Bound to [0, maxTime]
    const boundedTime = Math.max(0, Math.min(timeInSeconds, maxTime));

    setCurrentTime(boundedTime);

    // Sync video immediately
    syncVideoToTimeline(boundedTime, timelineMap);
  };

  // Calculate timeline area dimensions
  const getTimelineArea = () => {
    if (!timelineRef.current) return null;

    const rect = timelineRef.current.getBoundingClientRect();
    const TRACK_LABEL_WIDTH = 140; // Must match the w-32 class (128px) + gap
    const PADDING_RIGHT = 16;

    return {
      left: rect.left + TRACK_LABEL_WIDTH,
      width: rect.width - TRACK_LABEL_WIDTH - PADDING_RIGHT,
      rect
    };
  };

  // Handle timeline click to set playhead position
  const handleTimelineClick = (e) => {
    if (isDraggingPlayhead) return;

    const timelineArea = getTimelineArea();
    if (!timelineArea) return;

    const clickX = e.clientX - timelineArea.left;

    // Ignore clicks outside the track area
    if (clickX < 0 || clickX > timelineArea.width) return;

    const clickPercent = clickX / timelineArea.width;
    const newTime = clickPercent * duration;

    seekToTime(newTime);
  };

  // Handle playhead drag
  const handlePlayheadMouseDown = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDraggingPlayhead(true);

    const handleMouseMove = (moveEvent) => {
      const timelineArea = getTimelineArea();
      if (!timelineArea) return;

      const moveX = moveEvent.clientX - timelineArea.left;
      const movePercent = Math.max(0, Math.min(moveX / timelineArea.width, 1));
      const newTime = movePercent * duration;

      seekToTime(newTime);
    };

    const handleMouseUp = () => {
      setIsDraggingPlayhead(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  // ─── API MODE PIPELINE HANDLER ─────────────────────────────────────
  // Runs when processingMode === 'api': calls OpenAI Whisper, GPT, TTS
  // via thin backend helpers. Backend AI pipeline is never touched.
  const handleApiModeUpload = async (file) => {
    try {
      setJobStatus('loading');
      setJobProgress('🔑 Loading API key...');

      const result = await apiMode.runApiModePipeline(
        file,
        activeLang,
        (msg) => setJobProgress(msg),
      );

      setConvertedVideoUrl(result.videoUrl);

      // Populate timeline tracks
      setTracks(prev => [
        {
          id: 'video-1',
          type: 'video',
          label: 'Video Track',
          muted: false,
          clips: [{
            id: `v1-${result.jobId}`,
            startTime: 0,
            endTime: duration || 180,
            label: file.name,
            src: result.videoUrl,
            originalSrc: result.videoUrl,
            sourceStartTime: 0,
            sourceEndTime: duration || 180,
            color: 'bg-gradient-to-r from-violet-400 to-purple-500',
          }],
        },
        {
          id: 'audio-1',
          type: 'audio',
          label: 'AI Dubbed Audio',
          muted: false,
          clips: [{
            id: `a1-${result.jobId}`,
            startTime: 0,
            endTime: duration || 180,
            label: 'OpenAI TTS',
            src: result.videoUrl,
            originalSrc: result.videoUrl,
            sourceStartTime: 0,
            sourceEndTime: duration || 180,
            color: 'bg-gradient-to-r from-violet-400 to-fuchsia-500',
          }],
        },
        { id: 'caption-1', type: 'caption', label: 'Captions', muted: false, clips: [] },
      ]);

      setJobStatus('completed');
      setTimeout(() => setJobProgress(''), 3000);

    } catch (err) {
      setJobStatus('failed');
      setJobProgress(`❌ ${err.message}`);
      console.error('API mode pipeline error:', err);
    }
  };

  // ─── UNIFIED UPLOAD HANDLER ─────────────────────────────────────
  // Dispatches to backend or API mode based on the mode switcher
  const handleMainVideoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (processingMode === 'api') {
      await handleApiModeUpload(file);
      e.target.value = null;
      return;
    }

    // ── Original backend flow ──────────────────────────────────────
    try {
      setJobStatus('loading');
      setJobProgress('Uploading video...');

      // 1. Upload Video
      const uploadResult = await api.uploadVideo(file, activeLang);
      const jobId = uploadResult.job_id;

      setJobStatus('loading');
      setJobProgress('Processing video with AI...');

      // 2. Poll for completion
      await api.pollJobStatus(jobId, (status, progress) => {
        setJobProgress(`Processing: ${status} - ${progress || '...'}`);
      });

      setJobProgress('Finalizing...');

      // 3. Get Download URL
      const videoUrl = api.getDownloadUrl(jobId, 'video');

      // Set the URL to trigger video load
      setConvertedVideoUrl(videoUrl);

      // Update tracks with the new video
      setTracks(prev => [
        {
          id: 'video-1',
          type: 'video',
          label: 'Video Track',
          muted: false,
          clips: [
            {
              id: `v1-${jobId}`,
              startTime: 0,
              endTime: duration || 180,
              label: file.name,
              src: videoUrl,
              originalSrc: videoUrl,
              sourceStartTime: 0,
              sourceEndTime: duration || 180,
              color: 'bg-gradient-to-r from-slate-300 to-slate-400'
            }
          ]
        },
        {
          id: 'audio-1',
          type: 'audio',
          label: 'Original Audio',
          muted: false,
          clips: [
            {
              id: `a1-${jobId}`,
              startTime: 0,
              endTime: duration || 180,
              label: 'Original Audio',
              src: videoUrl,
              originalSrc: videoUrl,
              sourceStartTime: 0,
              sourceEndTime: duration || 180,
              color: 'bg-gradient-to-r from-indigo-400 to-purple-500'
            }
          ]
        },
        {
          id: 'caption-1',
          type: 'caption',
          label: 'Captions',
          muted: false,
          clips: []
        }
      ]);

      setJobStatus('completed');
      setJobProgress('Video processed successfully!');

      // Clear progress message after 2 seconds
      setTimeout(() => {
        setJobProgress('');
      }, 2000);

    } catch (error) {
      setJobStatus('failed');
      setJobProgress(`Error: ${error.message}`);
      console.error('Video processing error:', error);
    } finally {
      e.target.value = null; // Reset input
    }
  };

  // Calculate playhead position in pixels
  const getPlayheadPosition = () => {
    const timelineArea = getTimelineArea();
    if (!timelineArea) return 140; // Default to label width

    const TRACK_LABEL_WIDTH = 140;
    const progressPercent = (currentTime / duration);
    const trackAreaWidth = timelineArea.width;

    return TRACK_LABEL_WIDTH + (progressPercent * trackAreaWidth);
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
              <div className={`group relative rounded-[32px] overflow-hidden bg-[#0F172A] shadow-[0_20px_50px_rgba(0,0,0,0.2)] border border-slate-200/10 transition-all duration-700 flex items-center justify-center ${isFullscreen ? 'rounded-none border-none h-full w-full' : 'min-h-[300px] max-h-[70vh]'}`}>

                <div
                  className="w-full h-full flex items-center justify-center transition-all duration-1000 ease-in-out"
                  style={{ filter: videoFilter }}
                >
                  {convertedVideoUrl ? (
                    <video
                      ref={videoRef}
                      src={convertedVideoUrl}
                      className="max-w-full max-h-full object-contain"
                      onLoadedMetadata={(e) => {
                        const videoDuration = e.target.duration;
                        if (videoDuration && !isNaN(videoDuration)) {
                          setDuration(videoDuration);
                        }
                      }}
                      onTimeUpdate={handleVideoTimeUpdate}
                    // Don't respond to ended - timeline drives playback
                    />
                  ) : (
                    <img
                      src="https://images.unsplash.com/photo-1574717024653-61fd2cf4d44d?w=800"
                      className={`max-w-full max-h-full object-contain transition-transform duration-[10s] ease-linear ${isPlaying ? 'scale-110' : 'scale-100'}`}
                      alt="Video Preview"
                    />
                  )}

                  {/* GAP INDICATOR - Shows black screen when in gap */}
                  {isInGap && (
                    <div className="absolute inset-0 bg-black flex items-center justify-center z-20">
                      <div className="text-center">
                        <div className="text-slate-500 text-sm font-bold mb-2">GAP</div>
                        <div className="text-slate-600 text-xs">No content at this position</div>
                      </div>
                    </div>
                  )}
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
                      REC • {formatTime(currentTime)}
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
                      <span className="text-indigo-400 font-mono text-sm font-bold">{formatTime(currentTime)}</span>
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

              {/* TIMELINE EDITING BAR */}
              {!isFullscreen && (
                <div className={`border rounded-[24px] overflow-hidden shadow-xl transition-all hover:shadow-2xl ${isDark ? 'bg-slate-900 border-slate-700 shadow-slate-900/50' : 'bg-white border-slate-200 shadow-slate-200/50'}`}>

                  {/* Editing Toolbar */}
                  <div className={`px-6 py-3 border-b flex items-center justify-between ${isDark ? 'bg-slate-800/50 border-slate-700' : 'bg-gradient-to-r from-slate-50 to-white border-slate-100'}`}>
                    <div className="flex items-center gap-2">
                      {/* Tool Buttons */}
                      <EditToolButton
                        icon={<MousePointer2 size={16} />}
                        label="Select"
                        active={editMode === 'select'}
                        onClick={() => setEditMode('select')}
                        isDark={isDark}
                      />
                      <EditToolButton
                        icon={<Move size={16} />}
                        label="Move"
                        active={editMode === 'move'}
                        onClick={() => setEditMode('move')}
                        isDark={isDark}
                      />
                      <EditToolButton
                        icon={<Scissors size={16} />}
                        label="Cut"
                        active={editMode === 'cut'}
                        onClick={() => setEditMode('cut')}
                        isDark={isDark}
                        color="text-rose-500"
                      />

                      <div className={`w-px h-8 mx-2 ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`}></div>

                      <EditToolButton
                        icon={<Trash2 size={16} />}
                        label="Delete Selected"
                        onClick={() => {
                          if (selectedClipId) {
                            const track = tracks.find(t => t.clips.some(c => c.id === selectedClipId));
                            if (track) handleDeleteClip(track.id, selectedClipId);
                          }
                        }}
                        disabled={!selectedClipId}
                        isDark={isDark}
                        color="text-rose-500"
                      />

                      <div className={`w-px h-8 mx-2 ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`}></div>

                      <EditToolButton
                        icon={<ZoomIn size={16} />}
                        label="Zoom In"
                        onClick={() => setDuration(prev => Math.max(30, prev - 30))}
                        isDark={isDark}
                      />
                      <EditToolButton
                        icon={<ZoomOut size={16} />}
                        label="Zoom Out"
                        onClick={() => setDuration(prev => Math.min(600, prev + 30))}
                        isDark={isDark}
                      />
                    </div>

                    <div className="flex items-center gap-4">
                      <div className={`text-[10px] font-black tracking-[0.2em] uppercase flex items-center gap-2 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                        <div className={`w-2 h-2 rounded-full animate-pulse ${editMode === 'cut' ? 'bg-rose-500' : 'bg-indigo-500'}`}></div>
                        {editMode === 'cut' ? 'CUT MODE - Click on clip to split' : 'Timeline Editor'}
                      </div>
                      <div className={`font-mono text-[13px] font-bold px-4 py-1.5 rounded-xl border ${isDark ? 'text-indigo-400 bg-indigo-950 border-indigo-800' : 'text-indigo-600 bg-indigo-50 border-indigo-100'}`}>
                        {formatTime(currentTime)} / {formatTime(duration)}
                      </div>
                    </div>
                  </div>

                  {/* Timeline Tracks Area */}
                  <div
                    className={`p-6 space-y-3 relative ${isDark ? 'bg-slate-900' : 'bg-[#FBFCFD]'}`}
                    ref={timelineRef}
                  >
                    {/* Playhead - FIXED POSITIONING */}
                    <div
                      className="absolute top-0 bottom-0 w-0.5 bg-indigo-500 z-40 pointer-events-none"
                      style={{ left: `${getPlayheadPosition()}px` }}
                    >
                      {/* Draggable handle */}
                      <div
                        className={`absolute -top-1 -left-3 w-6 h-6 flex items-center justify-center z-50 pointer-events-auto ${isDraggingPlayhead ? 'cursor-grabbing' : 'cursor-grab'}`}
                        onMouseDown={handlePlayheadMouseDown}
                      >
                        <div className="w-4 h-4 bg-indigo-500 rounded-full shadow-lg shadow-indigo-500/50 hover:scale-125 transition-transform"></div>
                      </div>
                      <div className="absolute -top-0 -left-[5px] w-[10px] h-3 bg-indigo-500 pointer-events-none" style={{ clipPath: 'polygon(50% 100%, 0 0, 100% 0)' }}></div>
                    </div>

                    {/* Time Ruler */}
                    <div className={`h-7 w-full flex items-end pl-[140px] pr-4 text-[9px] font-bold border-b mb-4 ${isDark ? 'text-slate-500 border-slate-800' : 'text-slate-400 border-slate-100'}`}>
                      <div className="flex-1 flex justify-between">
                        {[...Array(11)].map((_, i) => {
                          const timeAtMarker = Math.round((i / 10) * duration);
                          return (
                            <div key={i} className="flex flex-col items-center">
                              <div className={`w-px h-2 mb-1 ${isDark ? 'bg-slate-700' : 'bg-slate-200'}`}></div>
                              <span>{formatTime(timeAtMarker)}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Dynamic Tracks */}
                    {tracks.map((track) => (
                      <div key={track.id} className="flex items-center gap-4 h-16 group">
                        {/* Track Label */}
                        <div className={`w-32 flex items-center gap-2 shrink-0 ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>
                          <div className={`p-1.5 rounded-lg shadow-sm border ${isDark ? 'bg-slate-800 border-slate-700' : 'bg-white border-slate-100'}`}>
                            {track.type === 'video' && <Video size={14} />}
                            {track.type === 'audio' && (track.muted ? <VolumeX size={14} className="text-rose-500" /> : <Volume2 size={14} />)}
                            {track.type === 'caption' && <Subtitles size={14} />}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="text-[10px] font-black uppercase tracking-tight truncate">{track.label}</div>
                            <div className="text-[9px] text-slate-400">{track.clips.length} clip{track.clips.length !== 1 ? 's' : ''}</div>
                          </div>
                          {/* Track Actions */}
                          <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            {track.type === 'audio' && (
                              <button
                                onClick={() => handleToggleTrackMute(track.id)}
                                className={`p-1 rounded-md transition-colors ${track.muted ? 'bg-rose-100 text-rose-600' : 'hover:bg-slate-100 text-slate-400'}`}
                                title={track.muted ? 'Unmute' : 'Mute'}
                              >
                                {track.muted ? <VolumeX size={12} /> : <Volume2 size={12} />}
                              </button>
                            )}
                            <button
                              onClick={() => handleAddClip(track.id)}
                              className={`p-1 rounded-md transition-colors ${isDark ? 'hover:bg-slate-700 text-slate-400' : 'hover:bg-slate-100 text-slate-400'}`}
                              title="Add clip"
                            >
                              <Plus size={12} />
                            </button>
                          </div>
                        </div>

                        {/* Track Timeline */}
                        <div
                          className={`timeline-track-bg flex-1 h-12 rounded-xl border relative overflow-hidden ${editMode === 'cut' ? 'cursor-crosshair' : 'cursor-pointer'} ${isDark ? 'bg-slate-800 border-slate-700' : 'bg-slate-100/50 border-slate-200/60'} ${track.muted ? 'opacity-50' : ''}`}
                          onClick={(e) => {
                            // Only handle clicks on empty timeline area, not on clips
                            if (e.target.classList.contains('timeline-track-bg')) {
                              handleTimelineClick(e);
                            }
                          }}
                        >
                          {/* Clips */}
                          {track.clips.map((clip) => {
                            const leftPercent = (clip.startTime / duration) * 100;
                            const widthPercent = ((clip.endTime - clip.startTime) / duration) * 100;
                            const isSelected = selectedClipId === clip.id;

                            return (
                              <div
                                key={clip.id}
                                className={`absolute top-1 bottom-1 rounded-lg transition-all group/clip ${clip.color} ${isSelected ? 'ring-2 ring-indigo-500 ring-offset-2 shadow-lg z-20' : 'hover:brightness-110 z-10'} ${editMode === 'cut' ? 'cursor-crosshair' : editMode === 'move' ? 'cursor-move' : 'cursor-pointer'}`}
                                style={{ left: `${leftPercent}%`, width: `${widthPercent}%` }}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  if (editMode === 'cut') {
                                    // Calculate cut time based on click position within the clip
                                    const clipElement = e.currentTarget;
                                    const clipRect = clipElement.getBoundingClientRect();
                                    const clickX = e.clientX - clipRect.left;
                                    const clickPercent = clickX / clipRect.width;
                                    const clipDuration = clip.endTime - clip.startTime;
                                    const cutTime = clip.startTime + (clickPercent * clipDuration);

                                    handleCutClip(track.id, clip.id, cutTime);
                                  } else {
                                    setSelectedClipId(isSelected ? null : clip.id);
                                  }
                                }}
                                draggable={editMode === 'move'}
                                onDragStart={(e) => {
                                  if (editMode !== 'move') return;
                                  e.dataTransfer.setData('clipId', clip.id);
                                  e.dataTransfer.setData('trackId', track.id);
                                  e.dataTransfer.effectAllowed = 'move';
                                }}
                                onDragEnd={(e) => {
                                  if (editMode !== 'move') return;
                                  const rect = e.currentTarget.parentElement.getBoundingClientRect();
                                  const dropX = e.clientX - rect.left;
                                  const newStartPercent = dropX / rect.width;
                                  const newStartTime = newStartPercent * duration;
                                  handleClipDrag(track.id, clip.id, newStartTime);
                                }}
                              >
                                {/* Clip Content */}
                                <div className="absolute inset-0 flex items-center justify-between px-2 overflow-hidden">
                                  <span className="text-[9px] font-bold text-white/90 truncate drop-shadow-sm">
                                    {clip.label}
                                  </span>
                                  <span className="text-[8px] font-mono text-white/70 shrink-0 ml-2">
                                    {formatTime(clip.endTime - clip.startTime)}
                                  </span>
                                </div>

                                {/* Audio waveform visualization */}
                                {track.type === 'audio' && !track.muted && (
                                  <div className="absolute inset-0 flex items-center justify-center gap-[2px] opacity-30 px-2">
                                    {[...Array(Math.min(40, Math.floor(widthPercent * 2)))].map((_, i) => (
                                      <div
                                        key={i}
                                        className="w-[2px] bg-white rounded-full"
                                        style={{ height: `${20 + Math.sin(i * 0.8) * 50}%` }}
                                      ></div>
                                    ))}
                                  </div>
                                )}

                                {/* Resize Handle (right edge) */}
                                {isSelected && editMode !== 'cut' && (
                                  <div
                                    className="absolute right-0 top-0 bottom-0 w-3 cursor-ew-resize bg-white/20 hover:bg-white/40 transition-colors flex items-center justify-center z-30"
                                    onMouseDown={(e) => {
                                      e.stopPropagation();
                                      const startX = e.clientX;
                                      const startEnd = clip.endTime;

                                      const handleMouseMove = (moveEvent) => {
                                        const delta = moveEvent.clientX - startX;
                                        const trackWidth = e.currentTarget.parentElement.parentElement.offsetWidth;
                                        const timeDelta = (delta / trackWidth) * duration;
                                        handleClipResize(track.id, clip.id, startEnd + timeDelta);
                                      };

                                      const handleMouseUp = () => {
                                        document.removeEventListener('mousemove', handleMouseMove);
                                        document.removeEventListener('mouseup', handleMouseUp);
                                      };

                                      document.addEventListener('mousemove', handleMouseMove);
                                      document.addEventListener('mouseup', handleMouseUp);
                                    }}
                                  >
                                    <GripVertical size={10} className="text-white/60" />
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    ))}

                    {/* Add Track Button */}
                    <div className="flex items-center justify-center pt-4">
                      <button
                        onClick={() => {
                          const newTrackId = `track-${Date.now()}`;
                          setTracks(prev => [...prev, {
                            id: newTrackId,
                            type: 'audio',
                            label: `Audio Track ${prev.filter(t => t.type === 'audio').length + 1}`,
                            muted: false,
                            clips: []
                          }]);
                        }}
                        className={`px-4 py-2 rounded-xl text-[11px] font-bold uppercase tracking-wider flex items-center gap-2 transition-all ${isDark ? 'text-slate-500 hover:text-slate-300 hover:bg-slate-800 border border-slate-700' : 'text-slate-400 hover:text-slate-600 hover:bg-slate-100 border border-slate-200'}`}
                      >
                        <Plus size={14} /> Add Track
                      </button>
                    </div>
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
              <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-[0.2em]">Source Video</h3>
            </div>
            {/* NO VIDEO ID INPUT anymore */}
            <p className="text-[10px] text-slate-400 mt-2 font-medium">Select a video to upload and process</p>
          </section>

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

        <div className="p-8 border-t border-slate-100 bg-white space-y-4">
          {/* Status/Progress display */}
          {jobProgress && (
            <div className={`text-center text-[11px] font-medium py-2 px-4 rounded-xl ${
              jobProgress.startsWith('❌') ? 'bg-red-50 text-red-600' : 'bg-indigo-50 text-indigo-600'
            }`}>
              {jobProgress}
            </div>
          )}

          {/* ── Processing Mode Switcher ── */}
          <div className="flex items-center gap-2 p-1 bg-slate-100 rounded-2xl">
            <button
              onClick={() => setProcessingMode('backend')}
              className={`flex-1 py-2.5 rounded-xl text-[11px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-1.5 ${
                processingMode === 'backend'
                  ? 'bg-white text-slate-800 shadow-md ring-1 ring-slate-200'
                  : 'text-slate-400 hover:text-slate-600'
              }`}
            >
              <span>🖥</span> Backend
            </button>
            <button
              onClick={() => setProcessingMode('api')}
              className={`flex-1 py-2.5 rounded-xl text-[11px] font-black uppercase tracking-widest transition-all flex items-center justify-center gap-1.5 ${
                processingMode === 'api'
                  ? 'bg-gradient-to-r from-violet-500 to-fuchsia-600 text-white shadow-lg shadow-violet-200'
                  : 'text-slate-400 hover:text-slate-600'
              }`}
            >
              <span>⚡</span> API Mode
            </button>
          </div>

          {/* Mode description badge */}
          <p className="text-center text-[10px] font-semibold text-slate-400 -mt-2">
            {processingMode === 'api'
              ? '⚡ OpenAI Whisper · GPT · TTS — no GPU needed'
              : '🖥 Local pipeline · Whisper · CLIP · gTTS'}
          </p>

          {/* Upload / Process button */}
          <button
            onClick={() => uploadInputRef.current?.click()}
            disabled={jobStatus === 'loading'}
            className={`w-full py-4 rounded-[20px] text-[12px] font-black uppercase tracking-[0.1em] shadow-xl transition-all active:scale-95 flex items-center justify-center gap-3 group ${
              jobStatus === 'loading'
                ? 'bg-slate-400 text-white cursor-not-allowed'
                : processingMode === 'api'
                  ? 'bg-gradient-to-r from-violet-500 to-fuchsia-600 hover:from-violet-600 hover:to-fuchsia-700 text-white shadow-violet-200'
                  : 'bg-indigo-600 hover:bg-indigo-700 text-white shadow-indigo-100'
            }`}
          >
            {jobStatus === 'loading' && <Loader2 size={16} className="animate-spin" />}
            {jobStatus === 'loading' ? 'Processing...' : 'Upload & Process New Video'}
            {jobStatus === 'idle' && <Upload size={16} className="group-hover:-translate-y-1 transition-transform" />}
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

      {/* Hidden File Input for Add Clip */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileSelect}
        className="hidden"
        accept="video/*,audio/*,image/*"
      />

      {/* Hidden File Input for Main Video Upload */}
      <input
        type="file"
        ref={uploadInputRef}
        onChange={handleMainVideoUpload}
        className="hidden"
        accept="video/*"
      />
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

const EditToolButton = ({ icon, label, active, onClick, disabled, isDark, color }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className={`group relative px-3 py-2 rounded-xl text-[11px] font-bold uppercase tracking-wider flex items-center gap-2 transition-all ${disabled
      ? 'opacity-40 cursor-not-allowed'
      : active
        ? isDark
          ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-900/50'
          : 'bg-indigo-600 text-white shadow-lg shadow-indigo-200'
        : isDark
          ? `${color || 'text-slate-400'} hover:bg-slate-700 hover:text-white`
          : `${color || 'text-slate-500'} hover:bg-slate-100 hover:text-slate-700`
      }`}
  >
    {icon}
    <span className="hidden sm:inline">{label}</span>
    {/* Tooltip */}
    <div className={`absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-2 py-1 text-[9px] font-bold rounded-lg opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity whitespace-nowrap z-50 ${isDark ? 'bg-slate-700 text-white' : 'bg-slate-900 text-white'}`}>
      {label}
    </div>
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