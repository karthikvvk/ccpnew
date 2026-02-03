import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useTheme } from './ThemeContext';

const menu = {
  'getting-started': {
    title: 'Getting Started',
    items: [
      { id: 'introduction', label: 'Introduction' },
      { id: 'quick-start', label: 'Quick Start' },
      { id: 'architecture', label: 'Architecture' }
    ]
  },
  'core-concepts': {
    title: 'Core Concepts',
    items: [
      { id: 'voice-cloning', label: 'Voice Cloning' },
      { id: 'lip-sync', label: 'Neural Lip Sync' },
      { id: 'localization', label: 'Localization Workflow' }
    ]
  },
  'api-reference': {
    title: 'API Reference',
    items: [
      { id: 'authentication', label: 'Authentication' },
      { id: 'endpoints', label: 'Endpoints' },
      { id: 'webhooks', label: 'Webhooks' }
    ]
  },
  'guides': {
    title: 'Guides',
    items: [
      { id: 'best-practices', label: 'Best Practices' },
      { id: 'troubleshooting', label: 'Troubleshooting' }
    ]
  }
};

const Documentation = () => {
  const { isDark, toggleTheme } = useTheme();
  const [activeCategory, setActiveCategory] = useState('getting-started');
  const [activeDoc, setActiveDoc] = useState('introduction');
  const [animating, setAnimating] = useState(false);
  // Helper to get all docs in order
  const menuOrder = Object.entries(menu).flatMap(([cat, section]) => section.items.map(item => ({ ...item, category: cat })));
  const currentIndex = menuOrder.findIndex(d => d.id === activeDoc && d.category === activeCategory);

  const goToDoc = (index) => {
    if (index < 0 || index >= menuOrder.length) return;
    setAnimating(true);
    setTimeout(() => {
      setActiveCategory(menuOrder[index].category);
      setActiveDoc(menuOrder[index].id);
      setAnimating(false);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }, 250);
  };

  const renderContent = () => {
    switch (activeDoc) {
      case 'introduction':
        return (
          <div className="space-y-6">
            <h1 className="text-4xl font-bold mb-6 text-theme-primary">Introduction to AuraFlow</h1>
            <p className="text-lg leading-relaxed text-theme-secondary">
              AuraFlow is an enterprise-grade AI dubbing and localization platform designed to break language barriers while preserving the authentic voice and intent of the original speaker.
            </p>
            <div className={`p-6 border rounded-xl my-6 ${isDark ? 'bg-blue-900/20 border-blue-800' : 'bg-indigo-50 border-indigo-100'}`}>
              <h3 className={`font-semibold mb-2 flex items-center gap-2 ${isDark ? 'text-blue-300' : 'text-indigo-900'}`}>
                <span className="material-symbols-rounded">info</span>
                Key Capabilities
              </h3>
              <ul className={`list-disc list-inside space-y-1 ml-1 ${isDark ? 'text-blue-200' : 'text-indigo-800'}`}>
                <li>High-fidelity Voice Cloning (200+ biomarkers)</li>
                <li>Frame-perfect Lip Synchronization</li>
                <li>40+ Language Support with Dialect Adaptation</li>
              </ul>
            </div>
            <p className="text-theme-secondary">
              Our documentation is organized into guides for creators, developers, and enterprise administrators. Select a section from the sidebar to get started.
            </p>
          </div>
        );
      case 'authentication':
        return (
          <div className="space-y-6">
            <h1 className="text-4xl font-bold text-theme-primary">Authentication</h1>
            <p className="text-theme-secondary">
              The AuraFlow API uses API keys to authenticate requests. You can view and manage your API keys in the Studio Dashboard.
            </p>
            <div className="rounded-lg p-6 overflow-x-auto shadow-lg my-6 group relative bg-theme-secondary border border-theme">
              <div className="flex justify-between items-center mb-4 text-theme-tertiary text-xs uppercase tracking-wider font-semibold border-b border-theme pb-2">
                <span>Bash</span>
                <button className="hover:text-theme-primary transition-colors">Copy</button>
              </div>
              <code className="font-mono text-sm text-blue-400">
                curl <span className="text-green-400">"https://api.auraflow.ai/v1/projects"</span> \<br />
                &nbsp;&nbsp;-H <span className="text-amber-400">"Authorization: Bearer YOUR_API_KEY"</span>
              </code>
            </div>
            <p className="text-theme-secondary">
              Your API keys carry many privileges, so be sure to keep them secure! Do not share your secret API keys in publicly accessible areas such as GitHub, client-side code, and so forth.
            </p>
          </div>
        );
      case 'voice-cloning':
        return (
          <div className="space-y-6">
            <h1 className="text-4xl font-bold text-theme-primary">Voice Cloning</h1>
            <p className="text-theme-secondary">
              AuraFlow's voice cloning engine captures both the timbre and prosody of a speaker.
            </p>
            <h3 className="text-2xl font-bold mt-8 text-theme-primary">Training Data Requirements</h3>
            <table className="w-full text-left border-collapse mt-4">
              <thead>
                <tr className="border-b text-sm border-theme text-theme-tertiary">
                  <th className="py-3 font-semibold">Quality Level</th>
                  <th className="py-3 font-semibold">Audio Length</th>
                  <th className="py-3 font-semibold">Use Case</th>
                </tr>
              </thead>
              <tbody className="text-theme-secondary">
                <tr className="border-b border-theme">
                  <td className="py-4 font-medium text-emerald-500">Instant</td>
                  <td className="py-4">30 seconds</td>
                  <td className="py-4">Social Media, Short clips</td>
                </tr>
                <tr className="border-b border-theme">
                  <td className="py-4 font-medium text-blue-500">Professional</td>
                  <td className="py-4">5-10 minutes</td>
                  <td className="py-4">Broadcast, Learning materials</td>
                </tr>
                <tr>
                  <td className="py-4 font-medium text-pink-500">Cinema</td>
                  <td className="py-4">30+ minutes</td>
                  <td className="py-4">Feature films, Character acting</td>
                </tr>
              </tbody>
            </table>
          </div>
        );
      default:
        return (
          <div className="flex flex-col items-center justify-center h-full text-center py-20 opacity-50">
            <span className="material-symbols-rounded text-6xl mb-4 text-theme-tertiary">construction</span>
            <h2 className="text-2xl font-bold text-theme-tertiary">Under Construction</h2>
            <p className="text-theme-tertiary">This section is being updated.</p>
          </div>
        );
    }
  };

  return (
    <div className="min-h-screen font-['Plus_Jakarta_Sans',_sans-serif] transition-colors duration-300 bg-theme-primary text-theme-primary">
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');
        
        /* Custom Scrollbar */
        .sidebar-scroll::-webkit-scrollbar {
            width: 5px;
        }
        .sidebar-scroll::-webkit-scrollbar-track {
            background: transparent;
        }
        .sidebar-scroll::-webkit-scrollbar-thumb {
            background: var(--border-subtle);
            border-radius: 20px;
        }
        .sidebar-scroll::-webkit-scrollbar-thumb:hover {
            background: var(--text-tertiary);
        }
      `}</style>

      {/* Header */}
      <nav className="fixed top-0 w-full z-50 border-b h-16 flex items-center justify-between px-6 transition-colors duration-300 bg-theme-primary border-theme">
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white">
              <span className="material-symbols-rounded text-xl">graphic_eq</span>
            </div>
            <span className="font-bold tracking-tight text-lg text-theme-primary">AuraFlow</span>
            <span className="px-2 italic text-theme-tertiary">/</span>
            <span className="font-medium text-theme-secondary">Docs</span>
          </Link>
        </div>
        <div className="flex items-center gap-4">
          <div className="hidden md:flex items-center relative">
            <span className="material-symbols-rounded absolute left-3 text-theme-tertiary text-lg">search</span>
            <input
              type="text"
              placeholder="Search documentation..."
              className="pl-10 pr-4 py-2 border-none rounded-lg text-sm w-64 focus:ring-2 focus:ring-indigo-500/50 transition-all outline-none bg-theme-secondary text-theme-primary focus:bg-white dark:focus:bg-zinc-800"
            />
            <div className="absolute right-3 flex gap-1">
              <kbd className="px-1.5 rounded text-[10px] border font-sans bg-theme-surface text-theme-tertiary border-theme">Ctrl</kbd>
              <kbd className="px-1.5 rounded text-[10px] border font-sans bg-theme-surface text-theme-tertiary border-theme">K</kbd>
            </div>
          </div>

          <button
            onClick={toggleTheme}
            className="relative w-14 h-9 rounded-full bg-slate-200 dark:bg-slate-800 transition-colors duration-300 flex items-center px-1 border border-slate-300 dark:border-slate-600 hover:border-indigo-500/50 focus:outline-none overflow-hidden"
            aria-label="Toggle Theme"
          >
            <div
              className={`absolute w-7 h-7 rounded-full bg-white dark:bg-[#020617] shadow-sm flex items-center justify-center transition-transform duration-300 overflow-hidden ${isDark ? 'translate-x-[20px]' : 'translate-x-0'}`}
            >
              <span className={`material-symbols-rounded text-[16px] absolute transition-all duration-300 ${isDark ? 'text-indigo-400 opacity-100 rotate-0 scale-100' : 'opacity-0 -rotate-90 scale-50'}`}>
                dark_mode
              </span>
              <span className={`material-symbols-rounded text-[16px] absolute text-amber-500 transition-all duration-300 ${isDark ? 'opacity-0 rotate-90 scale-50' : 'opacity-100 rotate-0 scale-100'}`}>
                light_mode
              </span>
            </div>
          </button>

          <Link to="/studio" className="px-4 py-2 bg-slate-900 dark:bg-zinc-100 text-white dark:text-black text-sm font-semibold rounded-lg hover:opacity-90 transition-opacity">
            Go to Studio
          </Link>
        </div>
      </nav>

      {/* Main Layout */}
      <div className="pt-16 flex min-h-screen">

        {/* Sidebar */}
        <aside className="w-64 fixed inset-y-0 left-0 pt-16 border-r sidebar-scroll overflow-y-auto hidden md:block transition-colors duration-300 bg-theme-secondary border-theme">
          <div className="p-6">
            {Object.entries(menu).map(([key, section]) => (
              <div key={key} className="mb-8">
                <h4 className="text-xs font-bold uppercase tracking-wider mb-3 text-theme-tertiary">{section.title}</h4>
                <ul className="space-y-1">
                  {section.items.map((item) => (
                    <li key={item.id}>
                      <button
                        onClick={() => {
                          setActiveCategory(key);
                          setActiveDoc(item.id);
                          window.scrollTo({ top: 0, behavior: 'smooth' });
                        }}
                        className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition-colors ${activeDoc === item.id
                          ? 'bg-indigo-500/10 text-indigo-600 dark:text-indigo-400'
                          : 'text-theme-secondary hover:text-theme-primary hover:bg-theme-surface'
                          }`}
                      >
                        {item.label}
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </aside>

        {/* Content Area */}
        <main className="flex-1 md:ml-64 transition-colors duration-300 bg-theme-primary">
          <div className="max-w-4xl mx-auto px-8 py-12 min-h-screen">
            {/* Breadcrumbs */}
            <div className="flex items-center gap-2 text-sm mb-8 text-theme-tertiary">
              <span className="hover:text-indigo-600 cursor-pointer">{menu[activeCategory].title}</span>
              <span className="material-symbols-rounded text-xs">chevron_right</span>
              <span className="font-semibold text-theme-primary">{menu[activeCategory].items.find(i => i.id === activeDoc)?.label}</span>
            </div>

            {/* Rendered Content */}
            <div className={`animate-fade-in transition-all duration-300 ${animating ? 'opacity-0 scale-95' : 'opacity-100 scale-100'}`}>
              {renderContent()}
            </div>
            {/* Page Navigation */}
            <div className="mt-20 pt-8 border-t flex justify-between border-theme">
              <button
                className={`text-sm font-medium flex items-center gap-1 group ${currentIndex <= 0 ? 'opacity-40 pointer-events-none' : ''} text-theme-secondary hover:text-indigo-500`}
                onClick={() => goToDoc(currentIndex - 1)}
              >
                <span className="material-symbols-rounded text-lg group-hover:-translate-x-1 transition-transform">arrow_back</span>
                Previous
              </button>
              <button
                className={`text-sm font-medium flex items-center gap-1 group ${currentIndex >= menuOrder.length - 1 ? 'opacity-40 pointer-events-none' : ''} text-theme-secondary hover:text-indigo-500`}
                onClick={() => goToDoc(currentIndex + 1)}
              >
                Next
                <span className="material-symbols-rounded text-lg group-hover:translate-x-1 transition-transform">arrow_forward</span>
              </button>
            </div>
          </div>
        </main>
      </div>

    </div>
  );
};

export default Documentation;