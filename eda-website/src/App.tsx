import { useState, useEffect } from 'react';
import { CanvasSequence } from './components/CanvasSequence';
import { HeroSection } from './components/HeroSection';
import { EdaSection } from './components/EdaSection';
import { LanguageContext, translations } from './locales/translations';
import gsap from 'gsap';
import ScrollTrigger from 'gsap/ScrollTrigger';
import './index.css';

gsap.registerPlugin(ScrollTrigger);

function App() {
  const [lang, setLang] = useState('ru');
  const t = translations[lang as keyof typeof translations];

  useEffect(() => {
    // Lenis is handled globally in main.tsx by <ReactLenis>
  }, []);

  return (
    <LanguageContext.Provider value={lang}>
      <div className="fixed top-4 right-4 z-50 flex gap-2">
        {['ru', 'en', 'uz'].map((l) => (
          <button 
            key={l}
            onClick={() => setLang(l)}
            className={"px-3 py-1 text-xs font-mono uppercase rounded border transition-colors " + (lang === l ? "bg-[#22f396]/20 border-[#22f396] text-[#22f396]" : "bg-black/50 border-cyan-900/50 text-slate-400 hover:border-[#22f396]/50")}
          >
            {l}
          </button>
        ))}
      </div>
      <HeroSection />
      {/* --- SCROLL-BOUND VIDEO SEQUENCE --- */}
      <CanvasSequence 
        texts={[t.seq1, t.seq2, t.seq3, t.seq4]}
        frameCount={240} 
        getFrameUrl={(index: number) => `${import.meta.env.BASE_URL}video-frames/frame_${index}.webp`} 
      />
      <EdaSection />
    </LanguageContext.Provider>
  );
}

export default App;
