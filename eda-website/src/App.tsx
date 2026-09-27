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
  const [lang, setLang] = useState('en');
  const t = translations[lang as keyof typeof translations];

  useEffect(() => {
    // Lenis is handled globally in main.tsx by <ReactLenis>
  }, []);

  const languageOptions = [
    { code: 'en', label: 'EN', title: 'English' },
    { code: 'uz', label: 'UZ', title: "O'zbekcha" },
    { code: 'ru', label: 'RU', title: 'Русский' }
  ];

  return (
    <LanguageContext.Provider value={lang}>
      <div className="fixed top-5 right-6 z-50 flex items-center bg-[#07131d]/90 backdrop-blur-md border border-[#22f396]/30 p-1 rounded-full shadow-[0_0_20px_rgba(34,243,150,0.15)]">
        {languageOptions.map((item) => (
          <button 
            key={item.code}
            onClick={() => setLang(item.code)}
            title={item.title}
            className={`px-3.5 py-1 text-xs font-mono uppercase rounded-full transition-all duration-300 font-bold tracking-wider ${
              lang === item.code 
                ? "bg-[#22f396] text-[#030708] shadow-[0_0_12px_rgba(34,243,150,0.6)] scale-105" 
                : "text-slate-400 hover:text-white hover:bg-white/5"
            }`}
          >
            {item.label}
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
