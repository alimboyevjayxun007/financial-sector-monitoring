import React, { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

interface CanvasSequenceProps {
  frameCount: number;
  texts?: string[];
  getFrameUrl: (index: number) => string;
}

export const CanvasSequence: React.FC<CanvasSequenceProps> = ({ frameCount, texts = [], getFrameUrl }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const pinWrapperRef = useRef<HTMLDivElement>(null);
  const textRefs = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const resizeCanvas = () => {
      if (!canvas || !ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      
      ctx.scale(dpr, dpr);
    };

    resizeCanvas();

    const images: HTMLImageElement[] = [];
    const airpods = { frame: 0 };

    for (let i = 0; i < frameCount; i++) {
      const img = new Image();
      img.src = getFrameUrl(i + 1);
      images.push(img);
    }

    images[0].onload = render;

    function render() {
      if (!canvas || !ctx) return;
      
      const logicalWidth = canvas.width / (window.devicePixelRatio || 1);
      const logicalHeight = canvas.height / (window.devicePixelRatio || 1);
      ctx.clearRect(0, 0, logicalWidth, logicalHeight);
      
      const frameIndex = Math.round(airpods.frame);
      const img = images[frameIndex];
      if (img && img.complete) {
        const scale = Math.max(logicalWidth / img.width, logicalHeight / img.height);
        const x = (logicalWidth / 2) - (img.width / 2) * scale;
        const y = (logicalHeight / 2) - (img.height / 2) * scale;
        ctx.drawImage(img, x, y, img.width * scale, img.height * scale);
      }
    }

    const handleResize = () => {
      resizeCanvas();
      render();
    };
    window.addEventListener('resize', handleResize);

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: pinWrapperRef.current,
        start: 'top top',
        end: 'bottom bottom',
        scrub: 2,
      }
    });

    // Animate the frames over a timeline duration equal to the total frames
    tl.to(airpods, {
      frame: frameCount - 1,
      duration: frameCount - 1,
      snap: 'frame',
      ease: 'none',
      onUpdate: render,
    }, 0);

    // Fade texts in and out if texts exist
    if (texts.length > 0) {
      const segment = (frameCount - 1) / texts.length;
      texts.forEach((_, i) => {
        const textElement = textRefs.current[i];
        if (textElement) {
          // Fade in
          tl.to(textElement, {
            opacity: 1,
            y: 0,
            duration: segment * 0.2, // 20% of segment to fade in
            ease: 'power2.out'
          }, segment * i);
          
          // Hold
          tl.to(textElement, {
            opacity: 1,
            duration: segment * 0.6 // 60% hold
          }, segment * i + segment * 0.2);
          
          // Fade out
          tl.to(textElement, {
            opacity: 0,
            y: -30,
            duration: segment * 0.2, // 20% of segment to fade out
            ease: 'power2.in'
          }, segment * i + segment * 0.8);
        }
      });
    }

    return () => {
      window.removeEventListener('resize', handleResize);
      tl.kill();
      ScrollTrigger.getAll().forEach((t) => t.kill());
    };
  }, [frameCount, getFrameUrl, texts]);

  return (
    <div ref={pinWrapperRef} className="w-full relative" style={{ height: '900vh' }}>
      <div ref={containerRef} className="sticky top-0 w-full h-screen bg-[#030708] overflow-hidden">
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover opacity-80" />
        
        {/* Cinematic Topline */}
        <div className="absolute top-0 left-0 w-full p-6 md:p-10 flex justify-between items-start z-20 pointer-events-none mix-blend-difference text-white/70 text-[10px] md:text-xs font-mono uppercase tracking-[0.2em]">
          <div className="w-8 border-t border-white/30 mt-2"></div>
          <div>AML INTELLIGENCE / CORE ENGINE</div>
        </div>

      {/* Text Sequences */}
      <div className="absolute inset-0 z-10 pointer-events-none">
        {texts.map((text, i) => (
          <div 
            key={i}
            ref={el => { textRefs.current[i] = el; }}
            className="absolute bottom-24 left-6 md:left-16 max-w-3xl transform translate-y-10 opacity-0"
          >
            <p className="text-[10px] md:text-xs font-mono uppercase tracking-[0.3em] text-[#22f396] mb-4 flex items-center gap-3">
              <span className="w-6 border-t border-[#22f396]"></span>
              STEP 0{i + 1} // ANALYSIS
            </p>
            <h2 className="text-3xl md:text-5xl lg:text-6xl font-light tracking-tight text-white leading-[1.1] drop-shadow-2xl">
              {text}
            </h2>
          </div>
        ))}
      </div>

      {/* Cinematic Bottomline */}
      <div className="absolute bottom-0 left-0 w-full p-6 md:p-10 flex justify-between items-end z-20 pointer-events-none mix-blend-difference text-white/50 text-[10px] md:text-xs font-mono uppercase tracking-[0.2em]">
        <div>SIGNAL PROCESSOR</div>
        <div className="flex items-center gap-2 text-[#22f396]">
          SCROLL TO EXPLORE <span className="animate-bounce">↓</span>
        </div>
      </div>
      
      {/* Subtle vignette/gradient overlay for better text readability */}
      <div className="absolute inset-0 z-0 pointer-events-none bg-gradient-to-t from-[#030708]/80 via-transparent to-transparent" />
    </div>
    </div>
  );
};
