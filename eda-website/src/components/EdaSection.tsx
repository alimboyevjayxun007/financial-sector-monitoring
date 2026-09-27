import React, { useContext, useEffect, useRef } from 'react';
import { LanguageContext, translations } from '../locales/translations';
import { data } from '../data/mockData';

export function EdaSection() {
  const lang = useContext(LanguageContext);
  const t = translations[lang as keyof typeof translations];
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    let width: number, height: number;
    let nodes: any[] = [];

    function resize() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      initNodes();
    }

    function initNodes() {
      nodes = [];
      const nodeCount = Math.floor((width * window.innerHeight) / 22000) + 35;
      for (let i = 0; i < nodeCount; i++) {
        nodes.push({
          x: Math.random() * width,
          y: Math.random() * height,
          vx: (Math.random() - 0.5) * 0.35,
          vy: (Math.random() - 0.5) * 0.35,
          radius: Math.random() * 1.5 + 1
        });
      }
    }

    let animId: number;
    function draw() {
      if(!ctx) return;
      ctx.clearRect(0, 0, width, height);

      ctx.strokeStyle = 'rgba(0, 242, 254, 0.1)';
      ctx.lineWidth = 1.0;
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < 160) {
            const alpha = (1 - dist / 160) * 0.35;
            ctx.strokeStyle = `rgba(0, 242, 254, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.stroke();
          }
        }
      }

      for (let i = 0; i < nodes.length; i++) {
        const n = nodes[i];
        n.x += n.vx;
        n.y += n.vy;

        if (n.x < 0 || n.x > width) n.vx *= -1;
        if (n.y < 0 || n.y > height) n.vy *= -1;

        ctx.fillStyle = 'rgba(34, 243, 150, 0.6)';
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
        ctx.fill();
        
        // Add subtle glow to nodes
        ctx.shadowBlur = 10;
        ctx.shadowColor = 'rgba(34, 243, 150, 0.4)';
        ctx.fill();
        ctx.shadowBlur = 0;
      }

      animId = requestAnimationFrame(draw);
    }

    window.addEventListener('resize', resize);
    setTimeout(resize, 500);
    draw();
    
    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animId);
    }
  }, []);

  return (
    <div id="details" className="relative z-30 min-h-screen font-sans flex flex-col items-center p-4 md:p-8 space-y-16 bg-[#0a0f14]">
      <div className="fixed inset-0 w-full h-full pointer-events-none z-0 overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full opacity-70"></canvas>
      </div>
      
      <div className="absolute -right-20 top-1/4 w-[480px] h-[480px] bg-emerald-500/10 rounded-full blur-[120px] pointer-events-none z-0"></div>
      <div className="absolute left-1/3 top-1/3 w-[360px] h-[360px] bg-cyan-500/10 rounded-full blur-[100px] pointer-events-none z-0"></div>

      <main className="relative z-10 w-full max-w-[1360px] bg-[#050b12] rounded-2xl overflow-hidden border border-cyan-950/40 shadow-2xl flex flex-col p-6 sm:p-10 mb-8 mt-4">
        <h2 className="text-3xl md:text-5xl font-normal tracking-wide text-white drop-shadow-md text-center mb-8">{t.amlTitle}</h2>
        <section className="relative z-10 w-full flex-1 flex flex-wrap items-center justify-around px-2 py-8 my-auto">
          <svg className="absolute inset-0 w-full h-full pointer-events-none z-0 hidden lg:block" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="cyberBeam" x1="0%" x2="100%" y1="0%" y2="0%">
                <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.2"></stop>
                <stop offset="40%" stopColor="#10b981" stopOpacity="0.75"></stop>
                <stop offset="70%" stopColor="#06b6d4" stopOpacity="0.85"></stop>
                <stop offset="100%" stopColor="#4ade80" stopOpacity="0.9"></stop>
              </linearGradient>
            </defs>
            <path d="M 220 150 C 350 150, 450 150, 500 150" fill="none" stroke="url(#cyberBeam)" strokeDasharray="4 3" strokeWidth="1.5"></path>
            <path d="M 650 150 C 750 150, 850 150, 950 150" fill="none" stroke="url(#cyberBeam)" strokeDasharray="3 2" strokeWidth="1.5"></path>
          </svg>

          <div className="relative z-10 flex flex-col items-center flex-1 min-w-[200px] m-4">
            <div className="relative w-40 h-40 sm:w-44 sm:h-44 flex items-center justify-center">
              <svg className="absolute inset-0 w-full h-full animate-spin-reverse" viewBox="0 0 160 160">
                <circle cx="80" cy="80" fill="none" r="72" stroke="#0f293d" strokeWidth="2"></circle>
                <path className="glow-cyan" d="M 80,8 A 72,72 0 1,1 25,126" fill="none" stroke="#06b6d4" strokeLinecap="round" strokeWidth="3"></path>
                <path d="M 80,18 A 62,62 0 0,1 142,80" fill="none" stroke="#34d399" strokeLinecap="round" strokeWidth="3.5"></path>
                <path d="M 35,120 A 62,62 0 0,0 80,142" fill="none" opacity="0.8" stroke="#22d3ee" strokeLinecap="round" strokeWidth="2"></path>
              </svg>
              <div className="w-32 h-32 rounded-full border border-cyan-500/30 bg-[#07131d]/60 backdrop-blur-sm flex items-center justify-center shadow-[inset_0_0_15px_rgba(6,182,212,0.15)]">
                <span className="font-hud text-3xl sm:text-4xl font-semibold text-white tracking-wider glow-text">{data.alerts}</span>
              </div>
            </div>
            <p className="mt-4 text-[13px] sm:text-sm font-light text-slate-300/80 tracking-wide text-center">{t.trainingAlerts}</p>
          </div>

          <div className="relative z-10 flex flex-col items-center flex-1 min-w-[200px] m-4">
            <div className="relative w-44 h-44 sm:w-48 sm:h-48 flex items-center justify-center">
              <svg className="absolute inset-0 w-full h-full animate-spin-slow" viewBox="0 0 180 180">
                <circle cx="90" cy="90" fill="none" opacity="0.6" r="82" stroke="#0ea5e9" strokeDasharray="3 5 8 5" strokeWidth="1.8"></circle>
                <circle className="glow-cyan" cx="90" cy="90" fill="none" opacity="0.9" r="74" stroke="#10b981" strokeDasharray="1 3" strokeLinecap="round" strokeWidth="4.5"></circle>
                <path d="M 90,16 A 74,74 0 0,1 164,90" fill="none" stroke="#34d399" strokeLinecap="round" strokeWidth="4"></path>
                <path d="M 40,140 A 74,74 0 0,0 90,164" fill="none" stroke="#06b6d4" strokeWidth="2.5"></path>
              </svg>
              <svg className="absolute inset-0 w-full h-full animate-spin-reverse opacity-70" viewBox="0 0 180 180">
                <circle cx="90" cy="90" fill="none" r="63" stroke="#22d3ee" strokeDasharray="16 10 4 10" strokeWidth="1.5"></circle>
                <circle cx="90" cy="90" fill="none" r="54" stroke="#065f46" strokeWidth="1"></circle>
              </svg>
              <div className="w-28 h-28 sm:w-32 sm:h-32 rounded-full border border-teal-400/40 bg-[#071720]/75 backdrop-blur-md flex items-center justify-center shadow-[inset_0_0_20px_rgba(20,184,166,0.2)]">
                <span className="font-hud text-3xl sm:text-4xl font-semibold text-white tracking-wider glow-text">{data.transactions}</span>
              </div>
            </div>
            <p className="mt-4 text-[13px] sm:text-sm font-light text-slate-300/80 tracking-wide text-center">{t.transactions}</p>
          </div>

          <div className="relative z-10 flex flex-col items-center flex-1 min-w-[200px] m-4">
            <div className="relative w-40 h-40 sm:w-44 sm:h-44 flex items-center justify-center">
              <svg className="absolute inset-0 w-full h-full animate-spin-slow" viewBox="0 0 160 160">
                <circle cx="80" cy="80" fill="none" opacity="0.6" r="70" stroke="#064e3b" strokeWidth="1.5"></circle>
                <path className="glow-cyan" d="M 80,10 A 70,70 0 0,1 150,80" fill="none" stroke="#4ade80" strokeLinecap="round" strokeWidth="3"></path>
                <path d="M 80,150 A 70,70 0 0,1 10,80" fill="none" stroke="#06b6d4" strokeLinecap="round" strokeWidth="2.5"></path>
              </svg>
              <svg className="absolute inset-0 w-full h-full animate-spin-reverse opacity-80" viewBox="0 0 160 160">
                <circle cx="80" cy="80" fill="none" r="58" stroke="#10b981" strokeDasharray="10 8" strokeWidth="1.5"></circle>
              </svg>
              <div className="w-28 h-28 sm:w-30 sm:h-30 rounded-full border border-emerald-500/30 bg-[#06141a]/70 backdrop-blur-sm flex items-center justify-center shadow-[inset_0_0_15px_rgba(16,185,129,0.2)]">
                <span className="font-hud text-3xl sm:text-4xl font-semibold text-white tracking-wider glow-text">{data.columns}</span>
              </div>
            </div>
            <p className="mt-4 text-[13px] sm:text-sm font-light text-slate-300/80 tracking-wide text-center">{t.selectedFeatures}</p>
          </div>

          <div className="relative z-10 flex flex-col items-center flex-[1.4] min-w-[280px] m-4">
            <div className="relative w-64 h-64 sm:w-72 sm:h-72 flex items-center justify-center animate-pulse-glow">
              <svg className="absolute inset-0 w-full h-full animate-spin-slow pointer-events-none" viewBox="0 0 280 280">
                <circle className="glow-cyan-lg" cx="140" cy="140" fill="none" r="130" stroke="url(#vortexGrad)" strokeDasharray="2 6 12 5 28 8" strokeLinecap="round" strokeWidth="4.5"></circle>
                <circle cx="140" cy="140" fill="none" opacity="0.6" r="122" stroke="#38bdf8" strokeDasharray="1 4" strokeWidth="1.5"></circle>
              </svg>
              <svg className="absolute inset-0 w-full h-full animate-spin-reverse pointer-events-none" viewBox="0 0 280 280">
                <defs>
                  <linearGradient id="vortexGrad" x1="0%" x2="100%" y1="0%" y2="100%">
                    <stop offset="0%" stopColor="#4ade80"></stop>
                    <stop offset="45%" stopColor="#22d3ee"></stop>
                    <stop offset="85%" stopColor="#3b82f6"></stop>
                    <stop offset="100%" stopColor="#a855f7"></stop>
                  </linearGradient>
                </defs>
                <path className="glow-cyan" d="M 140,25 A 115,115 0 0,1 255,140" fill="none" stroke="#4ade80" strokeLinecap="round" strokeWidth="4"></path>
                <path d="M 255,140 A 115,115 0 0,1 140,255" fill="none" stroke="#22d3ee" strokeDasharray="8 6" strokeWidth="3"></path>
                <circle cx="140" cy="140" fill="none" opacity="0.8" r="106" stroke="#2dd4bf" strokeDasharray="30 14 10 14" strokeWidth="2"></circle>
              </svg>
              <svg className="absolute inset-0 w-full h-full animate-spin-fast pointer-events-none" viewBox="0 0 280 280">
                <circle cx="140" cy="140" fill="none" opacity="0.75" r="92" stroke="#67e8f9" strokeDasharray="5 15" strokeWidth="2"></circle>
              </svg>
              <div className="w-48 h-48 sm:w-52 sm:h-52 rounded-full border-2 border-emerald-400/60 bg-gradient-to-br from-[#06242c]/90 via-[#03151f]/90 to-[#020b12]/95 backdrop-blur-md flex flex-col items-center justify-center shadow-[inset_0_0_35px_rgba(52,211,153,0.35),0_0_25px_rgba(34,211,238,0.25)]">
                <span className="font-hud text-5xl sm:text-6xl font-bold tracking-tight text-[#6ee7b7] glow-text">{data.auc}</span>
              </div>
            </div>
            <p className="mt-2 text-sm sm:text-base font-medium text-slate-300 tracking-wider text-center">{t.cvRocAuc}</p>
          </div>
        </section>
        <div className="relative z-10 w-full flex items-center justify-between text-[11px] text-slate-600 font-mono tracking-widest pt-2">
          <span className="opacity-40">SYSTEM // TELEMETRY LINK ACTIVE</span>
          <span className="opacity-40">QUANT_METRICS_V4.2</span>
        </div>
      </main>

      {/* --- ALL 5 ARTICLES FROM YOUR ORIGINAL DESIGN --- */}
      <section className="relative z-10 w-full max-w-[1400px] flex flex-col items-center px-4">
        <header className="w-full text-center mb-10 md:mb-16">
          <h2 className="text-3xl md:text-5xl font-light tracking-wide text-transparent bg-clip-text bg-gradient-to-r from-white via-[#6ee7b7] to-[#00f2fe] drop-shadow-[0_0_15px_rgba(34,243,150,0.4)]">
            {t.edaTitle}
          </h2>
        </header>
        <div className="w-full grid grid-cols-1 lg:grid-cols-2 gap-8 md:gap-10">
          
          {/* ==========================================
                 TOP-LEFT SECTION: Time Series Activity
            =========================================== */}
          <article className="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch" data-purpose="chart-card-group">
            <div className="md:col-span-7 flex flex-col justify-end pt-4 pr-2">
              <div className="relative w-full aspect-[16/10] sm:aspect-[16/9]">
                <svg aria-label={t.timeActivityChart} className="w-full h-full overflow-visible drop-shadow-xl" viewBox="0 0 460 260">
                  <defs>
                    <linearGradient id="neonGradient1" x1="0%" x2="100%" y1="0%" y2="0%">
                      <stop offset="0%" stopColor="#00f2fe"></stop>
                      <stop offset="65%" stopColor="#00f2fe"></stop>
                      <stop offset="85%" stopColor="#a3e635"></stop>
                      <stop offset="100%" stopColor="#4ade80"></stop>
                    </linearGradient>
                  </defs>
                  <g stroke="#232d42" strokeDasharray="0" strokeWidth="1">
                    <line x1="45" x2="440" y1="20" y2="20"></line>
                    <line x1="45" x2="440" y1="65" y2="65"></line>
                    <line x1="45" x2="440" y1="110" y2="110"></line>
                    <line x1="45" x2="440" y1="155" y2="155"></line>
                    <line x1="45" x2="440" y1="200" y2="200"></line>
                  </g>
                  <g fill="#94a3b8" fontFamily="sans-serif" fontSize="10">
                    <text textAnchor="end" x="40" y="24">1000</text>
                    <text textAnchor="end" x="40" y="69">800</text>
                    <text textAnchor="end" x="40" y="114">600</text>
                    <text textAnchor="end" x="40" y="159">400</text>
                    <text textAnchor="end" x="40" y="204">200</text>
                    <text textAnchor="end" x="40" y="222">0</text>
                    <text fill="#64748b" fontSize="9" textAnchor="middle" x="45" y="235">{t.day30}</text>
                    <text textAnchor="middle" x="135" y="235">{t.day20}</text>
                    <text textAnchor="middle" x="230" y="235">{t.day10}</text>
                    <text textAnchor="middle" x="325" y="235">{t.day5}</text>
                    <text textAnchor="middle" x="410" y="235">{t.alertDate}</text>
                    <text fill="#cbd5e1" fontSize="11" textAnchor="middle" x="235" y="255">{t.daysToAlert}</text>
                    <text fill="#cbd5e1" fontSize="11" textAnchor="middle" transform="rotate(-90)" x="-110" y="12">{t.transactions}</text>
                  </g>
                  <path className="glow-cyan" d="M 45,215 L 50,210 L 53,195 L 55,145 L 57,190 L 63,212 L 72,210 L 80,205 
                                         L 88,212 L 95,198 L 102,185 L 110,195 L 118,170 L 126,178 L 132,165 
                                         L 138,185 L 145,150 L 150,170 L 157,142 L 165,180 L 172,130 L 178,162 
                                         L 186,145 L 194,175 L 202,135 L 210,165 L 216,140 L 225,180 L 235,115 
                                         L 242,165 L 250,135 L 258,185 L 265,130 L 273,150 L 280,105 L 288,140 
                                         L 295,95 L 302,170 L 310,120 L 318,160 L 325,82 L 332,150 L 340,90 
                                         L 348,155 L 355,80 L 362,175 L 370,110 L 378,60 L 385,160 L 392,80 
                                         L 400,30 L 406,120 L 413,85 L 420,130 L 426,170" fill="none" stroke="url(#neonGradient1)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.6"></path>
                </svg>
              </div>
            </div>
            <div className="md:col-span-5 p-4 sm:p-6 flex flex-col justify-center">
              <h2 className="text-white text-lg font-medium mb-4">{t.timeActivity}</h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">{t.timeActivityDesc1}</p>
              <p className="text-slate-400 text-sm leading-relaxed">{t.timeActivityDesc2}</p>
            </div>
          </article>

          {/* ==========================================
                 TOP-RIGHT SECTION: Directions & Transaction Types
            =========================================== */}
          <article className="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch" data-purpose="chart-card-group">
            <div className="md:col-span-7 flex flex-col justify-end pt-4 pr-2">
              <div className="relative w-full aspect-[16/10] sm:aspect-[16/9]">
                <svg aria-label={t.transactionHistogram} className="w-full h-full overflow-visible drop-shadow-xl" viewBox="0 0 460 260">
                  <g stroke="#232d42" strokeWidth="1">
                    <line x1="45" x2="440" y1="20" y2="20"></line>
                    <line x1="45" x2="440" y1="56" y2="56"></line>
                    <line x1="45" x2="440" y1="92" y2="92"></line>
                    <line x1="45" x2="440" y1="128" y2="128"></line>
                    <line x1="45" x2="440" y1="164" y2="164"></line>
                    <line x1="45" x2="440" y1="200" y2="200"></line>
                  </g>
                  <g fill="#94a3b8" fontFamily="sans-serif" fontSize="10">
                    <text textAnchor="end" x="40" y="24">60M</text>
                    <text textAnchor="end" x="40" y="60">50M</text>
                    <text textAnchor="end" x="40" y="96">40M</text>
                    <text textAnchor="end" x="40" y="132">30M</text>
                    <text textAnchor="end" x="40" y="168">20M</text>
                    <text textAnchor="end" x="40" y="204">10M</text>
                    <text textAnchor="end" x="40" y="218">0</text>
                    <text fill="#cbd5e1" fontSize="11" textAnchor="middle" transform="rotate(-90)" x="-110" y="10">{t.transactions}</text>
                    <text fill="#cbd5e1" fontSize="11" textAnchor="middle" x="250" y="255">{t.transactions}</text>
                  </g>
                  <rect className="glow-cyan" fill="#00f2fe" fillOpacity="0.08" height="196" rx="3" stroke="#00f2fe" strokeWidth="2.5" width="28" x="58" y="20"></rect>
                  <rect className="glow-green" fill="#4ade80" fillOpacity="0.08" height="131" rx="3" stroke="#4ade80" strokeWidth="2.5" width="28" x="105" y="85"></rect>
                  <rect className="glow-cyan" fill="#00f2fe" fillOpacity="0.08" height="111" rx="3" stroke="#00f2fe" strokeWidth="2.5" width="28" x="152" y="105"></rect>
                  <rect className="glow-lime" fill="#a3e635" fillOpacity="0.08" height="71" rx="3" stroke="#a3e635" strokeWidth="2.5" width="28" x="199" y="145"></rect>
                  <rect className="glow-cyan" fill="#00f2fe" fillOpacity="0.08" height="74" rx="3" stroke="#00f2fe" strokeWidth="2.5" width="28" x="246" y="142"></rect>
                  <rect className="glow-cyan" fill="#00f2fe" fillOpacity="0.08" height="64" rx="3" stroke="#00f2fe" strokeWidth="2.5" width="28" x="293" y="152"></rect>
                  <rect className="glow-lime" fill="#a3e635" fillOpacity="0.08" height="38" rx="3" stroke="#a3e635" strokeWidth="2.5" width="28" x="340" y="178"></rect>
                  <rect className="glow-lime" fill="#a3e635" fillOpacity="0.08" height="32" rx="3" stroke="#a3e635" strokeWidth="2.5" width="28" x="387" y="184"></rect>
                  <g fill="#94a3b8" fontFamily="sans-serif" fontSize="9">
                    <text transform="rotate(30, 70, 226)" x="70" y="226">{t.cat1}</text>
                    <text transform="rotate(30, 117, 226)" x="117" y="226">{t.cat2}</text>
                    <text transform="rotate(30, 164, 226)" x="164" y="226">{t.cat3}</text>
                    <text transform="rotate(30, 211, 226)" x="211" y="226">{t.cat4}</text>
                    <text transform="rotate(30, 258, 226)" x="258" y="226">{t.cat5}</text>
                    <text transform="rotate(30, 305, 226)" x="305" y="226">{t.cat6}</text>
                    <text transform="rotate(30, 352, 226)" x="352" y="226">{t.cat7}</text>
                    <text transform="rotate(30, 399, 226)" x="399" y="226">{t.cat8}</text>
                  </g>
                </svg>
              </div>
            </div>
            <div className="md:col-span-5 p-4 sm:p-6 flex flex-col justify-center">
              <h2 className="text-white text-lg font-medium mb-4">{t.transactionTypes}</h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">{t.transactionTypesDesc1}</p>
              <p className="text-slate-400 text-sm leading-relaxed">{t.transactionTypesDesc2}</p>
            </div>
          </article>

          {/* ==========================================
                 BOTTOM-LEFT SECTION: Surge Before Alert
            =========================================== */}
          <article className="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch" data-purpose="chart-card-group">
            <div className="md:col-span-5 order-2 md:order-1 p-4 sm:p-6 flex flex-col justify-center">
              <h2 className="text-white text-lg font-medium mb-4 leading-snug">{t.patternBurst}</h2>
              <p className="text-slate-400 text-sm leading-relaxed">{t.patternBurstDesc}</p>
            </div>
            <div className="md:col-span-7 order-1 md:order-2 flex flex-col justify-end pt-4 pr-2">
              <div className="relative w-full aspect-[16/10] sm:aspect-[16/9]">
                <svg aria-label={t.burstActivityChart} className="w-full h-full overflow-visible drop-shadow-xl" viewBox="0 0 460 260">
                  <defs>
                    <linearGradient id="neonGradient2" x1="0%" x2="100%" y1="0%" y2="0%">
                      <stop offset="0%" stopColor="#4ade80"></stop>
                      <stop offset="60%" stopColor="#00f2fe"></stop>
                      <stop offset="78%" stopColor="#00f2fe"></stop>
                      <stop offset="95%" stopColor="#4ade80"></stop>
                    </linearGradient>
                  </defs>
                  <g stroke="#232d42" strokeWidth="1">
                    <line x1="45" x2="440" y1="40" y2="40"></line>
                    <line x1="45" x2="440" y1="85" y2="85"></line>
                    <line x1="45" x2="440" y1="130" y2="130"></line>
                    <line x1="45" x2="440" y1="175" y2="175"></line>
                    <line x1="45" x2="440" y1="215" y2="215"></line>
                  </g>
                  <g fill="#94a3b8" fontFamily="sans-serif" fontSize="10">
                    <text textAnchor="end" x="40" y="44">200</text>
                    <text textAnchor="end" x="40" y="89">150</text>
                    <text textAnchor="end" x="40" y="134">100</text>
                    <text textAnchor="end" x="40" y="179">50</text>
                    <text textAnchor="end" x="40" y="218">0</text>
                    <text textAnchor="middle" x="135" y="235">2010</text>
                    <text textAnchor="middle" x="210" y="235">2015</text>
                    <text textAnchor="middle" x="280" y="235">2020</text>
                    <text textAnchor="middle" x="345" y="235">2020</text>
                    <text textAnchor="middle" x="410" y="235">2020</text>
                    <text fill="#cbd5e1" fontSize="11" textAnchor="middle" x="270" y="255">{t.date}</text>
                    <text fill="#cbd5e1" fontSize="11" textAnchor="middle" transform="rotate(-90)" x="-125" y="12">{t.activityLevel}</text>
                  </g>
                  <path className="glow-cyan" d="M 52,205 C 55,200 60,195 65,198 C 70,202 75,212 85,212 
                                         C 95,212 100,202 110,203 C 120,204 125,208 135,208 
                                         C 145,208 152,196 160,198 C 170,200 178,212 188,210 
                                         C 198,208 205,196 215,197 C 225,198 230,207 240,205 
                                         C 248,203 252,185 260,186 C 268,187 274,204 282,200 
                                         C 288,197 292,175 298,170 C 304,165 308,182 314,180 
                                         C 322,176 325,145 330,140 C 335,135 340,165 344,160 
                                         C 348,155 352,125 356,85 C 360,40 366,35 370,85 
                                         C 374,130 378,160 382,125 C 386,95 390,140 395,150 
                                         C 400,160 404,180 408,165 C 412,145 418,105 422,125" fill="none" stroke="url(#neonGradient2)" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.6"></path>
                </svg>
              </div>
            </div>
          </article>

          {/* ==========================================
                 BOTTOM-RIGHT SECTION: Breakdown by Outcome
            =========================================== */}
          <article className="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch" data-purpose="chart-card-group">
            <div className="md:col-span-7 flex flex-col justify-end pt-4 pr-2">
              <div className="relative w-full aspect-[16/10] sm:aspect-[16/9]">
                <svg aria-label={t.outcomeChart} className="w-full h-full overflow-visible drop-shadow-xl" viewBox="0 0 460 260">
                  <g stroke="#232d42" strokeWidth="1">
                    <line x1="45" x2="380" y1="20" y2="20"></line>
                    <line x1="45" x2="380" y1="56" y2="56"></line>
                    <line x1="45" x2="380" y1="92" y2="92"></line>
                    <line x1="45" x2="380" y1="128" y2="128"></line>
                    <line x1="45" x2="380" y1="164" y2="164"></line>
                    <line x1="45" x2="380" y1="200" y2="200"></line>
                  </g>
                  <g fill="#94a3b8" fontFamily="sans-serif" fontSize="10">
                    <text textAnchor="end" x="40" y="24">1</text>
                    <text textAnchor="end" x="40" y="60">0.8</text>
                    <text textAnchor="end" x="40" y="96">0.6</text>
                    <text textAnchor="end" x="40" y="132">0.4</text>
                    <text textAnchor="end" x="40" y="168">0.2</text>
                    <text textAnchor="end" x="40" y="204">0</text>
                    <text fill="#cbd5e1" fontSize="11" textAnchor="middle" transform="rotate(-90)" x="-110" y="10">{t.transactionTypes}</text>
                  </g>
                  
                  <g className="glow-cyan">
                    <rect fill="#a3e635" fillOpacity="0.1" height="6" rx="2" stroke="#a3e635" strokeWidth="2" width="28" x="58" y="24"></rect>
                    <rect fill="#4ade80" fillOpacity="0.1" height="8" rx="2" stroke="#4ade80" strokeWidth="2" width="28" x="58" y="32"></rect>
                    <rect fill="#00f2fe" fillOpacity="0.1" height="158" rx="2" stroke="#00f2fe" strokeWidth="2.5" width="28" x="58" y="42"></rect>
                  </g>
                  <g className="glow-cyan">
                    <rect fill="#4ade80" fillOpacity="0.1" height="14" rx="2" stroke="#4ade80" strokeWidth="2" width="28" x="122" y="24"></rect>
                    <rect fill="#131e33" height="52" rx="2" stroke="#22d3ee" strokeWidth="1.8" width="28" x="122" y="40"></rect>
                    <rect fill="#00f2fe" fillOpacity="0.1" height="106" rx="2" stroke="#00f2fe" strokeWidth="2.5" width="28" x="122" y="94"></rect>
                  </g>
                  <g className="glow-cyan">
                    <rect fill="#a3e635" fillOpacity="0.1" height="14" rx="2" stroke="#a3e635" strokeWidth="2" width="28" x="186" y="24"></rect>
                    <rect fill="#00f2fe" fillOpacity="0.1" height="160" rx="2" stroke="#00f2fe" strokeWidth="2.5" width="28" x="186" y="40"></rect>
                  </g>
                  <g className="glow-cyan">
                    <rect fill="#4ade80" fillOpacity="0.1" height="18" rx="2" stroke="#4ade80" strokeWidth="2" width="28" x="250" y="24"></rect>
                    <rect fill="#131e33" height="30" rx="2" stroke="#22d3ee" strokeWidth="1.8" width="28" x="250" y="44"></rect>
                    <rect fill="#00f2fe" fillOpacity="0.1" height="124" rx="2" stroke="#00f2fe" strokeWidth="2.5" width="28" x="250" y="76"></rect>
                  </g>
                  <g className="glow-cyan">
                    <rect fill="#a3e635" fillOpacity="0.1" height="66" rx="2" stroke="#a3e635" strokeWidth="2.5" width="28" x="314" y="24"></rect>
                    <rect fill="#131e33" height="50" rx="2" stroke="#22d3ee" strokeWidth="2" width="28" x="314" y="92"></rect>
                    <rect fill="#00f2fe" fillOpacity="0.1" height="56" rx="2" stroke="#00f2fe" strokeWidth="2.5" width="28" x="314" y="144"></rect>
                  </g>

                  <g fill="#94a3b8" fontFamily="sans-serif" fontSize="9">
                    <rect fill="#00f2fe" height="7" width="7" x="390" y="32"></rect>
                    <text x="403" y="39">{t.total}</text>
                    <rect fill="#22c55e" height="7" width="7" x="390" y="47"></rect>
                    <text x="403" y="54">{t.escalated}</text>
                    <rect fill="#a3e635" height="7" width="7" x="390" y="62"></rect>
                    <text x="403" y="69">{t.dismissed}</text>
                  </g>
                  <g fill="#94a3b8" fontFamily="sans-serif" fontSize="9">
                    <text transform="rotate(30, 65, 215)" x="65" y="215">{t.kirim}</text>
                    <text transform="rotate(30, 127, 215)" x="127" y="215">{t.chiqim}</text>
                    <text transform="rotate(30, 190, 215)" x="190" y="215">{t.karta}</text>
                    <text transform="rotate(30, 252, 215)" x="252" y="215">{t.bankOtk}</text>
                    <text transform="rotate(30, 306, 215)" x="306" y="215">{t.naqdXalq}</text>
                  </g>
                </svg>
              </div>
            </div>
            <div className="md:col-span-5 p-4 sm:p-6 flex flex-col justify-center">
              <h2 className="text-white text-lg font-medium mb-4 leading-snug">{t.typesByOutcome}</h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">{t.typesByOutcomeDesc1}</p>
              <p className="text-slate-400 text-sm leading-relaxed">{t.typesByOutcomeDesc2}</p>
            </div>
          </article>
        </div>

        {/* --- TARGET DISTRIBUTION (Required by Hackathon TZ) --- */}
        <div className="w-full mt-8 md:mt-10">
          <article className="grid grid-cols-1 md:grid-cols-12 gap-5 items-stretch lg:col-span-2" data-purpose="chart-card-group">
            <div className="md:col-span-6 flex flex-col justify-end">
              <div className="relative w-full aspect-[2/1] sm:aspect-[16/9] rounded-xl border border-dashed border-cyan-500/20 bg-cyan-900/10 flex items-center justify-center">
                 <span className="text-cyan-500/40 font-mono text-sm">{t.targetPlaceholder}</span>
              </div>
            </div>
            <div className="md:col-span-6 p-4 sm:p-6 flex flex-col justify-center">
              <h2 className="text-white text-lg font-medium mb-4 leading-snug">{t.targetDistTitle}</h2>
              <p className="text-slate-400 text-sm leading-relaxed mb-4">
                {t.targetDistDesc.replace("{rate}", data.rate)}
              </p>
            </div>
          </article>
        </div>
      </section>

      {/* --- ABLATION TABLE SECTION --- */}
      <section 
        className="relative z-10 w-full max-w-[1024px] min-h-[559px] bg-[#070b10]/90 backdrop-blur-md rounded-xl shadow-[0_30px_60px_-15px_rgba(0,0,0,0.8),0_0_40px_rgba(34,243,150,0.1)] overflow-hidden flex flex-col justify-between p-7 border border-[#22f396]/20 mt-12 mb-20 transition-transform duration-700 hover:scale-[1.02]"
        style={{ transform: 'perspective(1500px) rotateX(4deg) rotateY(-2deg)' }}
      >
        <div aria-hidden="true" className="prism-background"></div>
        <div aria-hidden="true" className="prism-facets"></div>

        <header className="relative z-10 flex flex-col items-center w-full">
          <h2 className="text-[32px] sm:text-[42px] font-light tracking-wide text-transparent bg-clip-text bg-gradient-to-r from-[#22f396] via-white to-cyan-400 drop-shadow-[0_0_12px_rgba(34,243,150,0.3)] text-center mb-6">{t.ablationTitle}</h2>
          <aside className="w-full max-w-[850px] bg-[#0e161c]/80 backdrop-blur-md border border-[#4ade80]/90 rounded-xl px-5 py-3 shadow-[0_0_15px_rgba(74,222,128,0.15)] flex items-start gap-3.5">
            <span aria-hidden="true" className="text-[#4ade80] text-2xl font-serif font-black leading-none mt-0.5 select-none">“</span>
            <p className="text-[13.5px] leading-[1.4] text-slate-200 font-normal">
              {t.ablationDesc.replace('{rejected}', data.rejected_families.toString()).replace('{chance}', data.chance_families.toString()).replace('{drift}', data.drift_verdict).replace('{adv}', data.adversarial.toString())}
            </p>
          </aside>
        </header>

        <div className="relative z-10 w-full flex justify-center pb-2 mt-8">
          <div className="w-full glass-card rounded-2xl p-4 sm:p-6">
            <table className="w-full text-left text-sm border-collapse">
              <thead>
                <tr className="border-b border-[#22f396]/20 text-[#6ee7b7] font-mono tracking-widest uppercase text-[11px]">
                  <th className="pb-3 pl-4 font-semibold w-[44%]" scope="col">{t.familyAdded}</th>
                  <th className="pb-3 text-center font-semibold w-[16%]" scope="col">{t.columns}</th>
                  <th className="pb-3 text-center font-semibold w-[20%]" scope="col">
                    <span className="inline-flex items-center gap-2"><span className="text-[#22f396]/30 text-xs">|</span> CV mean <span className="text-[#22f396]/30 text-xs">|</span></span>
                  </th>
                  <th className="pb-3 pr-4 text-center font-semibold w-[20%]" scope="col">{t.decision}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#22f396]/10 text-[14px] font-normal tracking-wide text-slate-200">
                {data.selection_history.map((row, idx) => (
                  <tr key={idx} className="hover:bg-[#22f396]/5 transition-colors group">
                    <td className="py-2.5 pl-4 text-slate-100 group-hover:text-[#22f396] transition-colors">{row.family}</td>
                    <td className="py-2.5 text-center font-mono text-slate-300">{row.n_columns}</td>
                    <td className="py-2.5 text-center font-mono text-[#00f2fe]">{row.mean.toFixed(4)}</td>
                    <td className="py-2.5 pr-4 text-center">
                      <span className={`inline-block min-w-[70px] text-center px-3 py-1 rounded-full text-[11px] font-bold tracking-widest uppercase ${
                        row.accepted 
                          ? "bg-[#22f396]/20 text-[#22f396] shadow-[0_0_15px_rgba(34,243,150,0.3)] border border-[#22f396]/50" 
                          : "bg-red-500/10 text-red-400 shadow-[0_0_12px_rgba(239,68,68,0.2)] border border-red-500/30"
                      }`}>
                        {row.accepted ? t.kept : t.rejected}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* --- CONCLUSION (Required by Hackathon TZ) --- */}
      <section className="relative z-10 w-full max-w-[1024px] mb-16">
        <div className="w-full glass-card rounded-2xl p-6 sm:p-8 border border-emerald-500/20 shadow-[0_0_30px_rgba(16,185,129,0.1)] hover:-translate-y-1 transition-transform duration-500 group">
          <h2 className="text-3xl text-transparent bg-clip-text bg-gradient-to-r from-emerald-300 to-cyan-300 font-medium mb-6 flex items-center gap-3">
            <span className="w-2 h-2 rounded-full bg-[#22f396] neon-dot-pulse group-hover:scale-150 transition-transform duration-500"></span>{t.conclusionTitle}
          </h2>
          <div className="space-y-4 text-slate-300 text-[15px] leading-relaxed">
            <p>
              {t.conclusion1} 
            </p>
            <p>{t.conclusion2}</p>
          </div>
        </div>
      </section>

      {/* --- FOOTER / TEAM SECTION --- */}
      <footer className="relative z-10 w-full max-w-[1024px] mb-8 mt-auto flex flex-col items-center justify-center border-t border-cyan-900/30 pt-8 pb-4">
        <div className="flex items-center gap-4 mb-3">
          <div className="w-10 h-10 rounded-full border border-cyan-500/30 bg-cyan-950/40 flex items-center justify-center neon-logo-glow">
            <span className="font-mono text-cyan-400 font-bold">D</span>
          </div>
          <div>
            <h3 className="text-white font-medium text-lg tracking-wide">{t.teamInfo}</h3>
            <p className="text-cyan-500/70 text-xs font-mono tracking-widest uppercase">ID: 2ABB3C78</p>
          </div>
        </div>
        <p className="text-slate-600 text-xs mt-2 font-mono">
          {t.footerEvent}
        </p>
      </footer>

    </div>
  );
}
