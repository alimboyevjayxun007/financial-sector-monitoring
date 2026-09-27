import React, { useContext, useEffect, useRef } from 'react';
import { LanguageContext, translations } from '../locales/translations';

export function HeroSection() {
  const lang = useContext(LanguageContext);
  const t = translations[lang as keyof typeof translations];
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    
    // @ts-ignore
    const THREE = window.THREE;
    if (!THREE) return;

    const width = container.clientWidth || window.innerWidth;
    const height = container.clientHeight || window.innerHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, width / height, 1, 3000);
    camera.position.set(200, 300, 700);
    camera.lookAt(100, 50, 0);

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    
    // Clear old canvases if any
    while (container.firstChild) {
      container.removeChild(container.firstChild);
    }
    container.appendChild(renderer.domElement);

    const cols = 150;
    const rows = 90;
    const count = cols * rows;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const basePositions = new Float32Array(count * 3);

    const colorGreen = new THREE.Color(0x00ff88);
    const colorCyan = new THREE.Color(0x00e5ff);
    const colorDim = new THREE.Color(0x004433);

    let idx = 0;
    const spacingX = 14;
    const spacingZ = 12;
    const offsetX = (cols * spacingX) / 2 - 250;
    const offsetZ = (rows * spacingZ) / 2;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = c * spacingX - offsetX;
        const z = r * spacingZ - offsetZ;
        const y = 0;

        positions[idx * 3] = x;
        positions[idx * 3 + 1] = y;
        positions[idx * 3 + 2] = z;

        basePositions[idx * 3] = x;
        basePositions[idx * 3 + 1] = y;
        basePositions[idx * 3 + 2] = z;

        const t = (c / cols) * 0.7 + (r / rows) * 0.3;
        const pColor = new THREE.Color().lerpColors(colorGreen, colorCyan, Math.sin(t * Math.PI));
        if (r % 4 === 0 || c % 4 === 0) {
          pColor.lerp(colorDim, 0.4);
        }

        colors[idx * 3] = pColor.r;
        colors[idx * 3 + 1] = pColor.g;
        colors[idx * 3 + 2] = pColor.b;

        idx++;
      }
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    if(ctx) {
      const grad = ctx.createRadialGradient(16, 16, 0, 16, 16, 16);
      grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
      grad.addColorStop(0.3, 'rgba(0, 255, 170, 0.9)');
      grad.addColorStop(0.7, 'rgba(0, 200, 255, 0.3)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(16, 16, 16, 0, Math.PI * 2);
      ctx.fill();
    }

    const particleTexture = new THREE.CanvasTexture(canvas);

    const material = new THREE.PointsMaterial({
      size: 7.5,
      vertexColors: true,
      map: particleTexture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    const particleSystem = new THREE.Points(geometry, material);
    scene.add(particleSystem);

    const orbCount = 60;
    const orbGeo = new THREE.BufferGeometry();
    const orbPos = new Float32Array(orbCount * 3);
    const orbCol = new Float32Array(orbCount * 3);
    for (let i = 0; i < orbCount; i++) {
      orbPos[i * 3] = (Math.random() - 0.2) * 1200;
      orbPos[i * 3 + 1] = Math.random() * 350 - 50;
      orbPos[i * 3 + 2] = (Math.random() - 0.5) * 800;
      
      orbCol[i * 3] = 0.0;
      orbCol[i * 3 + 1] = 0.9 + Math.random() * 0.1;
      orbCol[i * 3 + 2] = 0.5 + Math.random() * 0.5;
    }
    orbGeo.setAttribute('position', new THREE.BufferAttribute(orbPos, 3));
    orbGeo.setAttribute('color', new THREE.BufferAttribute(orbCol, 3));
    const orbMat = new THREE.PointsMaterial({
      size: 14,
      vertexColors: true,
      map: particleTexture,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });
    const orbPoints = new THREE.Points(orbGeo, orbMat);
    scene.add(orbPoints);

    const clock = new THREE.Clock();
    let mouseX = 0;
    let mouseY = 0;

    const onMouseMove = (e: MouseEvent) => {
      mouseX = (e.clientX - window.innerWidth / 2) * 0.05;
      mouseY = (e.clientY - window.innerHeight / 2) * 0.05;
    };
    window.addEventListener('mousemove', onMouseMove);

    let animationId: number;
    function animate() {
      animationId = requestAnimationFrame(animate);
      const time = clock.getElapsedTime() * 0.8;
      const posArr = geometry.attributes.position.array as Float32Array;

      let i = 0;
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const baseZ = basePositions[i * 3 + 2];

          const wave1 = Math.sin(c * 0.12 + time * 1.5) * 85;
          const wave2 = Math.cos(r * 0.08 + time * 1.1) * 65;
          const wave3 = Math.sin((c + r) * 0.05 + time * 0.9) * 45;
          const wave4 = Math.sin(c * 0.04 - baseZ * 0.002 + time * 2.0) * 40;

          const rightWeight = Math.max(0, (c - 30) / (cols - 30));
          const elevation = (wave1 + wave2 + wave3 + wave4) * (0.6 + rightWeight * 1.2);

          posArr[i * 3 + 1] = elevation;
          i++;
        }
      }
      geometry.attributes.position.needsUpdate = true;

      camera.position.x += (200 + mouseX - camera.position.x) * 0.02;
      camera.position.y += (300 - mouseY - camera.position.y) * 0.02;
      camera.lookAt(100, 30, 0);

      const oArr = orbGeo.attributes.position.array as Float32Array;
      for (let j = 0; j < orbCount; j++) {
        oArr[j * 3 + 1] += Math.sin(time + j) * 0.3;
      }
      orbGeo.attributes.position.needsUpdate = true;

      renderer.render(scene, camera);
    }

    animate();

    const handleResize = () => {
      const w = container.clientWidth || window.innerWidth;
      const h = container.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', handleResize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', onMouseMove);
      renderer.dispose();
    };
  }, []);

  return (
    <section className="relative w-full min-h-screen flex flex-col justify-between bg-[#030708] overflow-hidden px-8 sm:px-14 md:px-20 lg:px-24 py-8 md:py-12" data-purpose="hero-section">
      <div className="absolute inset-0 w-full h-full pointer-events-none z-0" style={{ display: 'block' }}>
        <div ref={containerRef} style={{ width: '100%', height: '100%' }}></div>
      </div>
      
      <div className="absolute inset-0 bg-gradient-to-r from-[#030708] via-[#030708]/60 to-transparent pointer-events-none z-[1]"></div>
      <div className="scanlines"></div>
      <div className="glow-overlay"></div>
      
      <header className="relative z-10 w-full flex items-start justify-between">
        <div className="flex items-start space-x-6 sm:space-x-8">
          <div className="flex flex-col items-start pt-0.5">
            <span className="text-xs font-mono text-[#8b9ba7] font-semibold tracking-wider leading-none">01</span>
            <span className="text-[9px] font-mono uppercase text-[#475b68] tracking-wider leading-tight mt-1">HERO</span>
            <span className="text-[9px] font-mono uppercase text-[#475b68] tracking-wider leading-none">SECTION</span>
            <div className="w-[1.5px] h-10 bg-[#3a4d5b]/70 mt-2.5 ml-[1px]"></div>
          </div>
          
          <div className="flex items-baseline pt-0.5">
            <a className="text-2xl sm:text-3xl font-extrabold tracking-tight flex items-baseline select-none" href="#">
              <span className="text-white font-semibold">Dnk</span>
              <span className="text-[#22f396] neon-logo-glow font-bold ml-[1px]">Code</span>
            </a>
          </div>
        </div>
        
        <div className="text-right pt-1">
          <span className="font-mono text-xs sm:text-[13px] tracking-widest text-[#4e6473] uppercase">
            TEAM ID: <span className="text-[#647c8c] tracking-normal font-semibold">2ABB3C78</span>
          </span>
        </div>
      </header>
      
      <main className="relative z-10 max-w-3xl my-auto py-12 md:py-16">
        <div className="inline-flex items-center space-x-2 px-3.5 py-1 rounded-full border border-[#22f396]/60 bg-[#22f396]/10 neon-pill-glow mb-8">
          <span className="w-2 h-2 rounded-full bg-[#22f396] neon-dot-pulse"></span>
          <span className="font-mono text-[11px] sm:text-xs tracking-wider text-[#22f396] font-medium uppercase">
            AML Alert Prioritization
          </span>
        </div>
        
        <h1 className="text-4xl sm:text-6xl md:text-7xl font-extrabold tracking-tight leading-[1.08] mb-7">
          <span className="block text-white">{t.heroTitle1}</span>
          <span className="block text-transparent bg-clip-text bg-gradient-to-r from-[#22f396] via-[#0df2c9] to-[#22f396] neon-text-glow mt-1.5">
            {t.heroTitle2}
          </span>
        </h1>
        
        <p className="max-w-2xl text-sm sm:text-base md:text-[17px] text-[#788e9f] leading-relaxed font-normal">{t.heroDesc}</p>
      </main>
      
      
      <footer className="relative z-10 w-full flex justify-center items-center pb-2">
        <a aria-label={t.scrollDown} className="text-[#526b7c] hover:text-[#22f396] transition-colors duration-300 p-2" href="#details">
          <svg className="w-4 h-4 stroke-current transition-transform duration-300 hover:translate-y-0.5" fill="none" strokeWidth="2" viewBox="0 0 24 24">
            <path d="M19.5 13.5L12 21m0 0l-7.5-7.5M12 21V3" strokeLinecap="round" strokeLinejoin="round"></path>
          </svg>
        </a>
      </footer>
    </section>
  );
}
