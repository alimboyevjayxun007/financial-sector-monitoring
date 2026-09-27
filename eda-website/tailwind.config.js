/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark: '#030708',
          card: '#111726',
          border: '#1e293b',
          cyan: '#0df2c9',
          neon: '#22f396',
          dim: '#5d7589',
        },
        brandDark: '#080c11',
        cardBg: 'rgba(13, 19, 28, 0.72)',
        calloutBorder: '#4ade80',
        badgeGreen: '#22c55e',
        badgeRed: '#ef4444'
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Courier New', 'monospace'],
        hud: ['Rajdhani', 'sans-serif']
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries')
  ],
}
