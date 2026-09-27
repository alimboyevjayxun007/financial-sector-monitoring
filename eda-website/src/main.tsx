import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ReactLenis } from 'lenis/react'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ReactLenis root options={{ lerp: 0.08, smoothWheel: true }}>
      <App />
    </ReactLenis>
  </StrictMode>,
)
