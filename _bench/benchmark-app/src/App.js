import { useEffect, useState } from 'react';

// Custom SVG Icon Components
const TensorCubeIcon = () => (
  <svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Isometric cube representing tensor */}
    <g className="animate-[fadeInUp_0.6s_ease-out_forwards] opacity-0 stagger-1">
      {/* Top face */}
      <path d="M50 20 L75 32 L50 44 L25 32 Z" fill="url(#gradBlue)" stroke="#00F5FF" strokeWidth="1.5"/>
      {/* Left face */}
      <path d="M25 32 L25 62 L50 74 L50 44 Z" fill="url(#gradBlueDark)" stroke="#00B4D8" strokeWidth="1.5"/>
      {/* Right face */}
      <path d="M50 44 L50 74 L75 62 L75 32 Z" fill="url(#gradBlueMid)" stroke="#00D4FF" strokeWidth="1.5"/>
      {/* Grid lines */}
      <line x1="50" y1="32" x2="50" y2="44" stroke="#00F5FF" strokeWidth="0.5" opacity="0.5"/>
      <line x1="37.5" y1="38" x2="37.5" y2="68" stroke="#00B4D8" strokeWidth="0.5" opacity="0.5"/>
      <line x1="62.5" y1="38" x2="62.5" y2="68" stroke="#00D4FF" strokeWidth="0.5" opacity="0.5"/>
    </g>
    <defs>
      <linearGradient id="gradBlue" x1="50" y1="20" x2="50" y2="44" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#00F5FF"/>
        <stop offset="100%" stopColor="#00B4D8"/>
      </linearGradient>
      <linearGradient id="gradBlueDark" x1="25" y1="32" x2="25" y2="62" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#0A5F7F"/>
        <stop offset="100%" stopColor="#062838"/>
      </linearGradient>
      <linearGradient id="gradBlueMid" x1="75" y1="32" x2="75" y2="62" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#00B4D8"/>
        <stop offset="100%" stopColor="#006B8A"/>
      </linearGradient>
    </defs>
  </svg>
);

const NeuralNetIcon = () => (
  <svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g className="animate-[fadeInUp_0.6s_ease-out_forwards] opacity-0 stagger-2">
      {/* Neural network nodes */}
      <circle cx="30" cy="30" r="6" fill="#9D4EDD" className="animate-[pulse-glow_2s_ease-in-out_infinite]"/>
      <circle cx="30" cy="50" r="6" fill="#9D4EDD" className="animate-[pulse-glow_2s_ease-in-out_infinite_0.3s]"/>
      <circle cx="30" cy="70" r="6" fill="#9D4EDD" className="animate-[pulse-glow_2s_ease-in-out_infinite_0.6s]"/>

      <circle cx="70" cy="30" r="6" fill="#06FFA5" className="animate-[pulse-glow_2s_ease-in-out_infinite_0.4s]"/>
      <circle cx="70" cy="50" r="6" fill="#06FFA5" className="animate-[pulse-glow_2s_ease-in-out_infinite_0.7s]"/>
      <circle cx="70" cy="70" r="6" fill="#06FFA5" className="animate-[pulse-glow_2s_ease-in-out_infinite_1s]"/>

      {/* Connections */}
      <line x1="30" y1="30" x2="70" y2="30" stroke="url(#gradPurpleGreen)" strokeWidth="2" opacity="0.6"/>
      <line x1="30" y1="30" x2="70" y2="50" stroke="url(#gradPurpleGreen)" strokeWidth="1.5" opacity="0.4"/>
      <line x1="30" y1="50" x2="70" y2="30" stroke="url(#gradPurpleGreen)" strokeWidth="1.5" opacity="0.4"/>
      <line x1="30" y1="50" x2="70" y2="50" stroke="url(#gradPurpleGreen)" strokeWidth="2" opacity="0.6"/>
      <line x1="30" y1="50" x2="70" y2="70" stroke="url(#gradPurpleGreen)" strokeWidth="1.5" opacity="0.4"/>
      <line x1="30" y1="70" x2="70" y2="70" stroke="url(#gradPurpleGreen)" strokeWidth="2" opacity="0.6"/>
    </g>
    <defs>
      <linearGradient id="gradPurpleGreen" x1="30" y1="50" x2="70" y2="50" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#9D4EDD"/>
        <stop offset="100%" stopColor="#06FFA5"/>
      </linearGradient>
    </defs>
  </svg>
);

const ServerIcon = () => (
  <svg width="100" height="100" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
    <g className="animate-[fadeInUp_0.6s_ease-out_forwards] opacity-0 stagger-3">
      {/* Server rack */}
      <rect x="20" y="20" width="60" height="15" rx="2" fill="url(#gradOrange)" stroke="#FF6B35" strokeWidth="1.5"/>
      <rect x="20" y="40" width="60" height="15" rx="2" fill="url(#gradOrange)" stroke="#FF6B35" strokeWidth="1.5"/>
      <rect x="20" y="60" width="60" height="15" rx="2" fill="url(#gradOrange)" stroke="#FF6B35" strokeWidth="1.5"/>

      {/* Indicator lights */}
      <circle cx="28" cy="27.5" r="2" fill="#06FFA5" className="animate-[pulse-glow_1s_ease-in-out_infinite]"/>
      <circle cx="28" cy="47.5" r="2" fill="#00F5FF" className="animate-[pulse-glow_1s_ease-in-out_infinite_0.3s]"/>
      <circle cx="28" cy="67.5" r="2" fill="#9D4EDD" className="animate-[pulse-glow_1s_ease-in-out_infinite_0.6s]"/>

      {/* Data lines */}
      <line x1="35" y1="27.5" x2="72" y2="27.5" stroke="#FF8C5F" strokeWidth="1" opacity="0.6"/>
      <line x1="35" y1="47.5" x2="72" y2="47.5" stroke="#FF8C5F" strokeWidth="1" opacity="0.6"/>
      <line x1="35" y1="67.5" x2="72" y2="67.5" stroke="#FF8C5F" strokeWidth="1" opacity="0.6"/>
    </g>
    <defs>
      <linearGradient id="gradOrange" x1="20" y1="50" x2="80" y2="50" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stopColor="#8B3A1F"/>
        <stop offset="100%" stopColor="#FF6B35"/>
      </linearGradient>
    </defs>
  </svg>
);

const RacingCarSVG = ({ color = "#00B4D8" }) => (
  <svg width="60" height="30" viewBox="0 0 60 30" fill="none" xmlns="http://www.w3.org/2000/svg">
    {/* Simplified race car */}
    <path d="M5 15 L10 10 L25 10 L30 5 L45 5 L50 10 L55 15 L50 20 L45 25 L30 25 L25 20 L10 20 Z"
          fill={color} stroke="rgba(255,255,255,0.3)" strokeWidth="1"/>
    <circle cx="18" cy="22" r="4" fill="#1a1a1a" stroke={color} strokeWidth="1"/>
    <circle cx="42" cy="22" r="4" fill="#1a1a1a" stroke={color} strokeWidth="1"/>
    <rect x="32" y="8" width="8" height="6" fill="rgba(255,255,255,0.2)"/>
  </svg>
);

function App() {
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    setTimeout(() => setAnimated(true), 100);
  }, []);

  const maxGflops = 431;
  const implementations = [
    {
      name: 'jax-js',
      platform: 'GPU (WebGPU)',
      gflops: 431,
      relative: '1.00x',
      label: 'Baseline',
      color: '#06FFA5',
      barColor: 'from-green-400 to-emerald-500'
    },
    {
      name: 'jax-rs',
      platform: 'GPU (WebGPU)',
      gflops: 68.90,
      relative: '0.160x',
      color: '#FF6B35',
      barColor: 'from-orange-400 to-red-500'
    },
    {
      name: 'jax-rs',
      platform: 'CPU',
      gflops: 1.99,
      relative: '0.0046x',
      color: '#00B4D8',
      barColor: 'from-blue-400 to-cyan-500'
    },
    {
      name: 'jax-js',
      platform: 'WASM (Node.js)',
      gflops: 0.51,
      relative: '0.0012x',
      color: '#9D4EDD',
      barColor: 'from-purple-400 to-violet-500'
    },
  ];

  return (
    <div className="min-h-screen text-white noise-overlay gradient-mesh">
      {/* Header */}
      <header className="text-center py-16 px-4">
        <div className="animate-[fadeInUp_0.8s_ease-out_forwards] opacity-0">
          <h1 className="text-6xl md:text-7xl font-bold mb-4 text-glow">
            Conv2D Performance Showdown
          </h1>
          <p className="text-3xl md:text-4xl mb-2 bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            jax-rs vs. jax-js
          </p>
          <p className="text-xl text-blue-300 mono">Benchmark on Apple M2</p>
        </div>
      </header>

      {/* Benchmark Configuration */}
      <section className="max-w-7xl mx-auto px-4 mb-20">
        <h2 className="text-4xl font-bold mb-12 text-center text-cyan-400">Benchmark Configuration</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">

          <div className="glass-card rounded-2xl p-8 text-center transform hover:scale-105 transition-transform duration-300">
            <div className="flex justify-center mb-4">
              <TensorCubeIcon />
            </div>
            <h3 className="text-xl font-semibold mb-2 text-cyan-300">Input Shape</h3>
            <p className="text-2xl font-bold mono mb-1">1×64×256×256</p>
            <p className="text-sm text-gray-400">(NCHW)</p>
          </div>

          <div className="glass-card rounded-2xl p-8 text-center transform hover:scale-105 transition-transform duration-300">
            <div className="flex justify-center mb-4">
              <NeuralNetIcon />
            </div>
            <h3 className="text-xl font-semibold mb-2 text-purple-300">Kernel / Filters</h3>
            <p className="text-2xl font-bold mono mb-1">128 filters</p>
            <p className="text-sm text-gray-400">of size 3×3</p>
          </div>

          <div className="glass-card rounded-2xl p-8 text-center transform hover:scale-105 transition-transform duration-300">
            <div className="flex justify-center mb-4">
              <ServerIcon />
            </div>
            <h3 className="text-xl font-semibold mb-2 text-orange-300">9.66 Billion FLOPs</h3>
            <p className="text-lg mono mb-1">Per operation</p>
            <p className="text-sm text-gray-400">Floating-point ops calculated</p>
          </div>

        </div>
      </section>

      {/* Performance Results - Racing Track */}
      <section className="max-w-7xl mx-auto px-4 mb-20">
        <h2 className="text-4xl font-bold mb-8 text-center text-cyan-400">
          Performance Results <span className="text-2xl text-gray-400 mono">(GFLOP/s)</span>
        </h2>

        <div className="glass-card rounded-2xl p-8 overflow-hidden">
          {/* Racing track */}
          <div className="relative h-[500px] bg-gradient-to-r from-gray-900/50 to-gray-800/50 rounded-xl overflow-visible">

            {/* Track lanes */}
            <div className="absolute inset-0">
              {implementations.map((_, index) => (
                <div
                  key={index}
                  className="absolute left-0 right-0 border-b border-dashed border-gray-600/30"
                  style={{ top: `${(index + 1) * 20}%` }}
                />
              ))}
            </div>

            {/* Finish line */}
            <div className="absolute right-0 top-0 bottom-0 w-8 checkered-flag opacity-80" />

            {/* Race cars and data */}
            {implementations.map((impl, index) => {
              const position = Math.min((impl.gflops / maxGflops) * 95, 92);

              return (
                <div
                  key={`${impl.name}-${impl.platform}`}
                  className="absolute left-0 right-0"
                  style={{ top: `${index * 25 + 5}%` }}
                >
                  {/* Implementation label */}
                  <div className="absolute -left-2 top-0 text-left z-10 bg-gray-900/80 backdrop-blur-sm px-3 py-2 rounded-lg border border-gray-700">
                    <p className="font-bold text-sm" style={{ color: impl.color }}>
                      {impl.name}
                    </p>
                    <p className="text-xs text-gray-400">{impl.platform}</p>
                  </div>

                  {/* Race car */}
                  <div
                    className="absolute top-8 transition-all duration-[2500ms] ease-out"
                    style={{
                      left: animated ? `${position}%` : '0%',
                      filter: `drop-shadow(0 0 10px ${impl.color})`
                    }}
                  >
                    <RacingCarSVG color={impl.color} />
                  </div>

                  {/* Performance bar */}
                  <div className="absolute top-16 left-0 right-16 h-8 bg-gray-800/50 rounded-full overflow-hidden">
                    <div
                      className={`h-full bg-gradient-to-r ${impl.barColor} transition-all duration-[2500ms] ease-out relative`}
                      style={{ width: animated ? `${position}%` : '0%' }}
                    >
                      <div className="absolute inset-0 bg-white/20 animate-pulse" />
                    </div>
                  </div>

                  {/* Stats */}
                  <div
                    className="absolute top-16 transition-all duration-[2500ms] ease-out"
                    style={{ left: animated ? `${Math.max(position - 5, 2)}%` : '2%' }}
                  >
                    <div className="bg-gray-900/90 backdrop-blur-sm px-4 py-1 rounded-full border"
                         style={{ borderColor: impl.color }}>
                      <span className="font-bold text-lg mono" style={{ color: impl.color }}>
                        {impl.gflops} GFLOP/s
                      </span>
                      <span className="text-sm text-gray-400 ml-2">
                        {impl.relative}
                        {impl.label && <span className="text-xs ml-1">({impl.label})</span>}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}

            {/* Baseline note */}
            <div className="absolute right-12 bottom-4 text-right max-w-xs bg-gray-900/80 backdrop-blur-sm p-4 rounded-lg border border-cyan-500/30">
              <p className="font-bold text-cyan-400 mb-1">
                jax-js on WebGPU sets the performance baseline at ~431 GFLOP/s
              </p>
              <p className="text-sm text-gray-400">
                All other implementations are measured relative to this benchmark.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Analysis & Key Takeaways */}
      <section className="max-w-7xl mx-auto px-4 pb-20">
        <h2 className="text-4xl font-bold mb-12 text-center text-cyan-400">Analysis & Key Takeaways</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">

          <div className="glass-card animated-border rounded-2xl p-8 animate-[fadeInUp_0.6s_ease-out_forwards] opacity-0 stagger-1">
            <div className="flex items-start gap-4 mb-4">
              <div className="text-4xl">⚠️</div>
              <div>
                <h3 className="text-2xl font-bold mb-2 text-red-400">The GPU Gap</h3>
                <p className="text-3xl font-bold mono text-white mb-2">6.3× Slower</p>
              </div>
            </div>
            <p className="text-gray-300 leading-relaxed">
              The jax-rs GPU implementation is significantly slower than the jax-js WebGPU baseline.
            </p>
          </div>

          <div className="glass-card animated-border rounded-2xl p-8 animate-[fadeInUp_0.6s_ease-out_forwards] opacity-0 stagger-2">
            <div className="flex items-start gap-4 mb-4">
              <div className="text-4xl">✨</div>
              <div>
                <h3 className="text-2xl font-bold mb-2 text-blue-400">The CPU Advantage</h3>
                <p className="text-3xl font-bold mono text-white mb-2">3.9× Faster</p>
              </div>
            </div>
            <p className="text-gray-300 leading-relaxed">
              On CPU, jax-rs (native) outperforms the jax-js WASM backend significantly.
            </p>
          </div>

          <div className="glass-card animated-border rounded-2xl p-8 animate-[fadeInUp_0.6s_ease-out_forwards] opacity-0 stagger-3">
            <div className="flex items-start gap-4 mb-4">
              <div className="text-4xl">🔍</div>
              <div>
                <h3 className="text-2xl font-bold mb-2 text-purple-400">Potential Bottlenecks</h3>
              </div>
            </div>
            <ul className="space-y-2 text-gray-300">
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">▸</span>
                <span>Shader implementation needs optimization</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">▸</span>
                <span>Benchmark methodology refinement</span>
              </li>
              <li className="flex items-start">
                <span className="text-cyan-400 mr-2">▸</span>
                <span>WebGPU dispatch overhead reduction</span>
              </li>
            </ul>
          </div>

        </div>
      </section>

      {/* Footer credit */}
      <footer className="text-center pb-8 text-gray-500 text-sm mono">
        Generated with jax-rs benchmark suite
      </footer>
    </div>
  );
}

export default App;
