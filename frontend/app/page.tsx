export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-8 bg-zinc-950 text-zinc-100 selection:bg-zinc-800">
      <div className="max-w-2xl w-full text-center space-y-8">
        <div className="inline-flex items-center gap-2 px-3 py-1 text-xs font-mono tracking-wider uppercase border border-zinc-800 rounded-full text-zinc-400 bg-zinc-900/50">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          Project initialization in progress
        </div>

        <div className="space-y-4">
          <h1 className="text-5xl sm:text-6xl font-bold tracking-tight text-zinc-50 font-mono">
            PRISM
          </h1>
          <p className="text-xl sm:text-2xl font-medium text-zinc-300">
            Probing the Evolution of Visual Representations
          </p>
        </div>

        <blockquote className="text-base sm:text-lg italic text-zinc-400 border-y border-zinc-800/80 py-4 px-6">
          &ldquo;One visual problem. Multiple learning paradigms. A deeper understanding of how machines learn to see.&rdquo;
        </blockquote>

        <div className="pt-4 text-xs font-mono text-zinc-500">
          Backend foundation initialized &bull; Research contracts active &bull; UI dashboard in development
        </div>
      </div>
    </main>
  );
}
