import { useEffect } from "react";

export default function useSimTicker(simPlaying, setSimPlaying, simTotalSeconds, setSimProgress) {
  useEffect(() => {
    if (!simPlaying) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now) => {
      const dt = (now - last) / 1000;
      last = now;
      setSimProgress((p) => {
        const next = Math.min(1, p + dt / simTotalSeconds);
        if (next >= 1) setSimPlaying(false);
        return next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [simPlaying, simTotalSeconds, setSimProgress, setSimPlaying]);
}
