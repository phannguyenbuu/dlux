export default function createSimHandlers({
  scene,
  packedLabels,
  labelFontFamily,
  labelFontSize,
  simVideoLoading,
  setSimVideoLoading,
  setError,
  simPlaying,
  simProgress,
  setSimProgress,
  setSimPlaying,
}) {
  const handleSimVideoDownload = async () => {
    if (simVideoLoading || !scene) return;
    setSimVideoLoading(true);
    try {
      const res = await fetch("/api/export_sim_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          scene,
          packedLabels,
          fontName: labelFontFamily,
          fontSize: labelFontSize,
        }),
      });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `export failed: ${res.status}`);
      }
      const data = await res.json().catch(() => ({}));
      if (data?.name) {
        window.location = `/api/download_sim_video?name=${encodeURIComponent(data.name)}`;
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setSimVideoLoading(false);
    }
  };

  const handleSimPlayToggle = () => {
    if (!simPlaying && simProgress >= 1) {
      setSimProgress(0);
    }
    setSimPlaying((v) => !v);
  };

  return { handleSimVideoDownload, handleSimPlayToggle };
}
