import ToolbarStatus from "./ToolbarStatus";

export default function TopBar({
  loadScene,
  exportPdf,
  setSimProgress,
  setSimPlaying,
  setShowSim,
  exportMsg,
  error,
}) {
  return (
    <div className="panel toolbar">
      <button onClick={loadScene}>Load</button>
      <button onClick={exportPdf}>Export PDF</button>
      <button
        onClick={() => {
          setSimProgress(0);
          setSimPlaying(false);
          setShowSim(true);
        }}
      >
        Simulate
      </button>
      <ToolbarStatus exportMsg={exportMsg} error={error} />
    </div>
  );
}
