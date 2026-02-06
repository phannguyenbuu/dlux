export default function ToolbarStatus({ exportMsg, error }) {
  return (
    <>
      <div className="toolbar-spacer" />
      {exportMsg ? <div className="meta">{exportMsg}</div> : null}
      {error ? <div className="error">{error}</div> : null}
    </>
  );
}
