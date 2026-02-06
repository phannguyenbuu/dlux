export default function ExportModal({
  exportPdfLoading,
  exportPdfInfo,
  exportHtmlInfo,
  setExportPdfInfo,
}) {
  if (!exportPdfLoading && !exportPdfInfo) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <div className="modal-title">
          {exportPdfLoading ? "Creating PDF..." : "Successful created PDF !"}
        </div>
        {!exportPdfLoading && exportPdfInfo ? (
          <div className="modal-actions">
            <button className="btn ghost" onClick={() => setExportPdfInfo(null)}>
              Cancel
            </button>
            <button
              className="btn"
              onClick={() => {
                window.location = `/api/download_pdf?name=${encodeURIComponent(
                  exportPdfInfo.name
                )}`;
                setExportPdfInfo(null);
              }}
            >
              Download PDF
            </button>
            {exportHtmlInfo.map((name) => (
              <button
                key={name}
                className="btn"
                onClick={() => {
                  window.location = `/api/download_html?name=${encodeURIComponent(name)}`;
                }}
              >
                Download HTML
              </button>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}
