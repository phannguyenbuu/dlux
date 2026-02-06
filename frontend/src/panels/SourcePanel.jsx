import SourceHeader from "./SourceHeader";
import SourceStage from "./SourceStage";
import LeftDebug from "./LeftDebug";

export default function SourcePanel(props) {
  const {
    sceneLoading,
    leftRef,
    scene,
    ...rest
  } = props;

  return (
    <div className={`left ${sceneLoading ? "is-loading" : ""}`} ref={leftRef}>
      <SourceHeader scene={scene} {...rest} />
      <SourceStage scene={scene} {...rest} />
      {sceneLoading ? <div className="loading-overlay">Loading...</div> : null}
      <LeftDebug scene={scene} />
    </div>
  );
}
