import { useState } from "react";

export default function useEditState() {
  const [edgeMode, setEdgeMode] = useState(false);
  const [addNodeMode, setAddNodeMode] = useState(false);
  const [deleteEdgeMode, setDeleteEdgeMode] = useState(false);
  const [edgeCandidate, setEdgeCandidate] = useState(null);
  const [deleteEdgeCandidate, setDeleteEdgeCandidate] = useState(null);

  return {
    edgeMode,
    setEdgeMode,
    addNodeMode,
    setAddNodeMode,
    deleteEdgeMode,
    setDeleteEdgeMode,
    edgeCandidate,
    setEdgeCandidate,
    deleteEdgeCandidate,
    setDeleteEdgeCandidate,
  };
}
