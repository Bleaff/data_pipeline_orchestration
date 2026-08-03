import type { NodeTypeSpec } from "@/lib/api";
import styles from "./NodePalette.module.css";

export function NodePalette({
  nodeTypes,
  onAddNode,
}: {
  nodeTypes: NodeTypeSpec[];
  onAddNode: (nodeType: string) => void;
}) {
  return (
    <div className={styles.palette}>
      <h2 className={styles.heading}>Add a node</h2>
      <ul className={styles.list}>
        {nodeTypes.map((nt) => (
          <li key={nt.type}>
            <button type="button" className={styles.item} onClick={() => onAddNode(nt.type)} title={nt.description}>
              <span className={styles.type}>{nt.type}</span>
              <span className={styles.runsAs}>{nt.runs_as}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
