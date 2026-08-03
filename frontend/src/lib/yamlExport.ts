/** Renders a builder graph as the YAML a user would hand-write for `neudc.entrypoints.main` (#24). */

import { dump } from "js-yaml";
import type { NodeConfig } from "./api";

export function toPipelineYaml(nodes: NodeConfig[]): string {
  return dump({ nodes }, { noRefs: true, lineWidth: -1 });
}
