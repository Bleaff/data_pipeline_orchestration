import { describe, expect, it } from "vitest";
import { toPipelineYaml } from "../yamlExport";

describe("toPipelineYaml", () => {
  it("renders a nodes list under a top-level nodes key", () => {
    const yaml = toPipelineYaml([
      { id: "reader", type: "FolderImageNode", outputs: ["saver"], folder_path: "/data/in" },
      { id: "saver", type: "SaveImageNode", outputs: [], save_dir: "/data/out" },
    ]);

    expect(yaml).toContain("nodes:");
    expect(yaml).toContain("id: reader");
    expect(yaml).toContain("type: FolderImageNode");
    expect(yaml).toContain("folder_path: /data/in");
    expect(yaml).toContain("id: saver");
  });

  it("renders an empty node list as an empty array, not omitted", () => {
    const yaml = toPipelineYaml([]);

    expect(yaml.trim()).toBe("nodes: []");
  });
});
