import type { NodeFieldSpec, NodeTypeSpec } from "@/lib/api";
import styles from "./NodeConfigForm.module.css";

interface FieldInputProps {
  field: NodeFieldSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}

function FieldInput({ field, value, onChange }: FieldInputProps) {
  if (field.kind === "boolean") {
    return (
      <input
        type="checkbox"
        checked={Boolean(value ?? field.default ?? false)}
        onChange={(e) => onChange(e.target.checked)}
      />
    );
  }

  if (field.kind === "enum") {
    return (
      <select value={(value as string) ?? (field.default as string) ?? ""} onChange={(e) => onChange(e.target.value)}>
        <option value="" disabled>
          select…
        </option>
        {field.options?.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    );
  }

  if (field.kind === "integer" || field.kind === "number") {
    return (
      <input
        type="number"
        step={field.kind === "integer" ? 1 : "any"}
        value={value === undefined || value === null ? "" : String(value)}
        placeholder={field.default === null ? "" : String(field.default)}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") {
            onChange(undefined);
            return;
          }
          onChange(field.kind === "integer" ? parseInt(raw, 10) : parseFloat(raw));
        }}
      />
    );
  }

  return (
    <input
      type="text"
      value={(value as string) ?? ""}
      placeholder={field.default === null ? "" : String(field.default)}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function ModelConfigInput({
  nodeType,
  value,
  onChange,
}: {
  nodeType: NodeTypeSpec;
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const modelConfig = (value as Record<string, unknown>) ?? {};
  const selectedType = modelConfig.type as string | undefined;
  const modelType = nodeType.model_types?.find((mt) => mt.type === selectedType);

  return (
    <div className={styles.modelConfig}>
      <label className={styles.fieldRow}>
        <span>
          model type<span className={styles.required}> *</span>
        </span>
        <select value={selectedType ?? ""} onChange={(e) => onChange({ type: e.target.value })}>
          <option value="" disabled>
            select…
          </option>
          {nodeType.model_types?.map((mt) => (
            <option key={mt.type} value={mt.type}>
              {mt.type}
            </option>
          ))}
        </select>
      </label>
      {modelType?.fields.map((subField) => (
        <label key={subField.name} className={styles.fieldRow} title={subField.description}>
          <span>
            {subField.name}
            {subField.required && <span className={styles.required}> *</span>}
          </span>
          <FieldInput
            field={subField}
            value={modelConfig[subField.name]}
            onChange={(v) => onChange({ ...modelConfig, [subField.name]: v })}
          />
        </label>
      ))}
    </div>
  );
}

export function NodeConfigForm({
  nodeType,
  config,
  onFieldChange,
}: {
  nodeType: NodeTypeSpec;
  config: Record<string, unknown>;
  onFieldChange: (key: string, value: unknown) => void;
}) {
  if (nodeType.fields.length === 0) {
    return <p className={styles.empty}>This node type has no configurable fields.</p>;
  }

  return (
    <div className={styles.form}>
      {nodeType.fields.map((field) =>
        field.kind === "model_config" ? (
          <ModelConfigInput
            key={field.name}
            nodeType={nodeType}
            value={config[field.name]}
            onChange={(v) => onFieldChange(field.name, v)}
          />
        ) : (
          <label key={field.name} className={styles.fieldRow} title={field.description}>
            <span>
              {field.name}
              {field.required && <span className={styles.required}> *</span>}
            </span>
            <FieldInput field={field} value={config[field.name]} onChange={(v) => onFieldChange(field.name, v)} />
          </label>
        ),
      )}
    </div>
  );
}
