export interface TemplatedInput {
  queryInputs: string | string[];
  [additionalProperties: string]: string | string[];
}
