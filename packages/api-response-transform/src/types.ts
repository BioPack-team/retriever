import { JSONDoc } from "./json_transform/types";

interface KGAssociationObject {
  input_id?: string;
  input_type: string;
  output_id?: string;
  output_type: string;
  predicate: string;
  source?: string;
  api_name?: string;
}

interface XBTEParametersObject {
  [key: string]: string | number;
}

interface TransformerObject {
  wrap?: string;
  pair?: string;
}

interface TransformerSet {
  [transformerPattern: string]: TransformerObject;
}

interface QueryOperationInterface {
  path: string;
  method: string;
  server: string;
  tags: string[];
  path_params: string[];
  params: XBTEParametersObject;
  request_body: object;
  supportBatch: boolean;
  inputSeparator: string;
  transformer: TransformerSet;
}

interface SmartAPIKGOperationObject {
  association: KGAssociationObject;
  query_operation?: QueryOperationInterface;
  response_mapping?: any;
  id?: string;
  tags?: string[];
}

interface TemplatedInput {
  queryInputs: string | string[];
  [additionalAttributes: string]: string | string[];
}

export interface BTEKGOperationObject extends SmartAPIKGOperationObject {
  input: string | string[] | TemplatedInput;
  reasoner_edge?: any;
  filter?: string;
  original_input?: object;
  input_resolved_identifiers?: object;
}

export interface BTEQueryObject {
  response: JSONDoc | JSONDoc[] | { hits: JSONDoc[] };
  edge: BTEKGOperationObject;
}

export interface JQVariable {
  name: string;
  value: string;
}
