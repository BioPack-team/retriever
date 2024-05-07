import { APIEdge } from "@retriever/graph";
import { JSONDoc } from "./json_transform/types";
import { XBTEKGSOperationObject } from "@retriever/smartapi-kg";

export interface TransformerObject {
  wrap?: string;
  pair?: string;
}

export interface TransformerSet {
  [transformerPattern: string]: TransformerObject;
}

export interface TemplatedInput {
  queryInputs: string | string[];
  [additionalAttributes: string]: string | string[];
}

export interface RetrieverQueryObject {
  response: JSONDoc | JSONDoc[] | { hits: JSONDoc[] };
  edge: APIEdge;
}

export interface JQVariable {
  name: string;
  value: string;
}
