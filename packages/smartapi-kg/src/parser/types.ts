export interface XTranslatorObject {
  component: string;
  team: string[];
  infores?: string;
}

interface SmartAIPInfoObject {
  title: string;
  version: string;
  description?: string;
  "x-translator"?: XTranslatorObject;
  [propName: string]: any;
}

interface SmartAPIServerObject {
  url: string;
  description?: string;
  variables?: any;
  "x-location"?: string;
  "x-maturity": any;
}

export interface SmartAPIReferenceObject {
  $ref?: string;
}

export interface SmartAPIParameterObject {
  name: string;
  in: "query" | "header" | "path" | "cookie";
  description?: string;
  required?: boolean;
  deprecated?: boolean;
  allowEmptyValue?: boolean;
}

interface SmartAPIMediaTypeObject {
  example?: any;
  schema?: any;
  examples?: any;
  encoding?: any;
}

interface SmartAPIRequestBodyObject {
  description?: string;
  content: SmartAPIMediaTypeObject[];
  required?: boolean;
}

interface SmartAPIResponsesObject {
  description: string;
  headers: any;
  content: any;
  links: any;
}

export interface SmartAPIOperationObject {
  tags?: string[];
  summary?: string;
  description?: string;
  externalDocs?: any;
  operationID?: string;
  parameters?: SmartAPIParameterObject[];
  requestBody?: SmartAPIRequestBodyObject | SmartAPIReferenceObject;
  responses?: SmartAPIResponsesObject;
  callbacks?: any;
  deprecated?: boolean;
  security?: any;
  servers?: SmartAPIServerObject[];
  "x-bte-kgs-operations"?: SmartAPIReferenceObject;
  [propName: string]: any;
}

export interface SmartAPIPathItemObject {
  $ref?: string;
  summary?: string;
  description?: string;
  get?: SmartAPIOperationObject;
  [propName: string]: any;
  put?: SmartAPIOperationObject;
  post?: SmartAPIOperationObject;
  delete?: SmartAPIOperationObject;
  options?: SmartAPIOperationObject;
  head?: SmartAPIOperationObject;
  patch?: SmartAPIOperationObject;
  trace?: SmartAPIOperationObject;
  servers?: SmartAPIServerObject[];
  parameters?: SmartAPIParameterObject[] | SmartAPIReferenceObject[];
}

export interface SmartAPIPathsObject {
  [path: string]: SmartAPIPathItemObject;
}

interface SmartAPITagObject {
  name: string;
  description?: string;
  externalDocs?: any;
  "x-id"?: string;
}

export interface SmartAPIComponentObject {
  [propName: string]: any;
}

export interface XBTEKGSOperationBioEntityObject {
  id: string;
  semantic: string;
}

export interface XBTEKGSOperationObject {
  inputs: XBTEKGSOperationBioEntityObject[];
  outputs: XBTEKGSOperationBioEntityObject[];
  predicate: string;
  qualifiers?: { [qualifierType: string]: string };
  source?: string;
  parameters?: XBTEParametersObject;
  requestBody?: any;
  requestBodyType?: string;
  supportBatch?: boolean;
  useTemplating?: boolean;
  inputSeparator?: string;
  response_mapping?: SmartAPIReferenceObject;
  responseMapping?: SmartAPIReferenceObject;
  templateInputs?: object;
  batchSize?: number;
}

export interface SmartAPISpec {
  openapi: string;
  info: SmartAIPInfoObject;
  servers?: SmartAPIServerObject[];
  paths: SmartAPIPathsObject;
  tags?: SmartAPITagObject[];
  security?: any;
  externalDocs?: any;
  components?: SmartAPIComponentObject;
  _id?: string;
  _meta?: string;
}

export interface SmartAPIRegistryRecordObject {
  id: string;
  meta: any;
}

export interface XTRAPIObject {
  batch_size_limit?: number;
  rate_limit?: number;
}

export interface KGQualifiersObject {
  [qualifierType: string]: string | string[];
}

export interface KGAssociationObject {
  input_id?: string | string[];
  input_type: string;
  output_id?: string | string[];
  output_type: string;
  predicate: string;
  source?: string;
  api_name?: string;
  component?: string;
  apiIsPrimaryKnowledgeSource?: boolean;
  smartapi?: SmartAPIRegistryRecordObject;
  "x-translator"?: any;
  "x-trapi"?: XTRAPIObject;
  qualifiers?: KGQualifiersObject;
}

export interface QueryOperationInterface {
  path: string;
  method: string;
  server: string;
  tags: string[];
  path_params: string[];
  params: XBTEParametersObject;
  request_body: any;
  requestBodyType?: string;
  supportBatch: boolean;
  batchSize?: number;
  inputSeparator: string;
  useTemplating?: boolean;
  templateInputs?: any;
}

export interface SmartAPIKGOperationObject {
  association: KGAssociationObject;
  query_operation?: QueryOperationInterface;
  response_mapping?: any;
  id?: string;
  tags?: string[];
}

export interface ParsedAPIMetadataObject {
  title: string;
  tags: string[];
  url: string;
  paths: string[];
  components: any;
  smartapi: SmartAPIRegistryRecordObject;
  "x-translator": XTranslatorObject;
  operations: SmartAPIKGOperationObject[];
}

export interface XBTEParametersObject {
  [key: string]: string | number;
}

export interface APIClass {
  smartapiDoc: SmartAPISpec;
  metadata: ParsedAPIMetadataObject;
}
